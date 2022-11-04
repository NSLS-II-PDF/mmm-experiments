import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from constrainedmf.nmf.models import NMF
from constrainedmf.nmf.utils import iterative_nmf
from scipy.interpolate import interp1d

from mmm_experiments.viz.plotting import independent_waterfall, min_max_normalize

logger = logging.getLogger(name="mmm.ml_mixins")


class CMFMixin:
    AVAILABLE_ASK_MODES = {"autoconstrained", "unconstrained"}

    def __init__(self, *, num_components: int, ask_mode: str, **kwargs):
        """
        Constrained Matrix Factorization mixin for agents.base.Agent children

        Sorts the dataset  by independent variable (position) for easy visualization.

        Parameters
        ----------
        num_components : int
            Number of components to initially use. Can be updated via `update_num_components`
            through the agent Kafka interface.
        ask_mode : str
            One of AVAILABLE_ASK_MODES.
            autoconstrained : iteratively perform CMF by adding constraints drawn from the dataset
            unconstrained : plain old NMF
        kwargs :
            Keyword arguments to pass down the MRO. Likely to PDFAgent, BMMAgent or the like.
        """
        super().__init__(**kwargs)
        self.independent_cache = []
        self.dependent_cache = []
        self.current_weights = None
        self.current_components = None
        self.current_residuals = None
        self.sorted_positions, self.sorted_dataset = None, None
        self._current_num_components = num_components
        self._ask_mode = ask_mode.lower() if ask_mode.lower() in self.AVAILABLE_ASK_MODES else "unconstrained"

    @property
    def name(self):
        return "cmf"

    @property
    def num_components(self):
        return self._current_num_components

    @num_components.setter
    def num_components(self, n: int):
        if n < 2:
            logger.warning(f"Number of proposed components {n} less than minimum 2. No change made.")
        else:
            self._current_num_components = n

    def update_num_components(self, num_components: int):
        """Convenience method exposed to plans for updating via kafka."""
        self.num_components = num_components

    def tell(self, position, y) -> dict:
        relative_position = position - self.measurement_origin
        self.independent_cache.append(relative_position)
        self.dependent_cache.append(np.atleast_2d(y))
        self.sorted_positions, self.sorted_dataset = zip(
            *sorted(zip(self.independent_cache, self.dependent_cache), key=lambda a: a[0])
        )

        return dict(
            position=[position],
            rel_position=[relative_position],
            data=[y],
            cache_len=[len(self.independent_cache)],
        )

    def _calculate_nmf(self, tol=1e-8, max_iter=1000):
        data = torch.tensor(np.concatenate(self.sorted_dataset, axis=0), dtype=torch.float)
        nmf = NMF(data.shape, self.num_components)
        nmf.fit(data, beta=2, tol=tol, max_iter=max_iter)
        self._update_current(nmf, data)

    def _calculate_autoconstrained_nmf(self, tol=1e-8, max_iter=500):
        data = torch.tensor(np.concatenate(self.sorted_dataset, axis=0), dtype=torch.float)
        models = iterative_nmf(NMF, data, n_components=self.num_components, beta=2, tol=tol, max_iter=max_iter)
        self._update_current(models[-1], data)

    def _update_current(self, nmf, dataset):
        self.current_weights = nmf.W.data.cpu().numpy()
        self.current_components = nmf.H.data.cpu().numpy()
        self.current_residuals = nmf.reconstruct(nmf.H, nmf.W).detach().cpu().numpy() - dataset

    def report(self, auto_constrained=False, **kwargs):
        if auto_constrained:
            self._calculate_autoconstrained_nmf(**kwargs)
        else:
            self._calculate_nmf(**kwargs)
        return dict(
            num_components=[self.num_components],
            components=[self.current_components],
            weights=[self.current_weights],
            residuals=[self.current_residuals],
            sorted_dataset=[self.sorted_dataset],
            sorted_positions=[self.sorted_positions],
        )

    @property
    def ask_mode(self):
        return self._ask_mode

    @ask_mode.setter
    def ask_mode(self, mode: str):
        if mode.lower() not in self.AVAILABLE_ASK_MODES:
            logger.warning(f"Mode: {mode} is not an available ask mode for a CMF mixin. No change made.")
        else:
            self._ask_mode = mode.lower()

    def derived_component_positions(self, batch_size: Optional[int] = None):
        """Generate an ask that returns a batch of locations nearest the derived components.
        Will optionally trim that set of locations to a batch size in an ordered list.
        """
        if self.ask_mode == "autoconstrained":
            self._calculate_autoconstrained_nmf()
        elif self.ask_mode == "unconstrained":
            self._calculate_nmf()
        else:
            logger.warning(f"Unrecognized CMF mode {self.ask_mode}. I don't know how you got here...")
        points = []
        batch_size = batch_size or self.num_components
        for i, component in enumerate(self.current_components.T):
            if i == batch_size:
                break
            points.append(self.sorted_positions[(self.sorted_dataset - component).sum(axis=-1).argmin()])
        return points

    def ask(self, batch_size: Optional[int] = None) -> Tuple[dict, Sequence]:
        points = self.derived_component_positions(batch_size)
        doc = dict(
            num_components=[self.num_components],
            components=[self.current_components],
            weights=[self.current_weights],
            residuals=[self.current_residuals],
            sorted_dataset=[self.sorted_dataset],
            sorted_positions=[self.sorted_positions],
            ask_mode=[self.ask_mode],
            batch_size=[batch_size],
            suggestions=[points],
        )
        return doc, points

    @staticmethod
    def plot_from_report(doc):
        cmap = mpl.cm.get_cmap("tab10")
        num_components = doc["num_components"][0]
        norm = mpl.colors.Normalize(vmin=0, vmax=num_components)
        fig, axes = plt.subplots(1, 3)
        component_ax = axes[0]
        weight_ax = axes[1]
        residual_ax = axes[2]

        # Components plot
        for i, component in enumerate(doc["components"][0]):
            component_ax.plot(np.arange(len(component)), component, color=cmap(norm(i)))
        component_ax.set_xlabel("Data index")
        component_ax.set_ylabel("Stacked components")

        # Weights plot
        for i, weight in enumerate(doc["weights"][0].T):
            component_ax.plot(doc["sorted_positions"], weight, color=cmap(norm(i)), label=f"Component {i+1}")
        weight_ax.set_xlabel("Relative position")
        weight_ax.set_ylabel("Weights")
        weight_ax.legend()

        # Residual plot
        residuals = doc["residuals"][0]
        alpha = min_max_normalize(np.mean(residuals ** 2, axis=1))
        independent_waterfall(
            residual_ax, doc["sorted_positions"], np.arange(len(residuals)), residuals, alphas=alpha
        )
        residual_ax.set_xlabel("Data index")
        residual_ax.set_ylabel("Independent Var")
        return fig, axes


class XCAMixin:
    from databroker.client import BlueskyRun

    def __init__(
        self,
        *,
        model_checkpoint: Union[str, Path],
        model_qspace: np.ndarray,
        device: Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        **kwargs,
    ):
        from xca.ml.torch.cnn import EnsembleCNN
        from xca.ml.torch.training import JointVAEClassifierModule
        from xca.ml.torch.vae import VAE, CNNDecoder, CNNEncoder

        super().__init__(**kwargs)
        self.independent_cache = []
        self.dependent_cache = []
        self.q_space = model_qspace
        self.checkpoint = torch.load(str(model_checkpoint))
        self.device = torch.device(device)

        # Load lightning module
        checkpoint = torch.load(str("last.ckpt"), map_location=self.device)
        self.cnn = EnsembleCNN(**checkpoint["hyper_parameters"]["classifier_hparams"])
        self.encoder = CNNEncoder(**checkpoint["hyper_parameters"]["encoder_hparams"])
        self.decoder = CNNDecoder(**checkpoint["hyper_parameters"]["decoder_hparams"])
        self.vae = VAE(self.encoder, self.decoder)
        self.pl_module = JointVAEClassifierModule.load_from_checkpoint(
            model_checkpoint, classification_model=self.cnn, vae_model=self.vae
        )
        self.pl_module.eval()
        self.recon_loss = torch.nn.MSELoss(reduction="none")

        # Prediction caches
        self.probability_cache = []
        self.latent_cache = []
        self.reconstruction_cache = []
        self.loss_cache = []
        self.shannon_cache = []

    def unpack_run(self, run: BlueskyRun):
        """
        Overrides standard unpack run to extract extra metadata
        and interpolates intensity onto the standard Q space.
        """
        x = np.array(run.primary.data["chi_Q"][0])
        y = np.array(run.primary.data["chi_I"][0])
        f = interp1d(x, y, fill_value=0.0, bounds_error=False)
        spectra = f(self.q_space)
        spectra = (spectra - np.min(spectra)) / (np.max(spectra) - np.min(spectra))
        return run.start["Grid_X"]["Grid_X"]["value"], spectra

    def tell(self, position, intensity):
        rel_position = position - self.measurement_origin

        with torch.no_grad():
            x = torch.tensor(intensity.ravel(), dtype=torch.float, device=self.device)[None, None, :]
            res = self.pl_module(x)
            logits = res["y_pred"]
            prob = torch.sigmoid(logits)
            prediction_prob = prob.div_(prob.sum(dim=1, keepdims=True)).cpu().numpy().ravel()
            latent = res["mu"].cpu().numpy().ravel()
            reconstruction = res["x_recon"][0].cpu().numpy().ravel()
            loss = float(self.recon_loss(x, res["x_recon"][0]).sum(dim=(-1, -2)).cpu().numpy())
            shannon = np.sum(-1 * prediction_prob * np.log(prediction_prob))

        doc = dict(
            position=[position],
            rel_position=[rel_position],
            intensity=[intensity],
            prediction_prob=[prediction_prob],
            latent_rep=[latent],
            reconstruction=[reconstruction],
            shannon=[shannon],
            loss=[loss],
        )

        # Update caches for report with raveled arrays to be stacked
        self.independent_cache.append(rel_position)
        self.dependent_cache.append(intensity)
        self.probability_cache.append(prediction_prob)
        self.latent_cache.append(latent)
        self.reconstruction_cache.append(reconstruction)
        self.loss_cache.append(loss)
        self.shannon_cache.append(shannon)

        return doc

    def report(self):
        return dict(
            absolute_positions=[[pos + self.measurement_origin for pos in self.independent_cache]],  # List[float]
            relative_positions=[self.independent_cache],  # List[float]
            resampled_normalized_patterns=[
                np.stack(self.dependent_cache)
            ],  # [# of measurements, # of new q_points in pattern resampled for model expectation]
            prediction_probabilities=[np.stack(self.probability_cache)],  # [# of measurements, # of phases]
            latent_space_positions=[np.stack(self.latent_cache)],  # [# of measurements, latent dim]
            reconstructed_patterns=[np.stack(self.reconstruction_cache)],  # [# of measurements, # of new q_points]
            reconstruction_losses=[self.loss_cache],  # List[float]
            shannon_entropy=[self.shannon_cache],  # List[float]
        )

    def ask(self):
        raise NotImplementedError("Basic XCA Mixin is a passive agent.")
