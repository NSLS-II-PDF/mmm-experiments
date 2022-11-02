import logging
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from constrainedmf.nmf.models import NMF
from constrainedmf.nmf.utils import iterative_nmf

from mmm_experiments.viz.plotting import independent_waterfall, min_max_normalize

logger = logging.getLogger(name="mmm.ml_mixins")


class CMFMixin:
    """Mixin for Agents to have an agent perform constrained matrix factorization on the incoming data.
    Sorts the dataset  by independent variable (position) for easy visualization.
    """

    AVAILABLE_ASK_MODES = {"autoconstrained", "unconstrained"}

    def __init__(self, *, num_components: int, ask_mode: str, **kwargs):
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
        try:
            origin = self.measurement_origin
        except AttributeError:
            self.measurement_origin = 0.0
            origin = self.measurement_origin
            logger.warning(f"No origin detected for CMF agent {self}. Setting to 0. ")
        relative_position = position - origin
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
            components=self.current_components,
            weights=self.current_weights,
            residuals=self.current_residuals,
            sorted_dataset=self.sorted_dataset,
            sorted_positions=self.sorted_positions,
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

    def core_component_positions(self, batch_size: Optional[int] = None):
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
