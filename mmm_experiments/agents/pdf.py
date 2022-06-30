from collections import namedtuple
from pathlib import Path
from typing import Literal, Tuple, Union

import torch
from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from databroker.client import BlueskyRun
from gpytorch.mlls import ExactMarginalLogLikelihood
from xca.ml.torch.cnn import EnsembleCNN
from xca.ml.torch.vae import VAE, CNNDecoder, CNNEncoder

from .base import Agent

DATA_KEY = "pe1c_image"  # TODO: Change in accordance with analysis broker

Representation = namedtuple("Representation", "probabilities shannon_entropy reconstruction_loss")


class PDFAgent(Agent):
    server_host = "qserver1.nsls2.bnl.gov:60610"
    measurement_plan_name = "mv_and_jog"  # This plan does not exist yet

    def __init__(
        self,
        sample_origin: Tuple[float, float],
        model_checkpoint: Union[str, Path],
        device: Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        relative_bounds: Tuple[float, float],
        ucb_beta=0.1,
    ):
        """

        Parameters
        ----------
        sample_origin : Tuple[float, float]
            Decided origin of sample in raw motor coordinates
        model_checkpoint : str, Path
            Checkpoint path for models
        device : str
            Torch device to keep models. This is where both Deep and GP models will be stored.
        relative_bounds : Tuple[float, float]
            Relative bounds for the sample measurement. For a 10 cm sample, this would be something like (1, 99).
        ucb_beta : float
            Value for exporative weighting in upper confidence bound acquisition function
        """
        super().__init__(beamline_tla="pdf")
        self.sample_origin = sample_origin
        self.checkpoint = torch.load(str(model_checkpoint))
        self.device = torch.device(device)
        self.bounds = torch.tensor(relative_bounds).to(self.device)
        self.cnn = EnsembleCNN(**self.checkpoint["hyper_parameters"]["classifier_hparams"]).eval()
        encoder = CNNEncoder(**self.checkpoint["hyper_parameters"]["encoder_hparams"])
        decoder = CNNDecoder(**self.checkpoint["hyper_parameters"]["decoder_hparams"])
        self.vae = VAE(encoder, decoder).eval()
        self.reconstruction_loss = torch.nn.MSELoss()
        self.position_cache = []
        self.representation_cache = []
        self.beta = ucb_beta

    def measurement_plan_args(self, x_position) -> list:
        """Plan to be writen than moves to an x_position then jogs up and down relatively in y"""
        return [5, "Grid_X", x_position, "Grid_Y", -0.1, 0.1]

    @staticmethod
    def unpack_run(run: BlueskyRun):
        """"""
        # TODO: Review and revise as correct plan and md is written
        return run.start["Grid_X"], run.primary.read()[DATA_KEY]

    def tell(self, position, intensity):
        """
        1. Get relative position from raw motor position.
        2. From I(Q) intensity use a ensemble model to predict phase.
        3. Retain the shannon entropy of that model as a proxy for novelty.
        4. Use a complementary VAE to calculate reconstruction loss as a proxy for novelty.
        5. The caches of the independent and these dependent variables are then appended.

        A document with all key metadata is returned.

        Parameters
        ----------
        position : float
            Absolute motor position
        intensity : array
            Intensity from integrated I(Q)

        Returns
        -------

        """
        rel_position = position - self.sample_origin[0]
        with torch.no_grad():
            x = torch.tensor(intensity, dtype=torch.float, device=self.device)
            if len(x.shape) == 1:
                x = x[None, ...]
            elif x.shape[0] > 1:
                x = x.squeeze()[None, ...]
            prediction_prob = self.cnn(x)
            reconstruction = self.vae(x)
            shannon = torch.sum(prediction_prob)
            loss = self.reconstruction_loss(x, reconstruction)
        self.position_cache.append(rel_position)
        self.representation_cache.append(Representation(prediction_prob, shannon, loss))

        doc = dict(
            position=position,
            rel_position=rel_position,
            intensity=intensity,
            prediction_prob=prediction_prob.to("cpu").numpy(),
            reconstruction=x.to("cpu").numpy(),
            shannon=float(shannon.to("cpu")),
            loss=float(loss.to("cpu")),
        )
        return doc

    def ask(self, batch_size=1):
        """
        Train a single task Gaussian process to predict reconstruction loss as a function of relative position.
        Optimize over this GP using an upper confidence bound acquisition function to find the areas of "worst"
        reconstruction, i.e. "most novelty".

        Parameters
        ----------
        batch_size : int
            Number of new points to measure

        Returns
        -------
        doc : dict
            key metadata from the ask approach
        next_points : Sequence
            Sequence of independent variables of length batch size

        """
        train_x = torch.tensor(self.position_cache, dtype=torch.float).to(self.device)
        train_y = torch.tensor([y.reconstruction_loss for y in self.representation_cache], dtype=torch.float).to(
            self.device
        )
        gp = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_model(mll)
        if batch_size == 1:
            ucb = UpperConfidenceBound(gp, beta=self.beta)
        else:
            ucb = qUpperConfidenceBound(gp, beta=self.beta)
        next_points, acq_value = optimize_acqf(
            ucb,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=5,
            raw_samples=20,
        )
        if batch_size == 1:
            next_points = [float(next_points.to("cpu"))]
        else:
            next_points = [float(x.to("cpu")) for x in next_points]

        doc = dict(
            batch_size=batch_size,
            next_points=next_points,
            ucb_beta=self.beta,
            acq_value=[float(x.to("cpu")) for x in acq_value] if batch_size > 1 else [float(acq_value.to("cpu"))],
        )
        return doc, next_points
