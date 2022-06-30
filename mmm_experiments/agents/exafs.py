from collections import namedtuple
from pathlib import Path
from typing import Literal, Tuple, Union


from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from databroker.client import BlueskyRun
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch

from .base import Agent

DATA_KEY = None  # TODO


class DumbDistanceEXAFSAgent(Agent):
    """The point of this agent is to do precisely one thing: maximize the
    distance between some newly acquired point(s) and the provided reference
    spectra. The reference EXAFS represent "known phases". The argument is
    simple: if a new EXAFS has a e.g. Euclidean distance that is large compared
    to the reference, it is likely "different" than the reference. If it has
    a distance that is small, it is likely not different than the reference.
    Multiple references can be provided, and the distance considered will be
    the minimum distance for any of the references."""

    server_host = None  # TODO "qserver1.nsls2.bnl.gov:60610"
    measurement_plan_name = None  # TODO ? "mv_and_jog"

    def __init__(
        self,
        sample_origin,
        relative_bounds,
        reference_spectra,
        device="cpu",
        beta_UCB=0.1,
    ):
        """
        Parameters
        ----------
        sample_origin : float
            Decided origin of sample in raw motor coordinates. Assumed to be
            one dimensional for now. TODO
        relative_bounds : tuple
            Relative bounds for the sample measurement. For a 10 cm sample,
            this would be something like (1, 99).
        reference_spectra : array_like
            A list of lists or array where the first axis is the spectrum
            index. These are the references that we're comparing measured
            spectra against.
        device : str
            Torch device to keep the GP. This model should be quite cheap so
            by default we'll put it on the cpu.
        beta_UCB : float, optional
            Default value for the beta parameter in the Upper Confidence Bound
            acquisition function.
        """

        AVAIL_DEVICES = ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        if device not in AVAIL_DEVICES:
            raise ValueError(f"device={device} must be one of {AVAIL_DEVICES}")

        if len(relative_bounds) != 2:
            raise ValueError(
                f"relative_bounds={relative_bounds} must be a length-2 tuple "
                "of floats"
            )

        super().__init__(beamline_tla="bmm") 

        self.sample_origin = sample_origin
        self.device = torch.device(device)
        self.bounds = torch.tensor(relative_bounds).to(self.device).float()
        self.reference_spectra = np.array(reference_spectra)
        self.beta_UCB = beta_UCB

        # Store all of the measured EXAFS
        self.exafs_cache = []

        # Store all of the relative motor positions
        self.relative_position_cache = []

        # Store all of the measured distances to the reference spectra
        self.target_cache = []

    @property
    def server_host(self):
        """Host to POST requests to. Declare as property or as class level
        attribute. Something akin to 'http://localhost:60610'
        """

        raise NotImplementedError

    @property
    def measurement_plan_name(self):
        """String name of registered plan"""

        raise NotImplementedError

    def measurement_plan_args(self, *args):
        """List of arguments to pass to plan"""

        raise NotImplementedError

    @staticmethod
    def unpack_run(run):
        """
        Consume a Bluesky run from tiled and emit the relevant x and y for the
        agent.

        Parameters
        ----------
        run : databroker.client.BlueskyRun

        Returns
        -------
        independent_var :
            The independent variable of the measurement
        dependent_var :
            The measured data, processed for relevance
        """

        raise NotImplementedError

    def _get_distances_from_reference_spectra(self, intensity):
        """
        Parameters
        ----------
        intensity : array
            The EXAFS intensity.

        Returns
        -------
        float
        """

        # The reference spectra and intensity need to be "broadcastable"
        # abs_diff should be N x M, where N is the number of reference spectra
        # and M is the number of energy grid points
        abs_diff = np.abs(self.reference_spectra - intensity)
        summed_diff = abs_diff.sum(axis=1)
        return np.min(summed_diff).item()

    def tell(self, position, intensity):
        """Takes the position of the motor and the measured intensity (which
        is assumed to be on a standardized grid) and returns a dictionary
        of documentation (?? TODO).

        Parameters
        ----------
        position : float
            The one-dimensional position of the motor.
        intensity : array_like
            The EXAFS intensity.

        Returns
        -------
        dict
        """

        relative_position = position - self.sample_origin
        self.relative_position_cache.append(relative_position)
        intensity = np.array(intensity)
        self.exafs_cache.append(intensity)

        # Deal with the featurization of the EXAFS data
        new_distance = self._get_distances_from_reference_spectra(intensity)
        self.target_cache.append(new_distance)

        # Doc? TODO (what is this for?)
        return dict(
            position=position,
            rel_position=relative_position,
            intensity=intensity,
        )

    def ask(
        self,
        batch_size=1,
        ucb_kwargs=dict(),
        optimize_acqf_kwargs={"num_restarts": 5, "raw_samples": 20}
    ):
        """Trains a single task Gaussian process to predict the distance to the
        nearest reference spectrum as a function of relative position. Optimize
        over this GP using an upper confidence bound acquisition function to
        find the areas of highest difference i.e. most novelty.

        Parameters
        ----------
        batch_size : int
            Number of new points to measure
        ucb_kwargs : dict, optional
            Keyword arguments to pass to the UCB acquisition function or the
            MC version if batch_size > 1.
        optimize_acqf_kwargs : dict, optional
            Keyword arguments to pass to botorch's optimize_acqf

        Returns
        -------
        doc : dict
            key metadata from the ask approach
        next_points : Sequence
            Sequence of independent variables of length batch size
        """

        train_x = torch.tensor(self.position_cache, dtype=torch.float)
        train_x = train_x.to(self.device)
        train_y = torch.tensor(self.target_cache, dtype=torch.float)
        train_y = train_y.to(self.device)

        # If it works, it works!
        gp = SingleTaskGP(train_x, train_y).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_model(mll)

        if batch_size == 1:
            ucb = UpperConfidenceBound(gp, beta=self.beta_UCB, **ucb_kwargs)
        else:
            # Using a bunch of defaults here but whatever
            ucb = qUpperConfidenceBound(gp, beta=self.beta_UCB, **ucb_kwargs)

        next_points, acq_value = optimize_acqf(
            ucb,
            bounds=self.bounds,
            q=batch_size,
            **optimize_acqf_kwargs,
        )

        if batch_size == 1:
            next_points = [float(next_points.to("cpu"))]
        else:
            next_points = [float(x.to("cpu")) for x in next_points]

        doc = dict(
            batch_size=batch_size,
            next_points=next_points,
            acq_value=[float(x.to("cpu")) for x in acq_value]
            if batch_size > 1 else [float(acq_value.to("cpu"))],
        )
        return doc, next_points
