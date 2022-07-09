import ast
from abc import ABC
from typing import Optional, Sequence, Tuple

import databroker.client
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..data.bmm_utils import Pandrosus
from .base import Agent, DrowsyAgent, RandomAgentMixin, SequentialAgentMixin


class DrowsyBMMAgent(DrowsyAgent):
    """
    It's an agent that just lounges around all day.
    Alternates sending args vs kwargs to do the same thing.
    """

    server_host = "https://qserver.nsls2.bnl.gov/bmm"
    api_key = "zzzzz"

    def __init__(self):
        super().__init__(beamline_tla="bmm")


class BMMAgent(Agent, ABC):
    """
    Abstract Agent containing communication and data defaults for BMM
    While the BMM experiment will measure both the Cu and Ti edge.
    The agent will by default only consider the data from the Cu K-edge measurement.
    """

    server_host = "https://qserver.nsls2.bnl.gov/bmm"
    measurement_plan_name = "agent_move_and_measure"
    # measurement_plan_name = "agent_xafs"
    api_key = "zzzzz"
    sample_position_motors = ("xafs_x", "xafs_y")

    def __init__(
        self,
        *,
        Cu_origin: Tuple[float, float],
        Ti_origin: Tuple[float, float],
        Cu_det_position: float,
        Ti_det_position: float,
        relative_bounds: Tuple[float, float],
        metadata: Optional[dict] = None,
    ):
        """

        Parameters
        ----------
        Cu_origin : Tuple[float, float]
            Decided origin of sample in raw motor coordinates [xafs_x, xafs_y] for Cu measurement
        Ti_origin : Tuple[float, float]
            Decided origin of sample in raw motor coordinates [xafs_x, xafs_y] for Ti measurement
        Cu_det_position : float
            Default detector position for Cu measurement
        Ti_det_position : float
            Default detector position for Ti measurement
        relative_bounds : Tuple[float, float]
            Relative bounds for the sample measurement. For a 10 cm sample,
            this would be something like (1, 99).
        metadata : dict
            Optional metadata dictionary for the agent start document
        """
        super().__init__(beamline_tla="bmm", metadata=metadata)
        self.Cu_origin = Cu_origin
        self.Ti_origin = Ti_origin
        self.Cu_det_position = Cu_det_position
        self.Ti_det_position = Ti_det_position
        self._relative_bounds = relative_bounds

    @staticmethod
    def unpack_run(run: databroker.client.BlueskyRun):
        """Gets Chi(k) and absolute position"""
        run_preprocessor = Pandrosus()
        run_preprocessor.fetch(run, mode="fluorescence")
        # x_data = run_preprocessor.group.k
        y_data = run_preprocessor.group.chi
        md = ast.literal_eval(run.start["XDI"]["_comment"][0])
        return md["Cu_position"], y_data

    def measurement_plan_args(self, point):
        """List of arguments to pass to plan"""
        return (
            self.sample_position_motors[0],
            self.Cu_origin[0] + point,
            self.Ti_origin[0] + point,
            self.sample_position_motors[1],
            self.Cu_origin[1],
            self.Ti_origin[1],
        )

    def measurement_plan_kwargs(self, point) -> dict:
        return dict(
            Cu_det_position=self.Cu_det_position,
            Ti_det_position=self.Ti_det_position,
            filename="MultimodalMadness",
            nscans=1,
            start="next",
            mode="fluorescence",
            edge="K",
            sample="CuTi",
            preparation="film sputtered on silica",
            bounds="-200 -30 -10 25 12k",
            steps="10 2 0.3 0.05k",
            times="0.5 0.5 0.5 0.5",
            snapshots=False,
        )

    def trigger_condition(self, uid) -> bool:
        return (
            "XDI" in self.exp_catalog[uid].start
            and self.exp_catalog[uid].start["plan_name"].startswith("scan_nd")
            and self.exp_catalog[uid].start["XDI"]["Element"]["symbol"] == "Cu"
        )

    @property
    def measurement_origin(self):
        return self.Cu_origin[0]

    @property
    def relative_bounds(self):
        return self._relative_bounds

    def report(self):
        pass


class SequentialAgent(SequentialAgentMixin, BMMAgent):
    """
    Hears a stop document and immediately suggests the nearest neighbor
    Parameters
    ----------
    step_size : float
        How far to step forward in the measurement. Defaults to not moving at all.
    kwargs :
        Necessary kwargs for BMMAgent
    """

    def __init__(self, step_size: float = 0.0, **kwargs):
        super().__init__(step_size=step_size, **kwargs)


class RandomAgent(RandomAgentMixin, BMMAgent):
    """
    Hears a stop document and immediately suggests a random point within the bounds.
    Uses the same signature as SequentialAgent.
    """


class DumbDistanceEXAFSAgent(BMMAgent):
    """The point of this agent is to do precisely one thing: maximize the
    distance between some newly acquired point(s) and the provided reference
    spectra. The reference EXAFS represent "known phases". The argument is
    simple: if a new EXAFS has a e.g. Euclidean distance that is large compared
    to the reference, it is likely "different" than the reference. If it has
    a distance that is small, it is likely not different than the reference.
    Multiple references can be provided, and the distance considered will be
    the minimum distance for any of the references."""

    def __init__(
        self,
        Cu_origin: Tuple[float, float],
        Ti_origin: Tuple[float, float],
        Cu_det_position: float,
        Ti_det_position: float,
        relative_bounds,
        reference_spectra,
        device="cpu",
        beta_UCB=0.1,
        metadata: Optional[dict] = None,
        restart_from_uid: Optional[str] = None,
    ):
        """
        Parameters
        ----------
         Cu_origin : Tuple[float, float]
            Decided origin of sample in raw motor coordinates [xafs_x, xafs_y] for Cu measurement
        Ti_origin : Tuple[float, float]
            Decided origin of sample in raw motor coordinates [xafs_x, xafs_y] for Ti measurement
        Cu_det_position : float
            Absolute motor position for the xafs detector for the Cu measurement.
        Ti_det_position : float
            Absolute motor position for the xafs detector for the Ti measurement.
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
            raise ValueError(f"relative_bounds={relative_bounds} must be a length-2 tuple " "of floats")

        super().__init__(
            Cu_origin=Cu_origin,
            Ti_origin=Ti_origin,
            Cu_det_position=Cu_det_position,
            Ti_det_position=Ti_det_position,
            metadata=metadata,
            restart_from_uid=restart_from_uid,
        )

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
        of documentation

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

        relative_position = position - self.Cu_origin[0]
        self.relative_position_cache.append(relative_position)
        intensity = np.array(intensity)
        self.exafs_cache.append(intensity)

        # Deal with the featurization of the EXAFS data
        new_distance = self._get_distances_from_reference_spectra(intensity)
        self.target_cache.append(new_distance)

        return dict(
            position=position,
            rel_position=relative_position,
            intensity=intensity,
        )

    def ask(
        self, batch_size=1, ucb_kwargs=dict(), optimize_acqf_kwargs={"num_restarts": 5, "raw_samples": 20}
    ) -> Tuple[dict, Sequence]:
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
            acq_value=[float(x.to("cpu")) for x in acq_value] if batch_size > 1 else [float(acq_value.to("cpu"))],
        )
        return doc, next_points
