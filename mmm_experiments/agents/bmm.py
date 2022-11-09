import ast
from abc import ABC
from typing import Optional, Tuple

import databroker.client

from ..data.bmm_utils import Pandrosus
from .base import (
    Agent,
    DrowsyAgent,
    GeometricResolutionMixin,
    RandomAgentMixin,
    SequentialAgentMixin,
)
from .scientific_value import ScientificValueAgentMixin


class DrowsyBMMAgent(DrowsyAgent):
    """
    It's an agent that just lounges around all day.
    Alternates sending args vs kwargs to do the same thing.
    """

    server_host = "https://qserver.nsls2.bnl.gov/bmm"
    api_key = "zzzzz"

    def __init__(self):
        super().__init__(beamline_tla="bmm")


class BMMSingleEdgeAgent(Agent, ABC):
    """
    Abstract Agent containing communication and data defaults for BMM
    While the BMM experiment will measure a single fixed edge.
    This agent currently hard coded to focus on Ti.
    """

    server_host = "https://qserver.nsls2.bnl.gov/bmm"
    measurement_plan_name = "agent_measure_single_edge"
    api_key = "zzzzz"
    sample_position_motors = ("xafs_x", "xafs_y")

    def __init__(
        self,
        *,
        origin: Tuple[float, float],
        relative_bounds: Tuple[float, float],
        metadata: Optional[dict] = None,
        exp_filename: str = "MultimodalMadness",
        **kwargs,
    ):
        super().__init__(beamline_tla="bmm", metadata=metadata, **kwargs)
        self.origin = origin
        self._relative_bounds = relative_bounds
        self._exp_filename = exp_filename

    @property
    def exp_filename(self):
        return self._exp_filename

    def set_filename(self, filename: str):
        self._exp_filename = filename

    def get_relative_position(self, absolute_position):
        """Sample inverted w.r.t. PDF"""
        return self.origin[0] - absolute_position

    def get_absolute_position(self, relative_position):
        return self.origin[0] - relative_position

    def measurement_plan_args(self, point):
        """List of arguments to pass to plan.
        BMM agents are relative to Cu origin, but separate origins are needed for other element edges."""
        return (
            self.sample_position_motors[0],
            self.get_absolute_position(point),
            self.sample_position_motors[1],
            self.origin[1],
        )

    def measurement_plan_kwargs(self, point) -> dict:
        return dict(
            filename=self.exp_filename,
            nscans=1,
            start="next",
            mode="transmission",
            edge="L3",
            element="Pt",
            sample="PtZr",
            preparation="film sputtered on silica",
            bounds="-200 -30 -10 25 13k",
            steps="10 2 0.3 0.05k",
            times="0.5 0.5 0.5 0.5",
            snapshots=False,
            md={"relative_position": point},
        )

    @staticmethod
    def unpack_run(run: databroker.client.BlueskyRun):
        """Gets Chi(k) and absolute position"""
        run_preprocessor = Pandrosus()
        run_preprocessor.fetch(run, mode="transmission")
        # x_data = run_preprocessor.group.k
        y_data = run_preprocessor.group.chi
        md = ast.literal_eval(run.start["XDI"]["_comment"][0])
        return md["x_position"], y_data

    def trigger_condition(self, uid) -> bool:
        return (
            "XDI" in self.exp_catalog[uid].start
            and self.exp_catalog[uid].start["plan_name"].startswith("scan_nd")
            and self.exp_catalog[uid].start["XDI"]["Element"]["symbol"] == "Pt"
        )

    @property
    def measurement_origin(self):
        return self.origin[0]

    @property
    def relative_bounds(self):
        return self._relative_bounds

    def report(self):
        pass


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
        exp_filename: str = "MultimodalMadness",
        **kwargs,
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
        super().__init__(beamline_tla="bmm", metadata=metadata, **kwargs)
        self.Cu_origin = Cu_origin
        self.Ti_origin = Ti_origin
        self.Cu_det_position = Cu_det_position
        self.Ti_det_position = Ti_det_position
        self._relative_bounds = relative_bounds
        self.exp_filename = exp_filename

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
        """List of arguments to pass to plan.
        BMM agents are relative to Cu origin, but separate origins are needed for other element edges."""
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
            filename=self.exp_filename,
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
            md={"relative_position": point},
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


class SequentialAgent(SequentialAgentMixin, BMMSingleEdgeAgent):
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


class RandomAgent(RandomAgentMixin, BMMSingleEdgeAgent):
    """
    Hears a stop document and immediately suggests a random point within the bounds.
    Uses the same signature as SequentialAgent.
    """


class GeometricAgent(GeometricResolutionMixin, BMMSingleEdgeAgent):
    """Geometric series for exploration at BMM"""


class ScientificValue(ScientificValueAgentMixin, BMMSingleEdgeAgent):
    """Scientific value agent for BMM"""
