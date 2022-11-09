from abc import ABC
from collections import namedtuple
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import xarray
from databroker.client import BlueskyRun
from scipy.interpolate import interp1d
from tiled.client import from_profile

from .base import (
    Agent,
    DrowsyAgent,
    GeometricResolutionMixin,
    RandomAgentMixin,
    SequentialAgentMixin,
)
from .ml_mixins import CMFMixin, XCAMixin, XCAValueMixin
from .scientific_value import ScientificValueAgentMixin

Representation = namedtuple("Representation", "probabilities shannon_entropy reconstruction_loss")


class DrowsyPDFAgent(DrowsyAgent):
    """
    It's an agent that just lounges around all day.
    Alternates sending args vs kwargs to do the same thing.
    """

    server_host = "https://qserver.nsls2.bnl.gov/pdf"
    api_key = "yyyyy"

    def __init__(self):
        super().__init__(beamline_tla="pdf")


class PDFAgent(Agent, ABC):
    server_host = "https://qserver.nsls2.bnl.gov/pdf"
    measurement_plan_name = "agent_sample_count"
    api_key = "yyyyy"

    def __init__(
        self,
        *,
        sample_origin: Tuple[float, float],
        relative_bounds: Tuple[float, float],
        metadata: Optional[dict] = None,
        sample_number: int = 0,
        **kwargs,
    ):
        """
        Base class for all PDF agents

        Parameters
        ----------
        sample_origin : Tuple[float, float]
            Decided origin of sample in raw motor coordinates
        relative_bounds : Tuple[float, float]
            Relative bounds for the sample measurement. For a 10 cm sample, this would be something like (1, 99).
        metadata : dict
            Optional metadata dictionary for the agent start document
        """
        super().__init__(beamline_tla="pdf", metadata=metadata, **kwargs)
        self.exp_catalog = from_profile("pdf_bluesky_sandbox")
        self.sample_origin = sample_origin
        self._relative_bounds = relative_bounds
        self.xrd_background = self.get_wafer_background("xrd")
        self.pdf_background = self.get_wafer_background("pdf")
        self._sample_number = sample_number

    @property
    def measurement_origin(self):
        return self.sample_origin[0]

    @property
    def relative_bounds(self):
        return self._relative_bounds

    def get_relative_position(self, absolute_position):
        """Sample inverted w.r.t. PDF"""
        return absolute_position - self.measurement_origin

    def get_absolute_position(self, relative_position):
        return self.measurement_origin + relative_position

    @property
    def sample_number(self):
        return self._sample_number

    def set_sample_number(self, sample_number: int):
        self._sample_number = sample_number

    def measurement_plan_args(self, x_position) -> list:
        """Plan to be writen than moves to an x_position then jogs up and down relatively in y"""
        return ["Grid_X", self.get_absolute_position(x_position), 30]

    def measurement_plan_kwargs(self, point) -> dict:
        md = {"relative_position": point}
        return {"sample_number": self.sample_number, "md": md}

    def get_wafer_background(self, mode="pdf"):
        ignore_uids = [
            "f7c84b98-4a42-4a7c-a011-2087b9ef2196",
            "7498e441-51b3-443e-9005-19c923ea1f0b",
            "b0f0817a-cece-4bc4-8ba3-235fdd48776d",
        ]
        background_runs = [
            run
            for run in self.exp_catalog.search({"sample_name": f"{mode}_MTwafer"}).values_indexer[:]
            if run.start["uid"] not in ignore_uids
        ]
        return xarray.concat((r.primary.read(["chi_Q", "chi_I"]) for r in background_runs), dim="time").mean(
            dim="time"
        )

    @staticmethod
    def bkg_scaler(x, y, bkg, qmin=1.45, qmax=1.65):
        fgd_sum = np.sum(y[(x > qmin) & (x < qmax)])
        bgd_sum = np.sum(bkg["chi_I"][(bkg["chi_Q"] > qmin) & (bkg["chi_Q"] < qmax)])
        return fgd_sum / bgd_sum

    def unpack_run(self, run: BlueskyRun):
        """Removes background"""
        x = run.primary.data["chi_Q"][0]
        y = run.primary.data["chi_I"][0]
        # Big ditances use XRD, short distances are pdf
        if run.start["Det_1_Z"]["Det_1_Z"]["value"] > 4000:
            background = self.xrd_background
        else:
            background = self.pdf_background
        f = interp1d(x, y, fill_value=0.0, bounds_error=False)
        y = f(background["chi_Q"])
        scaler = self.bkg_scaler(background["chi_Q"], y, background)
        y = y - float(scaler) * np.array(background["chi_I"])
        y = (y - y.min()) / (y.max() - y.min())
        return run.start["Grid_X"]["Grid_X"]["value"], y

    def report(self):
        pass


class SequentialAgent(SequentialAgentMixin, PDFAgent):
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


class RandomAgent(RandomAgentMixin, PDFAgent):
    """
    Hears a stop document and immediately suggests a random point within the bounds.
    Uses the same signature as SequentialAgent.
    """


class GeometricAgent(GeometricResolutionMixin, PDFAgent):
    """Geometric series for resolution at PDF"""


class CMFAgent(CMFMixin, PDFAgent):
    """
    Constrained Matrix Factorization agent that will decompose an ordered dataset on `ask` or `report`.

    num_components : int
        Number of components to initially use. Can be updated via `update_num_components`
        through the agent Kafka interface.
    ask_mode : str
        One of AVAILABLE_ASK_MODES.
        autoconstrained : iteratively perform CMF by adding constraints drawn from the dataset
        unconstrained : plain old NMF
    kwargs
    """

    def __init__(self, *, num_components: int, ask_mode: str, **kwargs):
        super().__init__(num_components=num_components, ask_mode=ask_mode, **kwargs)


class XCAPassiveAgent(XCAMixin, PDFAgent):
    """
    Crystallography companion agent that will predict the phase, and provide a latent representation.
    This mixin has no mechanism for feedback via `ask`, it is strictly a passive analysis agent.
    Each `tell` will be documented with the expectation of the model, and a `report` will trigger a sorted
    and comprehensive report on history.

    Parameters
    ----------
    model_checkpoint : Union[str, Path]
        Path to the pre-trained model checkpoint
    model_qspace : np.ndarray
        Numpy array of the trained model qspace. Likely a linspace.
    device : Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        Device to deploy agent on. Available devices on tritium listed.
    kwargs
    """

    def __init__(
        self,
        *,
        model_checkpoint: Union[str, Path],
        model_qspace: np.ndarray,
        device: Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        **kwargs,
    ):
        super().__init__(model_checkpoint=model_checkpoint, model_qspace=model_qspace, device=device, **kwargs)


class XCAActiveAgent(XCAValueMixin, PDFAgent):
    """
    Crystallography companion agent that will predict the phase, and provide a latent representation.
    This mixin uses the scientific value function to optmize over in the `ask`. Uniquely it determines distance
    between two dependent variables by their latent representation instead of the spectral representation.
    Each `tell` will be documented with the expectation of the model, and a `report` will trigger a sorted
    and comprehensive report on history.

    Parameters
    ----------
    model_checkpoint : Union[str, Path]
        Path to the pre-trained model checkpoint
    model_qspace : np.ndarray
        Numpy array of the trained model qspace. Likely a linspace.
    xca_device : Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        Device to deploy forward model on. Available devices on tritium listed.
    botorch_device : Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        Device to deploy bayes opt model on. Available devices on tritium listed.
    beta : float
        beta value for upper confidence bound acquisition function
    kwargs
    """

    def __init__(
        self,
        *,
        model_checkpoint: Union[str, Path],
        model_qspace: np.ndarray,
        xca_device: Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        botorch_device: Literal["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        **kwargs,
    ):
        super().__init__(
            model_checkpoint=model_checkpoint,
            model_qspace=model_qspace,
            xca_device=xca_device,
            botorch_device=botorch_device,
            **kwargs,
        )


class ScientificValue(ScientificValueAgentMixin, PDFAgent):
    """Scientific value agent for PDF"""
