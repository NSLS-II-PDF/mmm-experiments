from abc import ABC
from collections import namedtuple
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from databroker.client import BlueskyRun
from tiled.client import from_profile

from .base import (
    Agent,
    DrowsyAgent,
    GeometricResolutionMixin,
    RandomAgentMixin,
    SequentialAgentMixin,
)
from .ml_mixins import CMFMixin, XCAMixin, XCAValueMixin

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
        super().__init__(beamline_tla="pdf", metadata=metadata)
        self.exp_catalog = from_profile("pdf_bluesky_sandbox")
        self.sample_origin = sample_origin
        self._relative_bounds = relative_bounds

    @property
    def measurement_origin(self):
        return self.sample_origin[0]

    @property
    def relative_bounds(self):
        return self._relative_bounds

    def measurement_plan_args(self, x_position) -> list:
        """Plan to be writen than moves to an x_position then jogs up and down relatively in y"""
        return ["Grid_X", x_position + self.measurement_origin, 30]

    def measurement_plan_kwargs(self, point) -> dict:
        md = {"relative_position": point}
        return {"sample_number": 16, "md": md}

    @staticmethod
    def unpack_run(run: BlueskyRun):
        """"""
        # x = np.array(run.primary.data["chi_2theta"][0])
        y = np.array(run.primary.data["chi_I"][0])
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
