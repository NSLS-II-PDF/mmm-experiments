import logging
from abc import ABC, abstractmethod
from typing import Tuple

from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.http import REManagerAPI

from .base import GeometricResolutionMixin, SequentialAgentMixin
from .bmm import BMMAgent
from .pdf import PDFAgent


class MonarchSubjectBase(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subject_manager = REManagerAPI(http_server_uri=self.subject_host)
        self.subject_manager.set_authorization_key(api_key=self.subject_api_key)

    @property
    @abstractmethod
    def subject_host(self):
        """
        Host to POST requests to for subject. Declare as property or as class level attribute.
        """
        ...

    @property
    @abstractmethod
    def subject_api_key(self):
        """
        Subject key for API security.
        """
        ...

    @property
    @abstractmethod
    def subject_plan_name(self) -> str:
        """String name of registered plan"""
        ...

    @abstractmethod
    def subject_plan_args(self, point) -> list:
        """List of arguments to pass to plan from a point to measure."""
        ...

    @abstractmethod
    def subject_plan_kwargs(self, point) -> dict:
        """List of arguments to pass to plan from a point to measure."""
        ...

    @property
    @abstractmethod
    def subject_origin(self):
        """Distinctly useful for having twinned samples and mixin classes. The origin of independent variable."""


# TODO: add the right mixin for PDF. Could be sequential, or maybe another geometic
class MonarchPDFSubjectTLA(SequentialAgentMixin, MonarchSubjectBase, PDFAgent):
    def __init__(self):
        raise NotImplementedError


class MonarchBMMSubjectPDF(GeometricResolutionMixin, MonarchSubjectBase, BMMAgent):
    subject_host = PDFAgent.server_host
    subject_api_key = PDFAgent.api_key
    subject_plan_name = PDFAgent.measurement_plan_name

    def __init__(self, pdf_origin: Tuple[float, float], **kwargs):
        self.pdf_origin = pdf_origin
        super().__init__(**kwargs)

    @property
    def subject_origin(self):
        return self.pdf_origin

    def subject_plan_args(self, point):
        return ["Grid_X", point + self.subject_origin, 5]

    def subject_plan_kwargs(self, point) -> dict:
        return {}

    def generate_subject_ask(self) -> list:
        """This is where the clever happens"""
        raise NotImplementedError

    def ask(self, batch_size: int = 1):
        """Standard Geometric ask, Then add plans to subject"""
        doc, points = super().ask(batch_size)
        # if done, publish to TLA subject
        if doc["ask_ready"][0] is True:
            subject_points = self.generate_subject_ask()
            for point in subject_points:

                plan = BPlan(
                    self.subject_plan_name, *self.subject_plan_args(point), **self.subject_plan_kwargs(point)
                )
                r = self.subject_manager.item_add(plan, pos="front")
                logging.info(f"Sent subject http-server request for point {point}\n." f"Received reponse: {r}")
            doc["subject_points"] = [subject_points]
