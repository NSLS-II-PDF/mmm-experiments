import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import xarray
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.http import REManagerAPI
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

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


class MonarchPDFSubjectBMM(GeometricResolutionMixin, MonarchSubjectBase, PDFAgent):
    subject_host = BMMAgent.server_host
    subject_api_key = BMMAgent.api_key
    subject_plan_name = BMMAgent.measurement_plan_name

    def __init__(
        self,
        *,
        Cu_origin: Tuple[float, float],
        Ti_origin: Tuple[float, float],
        Cu_det_position: float,
        Ti_det_position: float,
        bmm_bounds: Tuple[float, float],
        bmm_time_window: float = 30,  # Time in minutes
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_position_motors = ("xafs_x", "xafs_y")
        self.Cu_origin = Cu_origin
        self.Ti_origin = Ti_origin
        self.Cu_det_position = Cu_det_position
        self.Ti_det_position = Ti_det_position
        self.bmm_bounds = bmm_bounds
        self.bmm_request_cache = set()
        self.bmm_time_window = bmm_time_window
        self.bmm_last_request = time.time()
        self.background = self.get_wafer_background()
        self.dependent_cache = []
        self.model_method = "NMF"

    @property
    def subject_origin(self):
        return self.Cu_origin[0]

    def subject_plan_args(self, point):
        return BMMAgent.measurement_plan_args(self, point)

    def subject_plan_kwargs(self, point) -> dict:
        kwargs = BMMAgent.measurement_plan_kwargs(self, point)
        kwargs.setdefault("md", {})
        kwargs["md"]["agent"] = f"MonarchPDFSubjectBMM_{self.model_method}"
        kwargs["md"].update(self.default_plan_md)
        return kwargs

    def get_wafer_background(self):
        background_runs = self.exp_catalog.search({"sample_name": "MT_wafer5"}).values_indexer[4:]
        background_stack = xarray.concat((r.primary.read(["chi_Q", "chi_I"]) for r in background_runs), dim="time")
        background = background_stack.mean(dim="time")
        return background

    def generate_subject_ask(self) -> list:
        """Alternate NMF and KMeans to find the most interesting triplet."""
        data = np.stack(self.dependent_cache)
        if self.model_method == "Kmeans":
            self.model_method = "NMF"
            nmf = NMF(3)
            nmf.fit(data)
            components = nmf.components_
        else:
            self.model_method = "Kmeans"
            model = KMeans(3)
            model.fit(data)
            components = model.cluster_centers_
        nearest = []  # nearest indices to the derived components (MSE)
        for c in components:
            nearest.append(np.argmin(np.mean((data - c) ** 2, axis=-1)))
        return [self.independent_cache[i] for i in nearest]

    def tell(self, position, y):
        doc = super().tell(position, y)
        self.dependent_cache.append(y.data)
        return doc

    def ask(self, batch_size: int = 1):
        """Standard ask, Then checks time and adds plans to subject"""
        doc, points = super().ask(batch_size)
        if time.time() - self.bmm_last_request > self.bmm_time_window * 60:
            logging.info("Enough time elapsed. Generating new points to send to BMM.")
            subject_points = self.generate_subject_ask()
            for point in subject_points:
                if point > self.bmm_bounds[1]:
                    logging.info(f"Point {point} beyond BMM bounds, cycling to other end.")
                    point = self.bmm_bounds[0] + (point - self.bmm_bounds[1])
                if point in self.bmm_request_cache:
                    logging.info(f"Point {point} already measured by BMM. Skipping...")
                    continue
                plan = BPlan(
                    self.subject_plan_name, *self.subject_plan_args(point), **self.subject_plan_kwargs(point)
                )
                r = self.subject_manager.item_add(plan, pos="front")
                logging.info(
                    f"Sent BMM http-server priority request for point {point}\n. " f"Received reponse: {r}"
                )
                if r["success"] is True:
                    self.bmm_request_cache.add(point)
            self.bmm_last_request = time.time()
        return doc, points


class MonarchBMMSubjectPDF(GeometricResolutionMixin, MonarchSubjectBase, BMMAgent):
    subject_host = PDFAgent.server_host
    subject_api_key = PDFAgent.api_key
    subject_plan_name = PDFAgent.measurement_plan_name

    def __init__(self, pdf_origin: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.pdf_origin = pdf_origin
        self.dependent_cache = []

    @property
    def subject_origin(self):
        return self.pdf_origin[0]

    def subject_plan_args(self, point):
        return ["Grid_X", point + self.subject_origin, 30]

    def subject_plan_kwargs(self, point) -> dict:
        md = self.default_plan_md
        md["relative_position"] = point
        return {"sample_number": 17, "md": md}

    def measurement_plan_kwargs(self, point) -> dict:
        kwargs = super().measurement_plan_kwargs(point)
        kwargs.setdefault("md", {})
        kwargs["md"]["agent"] = "MonarchBMMSubjectPDF"
        return kwargs

    def generate_subject_ask(self) -> list:
        """BMM asks PDF to measure it's most distinct signal. Construct distance matrix and sum"""
        data = np.stack(self.dependent_cache)
        dists = -2 * np.dot(data, data.T) + np.sum(data**2, axis=1) + np.sum(data**2, axis=1)[:, np.newaxis]
        return [self.independent_cache[np.argmax(np.sum(dists, axis=-1))]]

    def tell(self, position, y):
        doc = super().tell(position, y)
        self.dependent_cache.append(y.data)
        return doc

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

        return doc, points


class SequentialMonarchPDF(SequentialAgentMixin, MonarchSubjectBase, PDFAgent):
    # FOR BMM
    subject_host = BMMAgent.server_host
    subject_api_key = BMMAgent.api_key
    subject_plan_name = BMMAgent.measurement_plan_name

    def __init__(
        self,
        *,
        Cu_origin: Tuple[float, float],
        Ti_origin: Tuple[float, float],
        Cu_det_position: float,
        Ti_det_position: float,
        bmm_bounds: Tuple[float, float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_position_motors = ("xafs_x", "xafs_y")
        self.Cu_origin = Cu_origin
        self.Ti_origin = Ti_origin
        self.Cu_det_position = Cu_det_position
        self.Ti_det_position = Ti_det_position
        self.bmm_bounds = bmm_bounds
        self.bmm_cache = [0.0]

    @property
    def subject_origin(self):
        return self.Cu_origin[0]

    def subject_plan_args(self, point):
        return BMMAgent.measurement_plan_args(self, point)

    def subject_plan_kwargs(self, point) -> dict:
        return BMMAgent.measurement_plan_kwargs(self, point)

    def ask(self, batch_size: int = 1):
        """Standard ask, Then add plans to subject
        The base class calls `ask` during `_add_to_queue`. In this way, the subject will get points added first,
        and the Monarch will get the points added after the ask.
        """
        doc, points = super().ask(batch_size)
        doc["bmm_proposed_points"] = []
        for i in range(len(points)):
            point = self.bmm_cache[-1] + self.step_size
            if point > self.bmm_bounds[1]:
                point = self.bmm_bounds[0]
            self.bmm_cache.append(point)
            doc["bmm_proposed_points"].append(point)
            plan = BPlan(self.subject_plan_name, *self.subject_plan_args(point), **self.subject_plan_kwargs(point))
            r = self.subject_manager.item_add(plan, pos="front")
            logging.info(f"Sent BMM http-server priority request for point {point}\n." f"Received reponse: {r}")
        return doc, points
