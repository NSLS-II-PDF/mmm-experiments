import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import databroker.client
import nslsii
import requests
from bluesky_kafka import RemoteDispatcher
from bluesky_live.run_builder import RunBuilder
from tiled.client import from_profile


class Agent(ABC):
    """
    Single Plan Agent. These agents should consume data, decide where to measure next,
    and execute a single type of plan (maybe, move and count).
    """

    def __init__(self, *, beamline_tla: str, metadata: Optional[dict] = None):
        logging.debug("Initializing Agent")
        self.kafka_config = nslsii._read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")
        self.kafka_group_id = f"echo-{beamline_tla}-{str(uuid.uuid4())[:8]}"
        self.kafka_dispatcher = RemoteDispatcher(
            topics=[f"{beamline_tla}.bluesky.runengine.documents"],
            bootstrap_servers=",".join(self.kafka_config["bootstrap_servers"]),
            group_id=self.kafka_group_id,
            consumer_config=self.kafka_config["runengine_producer_config"],
        )
        logging.debug("Kafka setup sucessfully.")
        self.exp_catalog = from_profile(beamline_tla)
        logging.info(f"Reading data from catalog: {self.exp_catalog}")
        self.agent_catalog = from_profile(beamline_tla)["bluesky_sandbox"]
        logging.info(f"Writing data to catalog: {self.agent_catalog}")
        self.metadata = metadata or {}
        self.metadata["beamline_tla"] = beamline_tla
        self.metadata["kafka_group_id"] = self.kafka_group_id
        self.builder = None

    @property
    @abstractmethod
    def server_host(self):
        """
        Host to POST requests to. Declare as property or as class level attribute.
        Something akin to 'http://localhost:60610'
        """
        ...

    @property
    @abstractmethod
    def measurement_plan_name(self) -> str:
        """String name of registered plan"""
        ...

    @abstractmethod
    def measurement_plan_args(self, *args) -> list:
        """List of arguments to pass to plan"""
        ...

    @staticmethod
    @abstractmethod
    def unpack_run(run: databroker.client.BlueskyRun):
        """
        Consume a Bluesky run from tiled and emit the relevant x and y for the agent.

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
        ...

    @abstractmethod
    def tell(self, x, y) -> dict:
        """
        Tell the agent about some new data
        Parameters
        ----------
        x :
            Independent variable for data observed
        y :
            Dependent variable for data observed

        Returns
        -------
        dict
            Dictionary to be unpacked or added to a document

        """
        ...

    @abstractmethod
    def ask(self, batch_size: int) -> Tuple[dict, Sequence]:
        """
        Ask the agent for a new batch of points to measure.

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
        ...

    def report(self):
        """
        Create a report given the data observed by the agent.
        This could be potentially implemented in the base class to write document stream.
        """

        raise NotImplementedError

    def tell_many(self, xs, ys) -> Sequence[dict]:
        """
        Tell the agent about some new data. It is likely that there is a more efficient approach to
        handling multiple observations for an agent. The default behavior is to iterate over all
        observations and call the ``tell`` method.

        Parameters
        ----------
        xs : list, array
            Array of independent variables for observations
        ys : list, array
            Array of dependent variables for observations

        Returns
        -------
        list_of_dict

        """
        tell_emits = []
        for x, y in zip(xs, ys):
            tell_emits.append(self.tell(x, y))
        return tell_emits

    @property
    def _queue_add_position(self) -> Union[int, Literal["front", "back"]]:
        return "back"

    def _add_to_queue(self, batch_size: int = 1):
        """
        Soft wrapper for the `ask` method that puts the agent's
        proposed next points on the queue.

        Parameters
        ----------
        batch_size : int
            Number of new points to measure

        Returns
        -------

        """
        doc, next_points = self.ask(batch_size)
        url = Path(self.server_host) / "api" / "queue" / "item" / "add"
        responses = {}
        for point in next_points:
            doc, data = dict(
                pos=self._queue_add_position,
                item=dict(
                    name=self.measurement_plan_name, args=self.measurement_plan_args(point), item_type="plan"
                ),
            )
            r = requests.post(str(url), data=data)
            responses[point] = r
            logging.info(f"Sent http-server request for point {point}\n." f"Received reponse: {r}")
        # TODO: Should I be checking responses for anything?
        return doc

    def _write_event(self, stream, doc):
        """Add event to builder as event page, and publish to catalog"""
        if stream in self.builder._streams:
            self.builder.add_data(stream, data=doc)
        else:
            self.builder.add_stream(stream, data=doc)
        self.agent_catalog.v1.insert(*self.builder._cache._ordered[-1])

    def _on_stop_router(self, name, doc):
        """Service that runs each time a stop document is seen."""
        if name == "stop":
            uid = doc["run_start"]
            logging.info(
                f"New data detected, telling the agent about this start doc "
                f"and asking for a new suggestion: {uid}"
            )
            run = self.exp_catalog[uid]
            independent_variable, dependent_variable = self.unpack_run(run)

            # Tell
            logging.debug("Telling agent about some new data.")
            doc = self.tell(independent_variable, dependent_variable)
            self._write_event("tell", doc)

            # Ask
            logging.debug("Issuing ask and adding to the queue.")
            doc = self._add_to_queue(1)
            self._write_event("ask", doc)

    def start(self):
        logging.debug("Issuing start document and start listening to Kafka")
        self.builder = RunBuilder(metadata=self.metadata)
        self.agent_catalog.v1.insert("start", self.builder._cache.start_doc)
        logging.info(f"Agent start document uuid={self.builder._cache.start_doc['uid']}")
        self.kafka_dispatcher.subscribe(self._on_stop_router)
        self.kafka_dispatcher.start()

    def stop(self, exit_status="success", reason=""):
        logging.debug("Attempting agent stop.")
        self.builder.close(exit_status=exit_status, reason=reason)
        self.agent_catalog.v1.insert("stop", self.builder._cache.stop_doc)
        self.kafka_dispatcher.stop()
        logging.info(
            f"Stopped agent with exit status {exit_status.upper()}"
            f"{(' for reason: ' + reason) if reason else '.'}"
        )
