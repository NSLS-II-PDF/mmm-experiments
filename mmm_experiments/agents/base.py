import logging
import signal
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Tuple, Union

import databroker.client
import nslsii
from bluesky_kafka import RemoteDispatcher
from bluesky_live.run_builder import RunBuilder
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.http import REManagerAPI
from tiled.client import from_profile


class Agent(ABC):
    """
    Single Plan Agent. These agents should consume data, decide where to measure next,
    and execute a single type of plan (maybe, move and count).

    Base agent sets up a kafka subscription to listen to new stop documents, a catalog to read for experiments,
    a catalog to write agent status to, and a manager API for the HTTP server.
    """

    def __init__(
        self,
        *,
        beamline_tla: str,
        metadata: Optional[dict] = None,
        restart_from_uid: Optional[str] = None,
    ):
        logging.debug("Initializing Agent")
        self.kafka_config = nslsii._read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")
        self.kafka_group_id = f"echo-{beamline_tla}-{str(uuid.uuid4())[:8]}"
        self.kafka_dispatcher = RemoteDispatcher(
            topics=[f"{beamline_tla}.bluesky.pdfstream.documents"]
            if beamline_tla == "pdf"
            else [f"{beamline_tla}.bluesky.runengine.documents"],  # PDF will attend to analyized data
            bootstrap_servers=",".join(self.kafka_config["bootstrap_servers"]),
            group_id=self.kafka_group_id,
            consumer_config=self.kafka_config["runengine_producer_config"],
        )
        logging.debug("Kafka setup sucessfully.")
        self.exp_catalog = from_profile(beamline_tla)
        logging.info(f"Reading data from catalog: {self.exp_catalog}")
        self.agent_catalog = from_profile("nsls2")[beamline_tla]["bluesky_sandbox"]
        logging.info(f"Writing data to catalog: {self.agent_catalog}")
        self.metadata = metadata or {}
        self.metadata["beamline_tla"] = beamline_tla
        self.metadata["kafka_group_id"] = self.kafka_group_id
        self.builder = None
        if restart_from_uid:
            self.restart(restart_from_uid)
        else:
            self.start()
        self.re_manager = REManagerAPI(http_server_uri=self.server_host)
        self.re_manager.set_authorization_key(api_key=self.api_key)

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
    def api_key(self):
        """
        Key for API security.
        """
        ...

    @property
    @abstractmethod
    def measurement_plan_name(self) -> str:
        """String name of registered plan"""
        ...

    @abstractmethod
    def measurement_plan_args(self, point) -> list:
        """List of arguments to pass to plan from a point to measure."""
        ...

    @abstractmethod
    def measurement_plan_kwargs(self, point) -> dict:
        """Construct dictionary of keyword arguments to pass the plan, from a point to measure."""
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
        for point in next_points:
            plan = BPlan(
                self.measurement_plan_name,
                *self.measurement_plan_args(point),
                **self.measurement_plan_kwargs(point),
            )
            r = self.re_manager.item_add(plan, pos=self._queue_add_position)
            logging.info(f"Sent http-server request for point {point}\n." f"Received reponse: {r}")
        # TODO: Should I be checking responses for anything?
        #  (Continue on 200, crash on anything else. Probably sensible stuff in requests package.)
        return doc

    def _write_event(self, stream, doc):
        """Add event to builder as event page, and publish to catalog"""
        if stream in self.builder._streams:
            self.builder.add_data(stream, data=doc)
        else:
            self.builder.add_stream(stream, data=doc)
            self.agent_catalog.v1.insert(*self.builder._cache._ordered[-2])  # Add descriptor for first time
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
            doc["exp_uid"] = uid
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

    def restart(self, uid):
        """
        Restart agent using all knowledge of a previous agent.
        Right now this uses an iteration over all previously exposed data and singular `tell` methods
        for two reasons:
            1. If agent parameters change how `tell` operates, then the exposure should be fresh.
            2. With current Tiled implementations, bulk loading has been buggy on analyzed data.
                For this reason, `tell` is used instead of `tell_many`, and should be addressed
                in future developments of agents.

        Parameters
        ----------
        uid : str
            Previous agent start document uid.
        """
        self.metadata["restarted_from_start"] = uid
        run = self.agent_catalog[uid]
        descriptors = {}  # {uid: "ask"/"tell"}
        load_uids = []
        for name, doc in run.documents():
            if name == "descriptor":
                descriptors[doc["uid"]] = doc["name"]
            if name == "event_page":
                if descriptors[doc["descriptor"]] == "tell":
                    load_uids.extend(doc["uid"])

        # Assemble all docs and knowledge first.
        docs = []
        for exp_uid in load_uids:
            run = self.exp_catalog[exp_uid]
            independent_variable, dependent_variable = self.unpack_run(run)
            logging.debug("Telling agent about some new data.")
            doc = self.tell(independent_variable, dependent_variable)
            docs.append(doc)

        # Then start and flush event docs while listening to kafka.
        self.start()
        for doc in docs:
            self._write_event("tell", doc)

    def stop(self, exit_status="success", reason=""):
        logging.debug("Attempting agent stop.")
        self.builder.close(exit_status=exit_status, reason=reason)
        self.agent_catalog.v1.insert("stop", self.builder._cache.stop_doc)
        self.kafka_dispatcher.stop()
        logging.info(
            f"Stopped agent with exit status {exit_status.upper()}"
            f"{(' for reason: ' + reason) if reason else '.'}"
        )

    def signal_handler(self, signal, frame):
        self.stop(exit_status="abort", reason="forced exit ctrl+c")
        sys.exit(0)


def example_run():
    """Example function for running an agent."""
    try:
        agent = Agent(beamline_tla="tla")
        signal.signal(signal.SIGINT, agent.signal_handler)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e


class DrowsyAgent(Agent):
    """
    It's an agent that just lounges around all day.
    Alternates sending args vs kwargs to do the same thing.
    """

    server_host = "qserver1.nslsl2.bnl.gov:60611"
    measurement_plan_name = "agent_driven_nap"

    def __init__(self, *, beamline_tla: str):
        super().__init__(beamline_tla=beamline_tla)
        self.counter = 0

    def measurement_plan_kwargs(self, point) -> dict:
        if self.counter % 2 == 0:
            return dict(delay_kwarg=1)
        else:
            return {}

    def measurement_plan_args(self, point) -> list:
        if self.counter % 2 == 0:
            return []
        else:
            return [1]

    def unpack_run(run: databroker.client.BlueskyRun):
        return 0.0, 0.0

    def tell(self, x, y) -> dict:
        return dict(x=x, y=y)

    def ask(self, batch_size: int) -> Tuple[dict, Sequence]:
        return dict(batch_size=batch_size), [0.0]

    def report(self):
        pass
