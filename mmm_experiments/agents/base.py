import logging
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import databroker.client
import nslsii
import numpy as np
from bluesky_kafka import Publisher
from bluesky_live.run_builder import RunBuilder
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.http import REManagerAPI
from tiled.client import from_profile
from xkcdpass import xkcd_password as xp

from mmm_experiments.adjudicators.msg import DEFAULT_NAME as ADJUDICATOR_STREAM_NAME
from mmm_experiments.adjudicators.msg import AdjudicatorMsg, Suggestion
from mmm_experiments.data.kafka_consumer import AgentConsumer

PASSWORD_LIST = xp.generate_wordlist(wordfile=xp.locate_wordfile(), min_length=3, max_length=8)


class Agent(ABC):
    """
    Single Plan Agent. These agents should consume data, decide where to measure next,
    and execute a single type of plan (maybe, move and count).

    Base agent sets up a kafka subscription to listen to new stop documents, a catalog to read for experiments,
    a catalog to write agent status to, and a manager API for the HTTP server.

    Parameters
    ----------
    beamline_tla: str
        Beamline three letter acronym. Used in setting up the connection to kafka and tiled server
    metadata: Optional[dict]
        Optional extra metadata to add
    ask_on_tell: bool
        Whether to ask for new points every time an agent is told about new data.
        To create a truly passive agent, it is best to implement ask as a method that does nothing.
        To create an agent that only suggests new points periodically or on another trigger, `ask_on_tell`
        should be set to False.
    direct_to_queue: Optional[bool]
        Whether the agent suggestions will be placed directly on the queue. If false,
        the suggestions will be sent to a Kafka topic for an Adjudicator to process.
    report_on_tell : bool
        Whether to create a report every time an agent is told about new data.
    default_report_kwargs : dict
        Default kwargs for generating automatic reports.
    """

    def __init__(
        self,
        *,
        beamline_tla: str,
        metadata: Optional[dict] = None,
        ask_on_tell: bool = True,
        direct_to_queue: Optional[bool] = True,
        report_on_tell: bool = False,
        default_report_kwargs=None,
    ):
        logging.debug("Initializing Agent")
        self.kafka_config = nslsii._read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")
        self.kafka_group_id = f"echo-{beamline_tla}-{str(uuid.uuid4())[:8]}"
        # Each agent attends to a data topic and a directive topic. (PDF to an analyzed data topic)
        topics = [
            f"{beamline_tla}.mmm.bluesky.agents",
            f"{beamline_tla}.bluesky.pdfstream.documents"
            if beamline_tla == "pdf"
            else f"{beamline_tla}.bluesky.runengine.documents",
        ]
        self.kafka_consumer = AgentConsumer(
            topics=topics,
            bootstrap_servers=",".join(self.kafka_config["bootstrap_servers"]),
            group_id=self.kafka_group_id,
            consumer_config=self.kafka_config["runengine_producer_config"],
            agent=self,
            beamline_tla=beamline_tla,
            bluesky_callbacks=(self._on_stop_router,),
        )
        self.kafka_producer = Publisher(
            topic=f"{beamline_tla}.mmm.bluesky.adjudicators",
            bootstrap_servers=",".join(self.kafka_config["bootstrap_servers"]),
            key="",
            producer_config=self.kafka_config["runengine_producer_config"],
        )
        logging.debug("Kafka setup sucessfully.")
        exp_profile_name = "pdf_bluesky_sandbox" if beamline_tla == "pdf" else beamline_tla
        self.exp_catalog = from_profile(exp_profile_name)
        logging.info(f"Reading data from catalog: {self.exp_catalog}")
        agent_profile_name = f"{beamline_tla}_bluesky_sandbox"
        self.agent_catalog = from_profile(agent_profile_name)
        logging.info(f"Writing data to catalog: {self.agent_catalog}")
        self.metadata = metadata or {}
        self.metadata["beamline_tla"] = self.beamline_tla = beamline_tla
        self.metadata["kafka_group_id"] = self.kafka_group_id
        self.metadata[
            "agent_name"
        ] = self.agent_name = f"{self.name}-{xp.generate_xkcdpassword(PASSWORD_LIST, numwords=2, delimiter='-')}"
        self._ask_on_tell = ask_on_tell
        self._report_on_tell = report_on_tell
        self.default_report_kwargs = {} if default_report_kwargs is None else default_report_kwargs
        self.builder = None
        self.re_manager = REManagerAPI(http_server_uri=self.server_host)
        self.re_manager.set_authorization_key(api_key=self.api_key)
        self._queue_add_position = "back"
        self._direct_to_queue = direct_to_queue
        self.default_plan_md = dict(
            agent_name=self.agent_name,
            agent_class=str(type(self)),
            tiled_data_profile=exp_profile_name,
            tiled_agent_profile=agent_profile_name,
        )

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
    def measurement_origin(self):
        """Distinctly useful for having twinned samples and mixin classes. The origin of independent variable."""
        ...

    @property
    @abstractmethod
    def relative_bounds(self):
        """Relative measurement bounds to consider for the experiment"""
        ...

    @property
    @abstractmethod
    def measurement_plan_name(self) -> str:
        """String name of registered plan"""
        ...

    @staticmethod
    @abstractmethod
    def measurement_plan_args(point) -> list:
        """
        List of arguments to pass to plan from a point to measure.
        This is a good place to transform relative into absolute motor coords.
        """
        ...

    @staticmethod
    @abstractmethod
    def measurement_plan_kwargs(point) -> dict:
        """
        Construct dictionary of keyword arguments to pass the plan, from a point to measure.
        This is a good place to transform relative into absolute motor coords.
        """
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

    def report(self, **kwargs) -> dict:
        """
        Create a report given the data observed by the agent.
        This could be potentially implemented in the base class to write document stream.
        Additional functionality for converting the report dict into an image or formatted report is
        the duty of the child class.
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
    def queue_add_position(self) -> Union[int, Literal["front", "back"]]:
        return self._queue_add_position

    @queue_add_position.setter
    def queue_add_position(self, position: Union[int, Literal["front", "back"]]):
        self._queue_add_position = position

    def update_priority(self, position: Union[int, Literal["front", "back"]]):
        self.queue_add_position = position

    @property
    def ask_on_tell(self) -> bool:
        return self._ask_on_tell

    @ask_on_tell.setter
    def ask_on_tell(self, flag: bool):
        self._ask_on_tell = flag

    @property
    def report_on_tell(self) -> bool:
        return self._ask_on_tell

    @report_on_tell.setter
    def report_on_tell(self, flag: bool):
        self._report_on_tell = flag

    def enable_continuous_reporting(self):
        """Enable agent to report each time it receives data."""
        self.report_on_tell = True

    def disable_continuous_reporting(self):
        """Disable agent to report each time it receives data."""
        self.report_on_tell = False

    def enable_continuous_suggesting(self):
        """Enable agent to suggest new points to the queue each time it receives data."""
        self.ask_on_tell = True

    def disable_continuous_suggesting(self):
        """Disable agent to suggest new points to the queue each time it receives data."""
        self.ask_on_tell = False

    def enable_direct_to_queue(self):
        self._direct_to_queue = True

    def disable_direct_to_queue(self):
        self._direct_to_queue = False

    @property
    def name(self) -> str:
        """Short string name"""
        return "agent"

    def _add_to_queue(self, next_points, uid):
        """
        Soft wrapper for the `ask` method that puts the agent's
        proposed next points on the queue.

        Parameters
        ----------
        next_points : Iterable
            New points to measure
        uid : str

        Returns
        -------

        """
        for point in next_points:
            kwargs = self.measurement_plan_kwargs(point)
            kwargs.setdefault("md", {})
            kwargs["md"].update(self.default_plan_md)
            kwargs["md"]["agent_ask_uid"] = uid
            plan = BPlan(
                self.measurement_plan_name,
                *self.measurement_plan_args(point),
                **kwargs,
            )
            r = self.re_manager.item_add(plan, pos=self.queue_add_position)
            logging.debug(f"Sent http-server request for point {point}\n." f"Received reponse: {r}")
        return

    def _check_queue_and_start(self):
        """
        If the queue runs out of plans, it will stop.
        That is, adding a plan to an empty queue will not run the plan.
        This will not be an issue when there are many agents adding plans to a queue.
        Giving agents the autonomy to start the queue is a risk that will be mitigated by
        only allowing the beamline scientists to open and close the environment.
        A queue cannot be started in a closed environment.
        """
        status = self.re_manager.status(reload=True)
        if (
            status["items_in_queue"] == 1
            and status["worker_environment_exists"] is True
            and status["manager_state"] == "idle"
        ):
            self.re_manager.queue_start()
            logging.info("Agent is starting an idle queue with exactly 1 item.")

    def _write_event(self, stream, doc):
        """Add event to builder as event page, and publish to catalog"""
        uid = str(uuid.uuid4())
        if not doc:
            return
        if stream in self.builder._streams:
            self.builder.add_data(stream, data=doc)
        else:
            self.builder.add_stream(stream, data=doc)
            self.agent_catalog.v1.insert(*self.builder._cache._ordered[-2])  # Add descriptor for first time
        self.agent_catalog.v1.insert(*self.builder._cache._ordered[-1])
        return uid

    def add_suggestions_to_queue(self, batch_size: int):
        """Calls ask, adds suggestions to queue, and writes out event"""
        logging.debug("Issuing ask and adding to the queue.")
        doc, next_points = self.ask(batch_size)
        uid = self._write_event("ask", doc)
        self._add_to_queue(next_points, uid)
        self._check_queue_and_start()

    def generate_suggestions_for_adjudicator(self, batch_size: int):
        logging.debug("Issuing ask and sending to adjudicator.")
        doc, next_points = self.ask(batch_size)
        uid = self._write_event("ask", doc)
        suggestions = []
        for point in next_points:
            kwargs = self.measurement_plan_kwargs(point)
            kwargs.setdefault("md", {})
            kwargs["md"].update(self.default_plan_md)
            kwargs["md"]["agent_ask_uid"] = uid
            suggestions.append(
                Suggestion(
                    ask_uid=uid,
                    plan_name=self.measurement_plan_name,
                    args=self.measurement_plan_args(point),
                    kwargs=self.measurement_plan_kwargs(point),
                )
            )
        msg = AdjudicatorMsg(
            agent_name=self.agent_name,
            suggestions_uid=str(uuid.uuid4()),
            suggestions={self.beamline_tla: suggestions},
        )
        self.kafka_producer(ADJUDICATOR_STREAM_NAME, msg.dict())

    def generate_report(self, **kwargs):
        logging.debug("Issuing report request and writing to Mongo.")
        doc = self.report(**kwargs)
        self._write_event("report", doc)

    @staticmethod
    def trigger_condition(uid) -> bool:
        return True

    def _on_stop_router(self, name, doc):
        """Service that runs each time a stop document is seen."""
        if name == "stop":
            uid = doc["run_start"]
            if not self.trigger_condition(uid):
                logging.debug(
                    f"New data detected, but trigger condition not met. "
                    f"The agent will ignore this start doc: {uid}"
                )
                return

            logging.info(
                f"New data detected, telling the agent about this start doc "
                f"and asking for a new suggestion: {uid}"
            )
            run = self.exp_catalog[uid]
            try:
                independent_variable, dependent_variable = self.unpack_run(run)
            except KeyError as e:
                logging.warning(f"Ignoring key error in unpack for data {uid}:\n {e}")
                return

            # Tell
            logging.debug("Telling agent about some new data.")
            doc = self.tell(independent_variable, dependent_variable)
            doc["exp_uid"] = [uid]
            self._write_event("tell", doc)

            # Ask
            if self.report_on_tell:
                self.generate_report(**self.default_report_kwargs)
            if self.ask_on_tell:
                if self._direct_to_queue:
                    self.add_suggestions_to_queue(1)
                else:
                    self.generate_suggestions_for_adjudicator(1)

    def tell_agent_by_uid(self, uids: Iterable):
        """Give an agent an iterable of uids to learn from.
        This is an optional behavior for priming an agent without a complete restart."""
        for uid in uids:
            run = self.exp_catalog[uid]
            independent_variable, dependent_variable = self.unpack_run(run)
            doc = self.tell(independent_variable, dependent_variable)
            doc["exp_uid"] = [uid]
            self._write_event("tell", doc)

    def start(self, ask_at_start=False):
        logging.debug("Issuing start document and start listening to Kafka")
        self.builder = RunBuilder(metadata=self.metadata)
        self.agent_catalog.v1.insert("start", self.builder._cache.start_doc)
        logging.info(f"Agent name={self.builder._cache.start_doc['agent_name']}")
        logging.info(f"Agent start document uuid={self.builder._cache.start_doc['uid']}")
        if ask_at_start:
            self.add_suggestions_to_queue(1)
        self.kafka_consumer.start()

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
        self.kafka_consumer.stop()
        logging.info(
            f"Stopped agent with exit status {exit_status.upper()}"
            f"{(' for reason: ' + reason) if reason else '.'}"
        )

    def signal_handler(self, signal, frame):
        self.stop(exit_status="abort", reason="forced exit ctrl+c")
        sys.exit(0)


class SequentialAgentMixin:
    """Mixin to be used with Agent children"""

    def __init__(self, *, step_size: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.step_size = step_size
        self.relative_min, self.relative_max = self.relative_bounds
        self.independent_cache = []

    def tell(self, position, y) -> dict:
        relative_position = position - self.measurement_origin
        self.independent_cache.append(relative_position)
        return dict(position=[position], rel_position=[relative_position], cache_len=[len(self.independent_cache)])

    def ask(self, batch_size: int = 1) -> Tuple[dict, Sequence]:
        if self.independent_cache:
            last = self.independent_cache[-1]
            point = last + self.step_size
            if point > self.relative_max:
                point = self.relative_min
        else:
            last = 0.0
            point = 0.0
        doc = dict(last_point=[last], next_point=[point])
        return doc, [point]


class GeometricResolutionMixin(SequentialAgentMixin):
    """
    Mixin to be used with Agent children.
    Performs a gridsearch with increasing resolution
    Uses `tell` method of SequentialAgentMixin
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points_per_batch = 0
        self.independent_cache = []
        self.acumulated_stops = 0
        self.first_ask = True

    def ask(self, batch_size: int = 1) -> Tuple[dict, Sequence]:
        if self.first_ask:
            self.first_ask = False
            points = [
                self.relative_min,
                self.relative_min + (self.relative_max - self.relative_min) / 2,
                self.relative_max,
            ]
            doc = dict(ask_ready=[False], acummulated_stops=[self.acumulated_stops], proposal=[points])
            self.points_per_batch = 3
            return doc, points

        self.acumulated_stops += 1
        if self.acumulated_stops == self.points_per_batch:
            # 3 then 2 then geometric expansion
            self.acumulated_stops = 0
            self.independent_cache = sorted(self.independent_cache)
            points = []
            for i in range(len(self.independent_cache) - 1):
                points.append(
                    self.independent_cache[i] + ((self.independent_cache[i + 1] - self.independent_cache[i]) / 2)
                )
            self.points_per_batch = len(points)
            doc = dict(
                ask_ready=[True] * len(points),
                acummulated_stops=[self.acumulated_stops] * len(points),
                proposal=points,
            )
        else:
            doc = dict(ask_ready=[False], acummulated_stops=[self.acumulated_stops], proposal=[-10000.0])
            points = []
        return doc, points

    def _check_queue_and_start(self):
        """
        Override the exaclty 1 rule and always start the queue if its idle and I just added plans.
        """
        status = self.re_manager.status(reload=True)
        if status["worker_environment_exists"] is True and status["manager_state"] == "idle":
            self.re_manager.queue_start()
            logging.info("Agent is starting an idle queue.")

    @property
    def name(self):
        return "geometric"


class RandomAgentMixin:
    """
    Hears a stop document and immediately suggests a random point within the bounds.
    Mixin to be used with Agent Children
    """

    def tell(self, x, y):
        return {}

    def ask(self, batch_size: int = 1) -> Tuple[dict, Sequence]:
        point = np.random.uniform(*self.relative_bounds)
        doc = dict(next_point=[point])
        return doc, [point]

    @property
    def name(self):
        return "random"


class DrowsyAgent(Agent, ABC):
    """
    It's an agent that just lounges around all day.
    Alternates sending args vs kwargs to do the same thing.
    """

    measurement_plan_name = "agent_driven_nap"

    def __init__(self, *, beamline_tla: str):
        super().__init__(beamline_tla=beamline_tla)
        self.counter = 0

    def measurement_plan_kwargs(self, point) -> dict:
        if self.counter % 2 == 0:
            return dict(delay=1.2)
        else:
            return {}

    def measurement_plan_args(self, point) -> list:
        if self.counter % 2 == 0:
            return []
        else:
            return [1.2]

    @staticmethod
    def unpack_run(run: databroker.client.BlueskyRun):
        return [0.0], [0.0]

    def tell(self, x, y) -> dict:
        return dict(x=x, y=y)

    def ask(self, batch_size: int) -> Tuple[dict, Sequence]:
        self.counter += 1
        logging.debug(f"Counter={self.counter}")
        return dict(batch_size=[batch_size]), [0.0 for _ in range(batch_size)]

    def report(self):
        pass

    def measurement_origin(self):
        pass

    def relative_bounds(self):
        pass

    @property
    def name(self):
        return "drowsy"
