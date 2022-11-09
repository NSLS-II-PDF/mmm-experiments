import logging
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from threading import Lock
import pprint

from bluesky_kafka import BlueskyConsumer
from bluesky_queueserver_api import BPlan
from bluesky_queueserver_api.http import REManagerAPI

from mmm_experiments.adjudicators.msg import AdjudicatorMsg
from mmm_experiments.devices.switchboard import SwitchBoardBackend

try:
    from nslsii import _read_bluesky_kafka_config_file
except ImportError:
    from nslsii.kafka_utils import _read_bluesky_kafka_config_file

logger = logging.getLogger(name="mmm.adjudicators")


class DequeSet:
    def __init__(self, maxlen=100):
        self._set = set()
        self._dequeue = deque()
        self._maxlen = maxlen

    def __contains__(self, d):
        return d in self._set

    def append(self, d):
        if d in self:
            raise ValueError("do not add the same value twice")
        self._set.add(d)
        self._dequeue.append(d)
        while len(self._dequeue) >= self._maxlen:
            self._dequeue.popleft()
            # self._set.remove(discared)


class AdjudicatorBase(BlueskyConsumer, ABC):
    """
    An agent adjudicator that listens to published suggestions by agents.
    This Base approach (as per `process_document`) only retains the most recent suggestions by any named agents.
    """

    def __init__(self, *args, tla: str, switchboard_prefix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Lock()
        self._thread = None
        self._tla = tla.lower()
        switchboard_prefix = switchboard_prefix if switchboard_prefix is not None else f"{tla}:switchboard:"
        self._switchboard = SwitchBoardBackend(switchboard_prefix, name="switchboard")
        self._current_suggestions = {}  # agent_name: AdjudicatorMsg
        self.re_manager = REManagerAPI(http_server_uri=self.server_host)
        self.re_manager.set_authorization_key(api_key=self.api_key)
        self._uid_deque_set = DequeSet()

    
    def start(self, *args, **kwargs):
        self._callback_id = self._switchboard.publish_to_queue.subscribe(self.switchboard_callback)
        try:
            super().start(*args, **kwargs)
        finally:
            self._switchboard.publish_to_queue.unsubscribe(self._callback_id)

    # need kwargs because the caller passes a bunch of stuff in
    def switchboard_callback(self, value, **kwargs):
        """Process to run when switch board signal is set to on."""
        if value == 0:
            return
        try:
            self._switchboard.adjudicate_status.set(f"Starting work on {value}").wait()
            self.make_judgements(value)
        finally:
            self._switchboard.publish_to_queue.set(0).wait()
            self._switchboard.adjudicate_status.set("Done").wait()

    def process_document(self, topic, name, doc):
        if name != "agent_suggestions":
            return True
        with self._lock:
            logger.info(f"{doc['agent_name']=}, {doc['suggestions_uid']=}")
            self._current_suggestions[doc["agent_name"]] = AdjudicatorMsg(**doc)

    @property
    def current_suggestions(self):
        """Dictionary of {agent_name:AdjudicatorMsg}, deep copied at each grasp."""
        with self._lock:
            ret = deepcopy(self._current_suggestions)
        return ret

    @property
    def agent_names(self):
        with self._lock:
            ret = list(self._current_suggestions.keys())
        return ret

    @abstractmethod
    def make_judgements(self, value):
        """How to make decisions when given the call of duty"""
        ...

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

    def _add_suggestion_to_queue(self, agent_name, suggestion):
        if suggestion.ask_uid in self._uid_deque_set:
            logger.warning(
                f"Ask uid {suggestion.ask_uid} has already been seen. Not adding anything to the queue."
            )
            return
        else:
            self._uid_deque_set.append(suggestion.ask_uid)
        kwargs = dict(suggestion.plan_kwargs)
        kwargs.setdefault("md", {})
        kwargs["md"]["agent_ask_uid"] = suggestion.ask_uid
        kwargs["md"]["agent_name"] = agent_name
        plan = BPlan(suggestion.plan_name, *suggestion.plan_args, **kwargs)
        r = self.re_manager.item_add(plan, pos="back")
        logger.debug(f"Sent http-server request by adjudicator\n." f"Received reponse: {r}")


class AgentByModeAdjudicator(AdjudicatorBase):
    def make_judgements(self, value):
        agent_name = self._switchboard.adjudicate_mode.get()
        try:
            adjudicator_msg = self.current_suggestions[agent_name]
        except KeyError:
            logger.warning(f"Agent {agent_name} not known to the Adjudicator. "
            f"Known agenst are {list(self.current_suggestions)!r}")
        else:
            for suggestion in adjudicator_msg.suggestions[self._tla]:
                self._add_suggestion_to_queue(agent_name, suggestion)


class NoisyButSafeAdjudicator(AgentByModeAdjudicator):
    def _add_suggestion_to_queue(self, agent_name, suggestion):
        if suggestion.ask_uid in self._uid_deque_set:
            logger.warning(
                f"Ask uid {suggestion.ask_uid} has already been seen. Not adding anything to the queue."
            )
            return
        else:
            self._uid_deque_set.append(suggestion.ask_uid)
        kwargs = dict(suggestion.plan_kwargs)
        kwargs.setdefault('md', {})
        kwargs["md"]["agent_ask_uid"] = suggestion.ask_uid
        kwargs["md"]["agent_name"] = agent_name
        pprint.pprint((suggestion.plan_name, suggestion.plan_args, kwargs))


if __name__ == "__main__":
    import uuid

    kafka_config = _read_bluesky_kafka_config_file(config_file_path="/etc/bluesky/kafka.yml")
    tla = "pdf"

    class SimpleAdjudicator(NoisyButSafeAdjudicator):
        server_host = "https://qserver.nsls2.bnl.gov/pdf"
        api_key = "yyyyy"

    adjudicator = SimpleAdjudicator(
        tla=tla,
        topics=[f"{tla}.mmm.bluesky.adjudicators"],
        bootstrap_servers=",".join(kafka_config["bootstrap_servers"]),
        group_id=f"echo-{tla}-{str(uuid.uuid4())[:8]}",
        consumer_config={"auto.offset.reset": 'smallest', **kafka_config["runengine_producer_config"]},
        switchboard_prefix='XF:28ID1-DA{SB:1}'
    )
    adjudicator._switchboard.wait_for_connection()
    print(adjudicator._switchboard.read())
    adjudicator.start()
