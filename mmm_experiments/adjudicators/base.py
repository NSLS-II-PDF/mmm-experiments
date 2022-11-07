import logging
from abc import ABC, abstractmethod
from collections import deque
from threading import Lock, Thread

from bluesky_kafka import BlueskyConsumer

from mmm_experiments.devices.switchboard import SwitchBoardBackend

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
            discared = self._dequeue.popleft()
            self._set.remove(discared)


class Adjudicator(BlueskyConsumer, ABC):
    """An agent adjudicator that listens to published suggestions by agents."""

    def __init__(self, *args, tla: str, switchboard_prefix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Lock()
        self._thread = None
        self._tla = tla.lower()
        switchboard_prefix = switchboard_prefix if switchboard_prefix is not None else f"{tla}:switchboard:"
        self._switchboard = SwitchBoardBackend(switchboard_prefix, name="switchboard")
        self._current_suggestions = {}  # agent_name: doc

    def start(self, *args, **kwargs):
        self._callback_id = self._switchboard.publish_to_queue.subscribe(self.switchboard_callback)
        try:
            super().start(*args, **kwargs)
        finally:
            self._switchboard.publish_to_queue.unsubscribe(self._callback_id)

    def spin(self):
        if self._thread is not None:
            raise RuntimeError("Adjudicator has already started spinning, this can't be done twice.")
        self._thread = Thread(target=self.start)
        self._thread.start()

    def switchboard_callback(self, value):
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
            self._current_suggestions[doc["agent_name"]] = doc

    @property
    def current_suggestions(self):
        with self._lock:
            ret = dict(self._current_suggestions)
        return ret

    @abstractmethod
    def make_judgements(self, value):
        """How to make decisions when given the call of duty"""
        ...
