import msgpack
import logging
from confluent_kafka import Consumer

from threading import Lock, Thread
from collections import defaultdict
from bluesky_kafka import BlueskyConsumer

logger = logging.getLogger(name="mmm.kafka")


class LatestNews(BlueskyConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Lock()
        self._by_agent = {}
        self._thread = None

    def process_document(self, topic, name, doc):
        if name != "agent_suggestions":
            return True
        with self._lock:
            # TODO by beamline level union?
            # TODO provide a beamline-first view?
            print(f'{doc["agent"]=}, {doc["publish_uid"]=}')
            self._by_agent[doc["agent"]] = doc
        return True

    @property
    def by_agent(self):
        with self._lock:
            # TODO worry about deepcopy
            ret = dict(self._by_agent)
        return ret

    def yeet(self):
        if self._thread is not None:
            raise RuntimeError("you can only yeet once")

        self._thread = Thread(target=self.start)
        self._thread.start()
