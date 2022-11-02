import logging

from threading import Lock, Thread
from bluesky_kafka import BlueskyConsumer

from mmm_experiments.devices.switchboard import SwitchBoardBackend

logger = logging.getLogger(name="mmm.kafka")


class LatestNews(BlueskyConsumer):
    def __init__(self, *args, tla="PDF", switchboard_prefix="pdf:switchboard:", **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Lock()
        self._by_agent = {}
        self._thread = None
        self._tla = tla
        self._sb = SwitchBoardBackend(switchboard_prefix, name="sb")
        self._sb.publish.subscribe(self.on_process)

    def on_process(self, value, **kwargs):
        if value == 0:
            return
        try:
            self._sb.adjudicate_status.set(f"Starting work on: {value}").wait()
            self.do_work(value)
        finally:
            self._sb.publish.set(0).wait()
            self._sb.adjudicate_status.set("Done").wait()

    def do_work(self, value):
        raise NotImplementedError

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


class LazyDemo(LatestNews):
    def do_work(self, value):
        import time

        self._sb.adjudicate_status.set("starting work").wait()
        time.sleep(1)
        self._sb.adjudicate_status.set("still working").wait()
        time.sleep(1)
        # TODO think through how to do this mode better
        agent = self._sb.adjudicate_mode.get()
        by_agent = self.by_agent
        if agent not in by_agent:
            print("no agent found, not sure about this mode")
        else:
            for suggestion in by_agent[agent]["suggestions"][self._tla]:
                print(f"I suggest {suggestion}")


class AgentSelector(LatestNews):
    def do_work(self, value):
        # TODO maybe use read?
        # TODO think through how to do this mode better
        agent = self._sb.adjudicate_mode.get()
        self._sb.adjudicate_status.set(f"Listening to {agent}").wait()

        by_agent = self.by_agent
        if agent not in by_agent:
            print(f"{agent} not one we have heard of.  Know about {list(by_agent)}")
        else:
            for suggestion in by_agent[agent]["suggestions"][self._tla]:
                # TODO add to queue here!
                print(f"I suggest {suggestion}")


if __name__ == "__main__":
    import uuid
    import json

    judge = AgentSelector(
        ["suggestion.box"],
        "localhost:9092",
        str(uuid.uuid4()),
        deserializer=json.loads,
    )
    judge.start()
