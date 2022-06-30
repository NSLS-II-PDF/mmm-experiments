import logging
import uuid

import numpy as np
from bluesky_live.run_builder import RunBuilder
from tiled.client import from_profile


class DummyAgent:
    """Built to mirror agents.base.Agent functionality"""

    def __init__(self, metadata=None, with_database=True, restart_from_uid=None):
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        self.with_database = with_database
        if self.with_database:
            self.catalog = from_profile("testing_sandbox")
        self.builder = None
        self.run = None
        if restart_from_uid:
            agent._restart(restart_from_uid)

    def start(self):
        self.builder = RunBuilder(metadata=self.metadata)
        logging.info(f"Agent start document uuid={self.builder._cache.start_doc['uid']}")
        if self.with_database:
            self.catalog.v1.insert("start", self.builder._cache.start_doc)
        self.run = self.builder.get_run()

    def spin(self):
        for i in range(3):
            doc = dict(
                position=[75.5],
                rel_position=[0.3],
                uid=[str(uuid.uuid4())],
                intensity=[np.random.random(10)[None, :]],
                prediction_prob=[np.random.rand(4)[None, :]],
            )
            if "tell" in self.builder._streams:
                self.builder.add_data("tell", data=doc)
                if self.with_database:  # Just event page
                    self.catalog.v1.insert(*self.builder._cache._ordered[-1])
            else:
                self.builder.add_stream("tell", data=doc)
                if self.with_database:  # Descriptor for the first time
                    self.catalog.v1.insert(*self.builder._cache._ordered[-2])
                    self.catalog.v1.insert(*self.builder._cache._ordered[-1])

            doc = dict(next_points=[0.4, 0.5])
            if "ask" in self.builder._streams:
                self.builder.add_data("ask", data=doc)
                if self.with_database:  # Just event page
                    self.catalog.v1.insert(*self.builder._cache._ordered[-1])
            else:
                self.builder.add_stream("ask", data=doc)
                if self.with_database:  # Descriptor for the first time
                    self.catalog.v1.insert(*self.builder._cache._ordered[-2])
                    self.catalog.v1.insert(*self.builder._cache._ordered[-1])

    def signal_handler(self, signal, frame):
        self.stop(exit_status="abort", reason="forced exit ctrl+c")
        self.print_run()
        sys.exit(0)

    def print_run(self):
        for name, doc in self.run.documents(fill="no"):
            print(name)
            print(doc)

    def stop(self, exit_status="success", reason=""):
        self.builder.close(exit_status=exit_status, reason=reason)
        self.catalog.v1.insert("stop", self.builder._cache.stop_doc)

    def _restart(self, uid):
        """Pretend restart from previous agent uid"""
        self.metadata["restarted_from_start"] = uid
        run = self.catalog[uid]
        """Doing this restart the iterative way, to avoid merge errors with tiled for now."""
        descriptors = {}  # {uid: "ask"/"tell"}
        load_uids = []
        for name, doc in run.documents():
            if name == "descriptor":
                descriptors[doc["uid"]] = doc["name"]
            if name == "event_page":
                if descriptors[doc["descriptor"]] == "tell":
                    load_uids.extend(doc["uid"])

        agent.start()
        print("If I were actually restarting something, I would load the following uids:")
        for uid in load_uids:
            print(uid)


if __name__ == "__main__":
    import signal
    import sys

    try:
        # Run an agent that still closes out the data stream on bad exit.
        agent = DummyAgent()
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start()
        agent.spin()
        agent.stop()
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
    finally:
        agent.print_run()
