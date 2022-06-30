import numpy as np
from bluesky_live.run_builder import RunBuilder


class DummyAgent:
    """Built to mirror agents.base.Agent functionality"""

    def __init__(self, metadata={}):
        self.metadata = metadata
        # self.catalog = ... # THIS  REQUIRES DATABASE SETUP

    def start(self):
        self.builder = RunBuilder(metadata=self.metadata)
        self.run = self.builder.get_run()

    def spin(self):
        for i in range(3):
            doc = dict(
                position=[75.5],
                rel_position=[0.3],
                intensity=[np.random.random(10)],
                prediction_prob=[np.random.rand(4)],
            )
            if "tell" in self.builder._streams:
                self.builder.add_data("tell", data=doc)
            else:
                self.builder.add_stream("tell", data=doc)
            doc = dict(next_points=[0.4, 0.5])
            if "ask" in self.builder._streams:
                self.builder.add_data("ask", data=doc)
            else:
                self.builder.add_stream("ask", data=doc)

            # ====THIS SECTION REQUIRES DATABASE SETUP====
            # This may be incredibly redundant.
            # A smarter/fast way should be considered.
            # for name, doc in self.run.documents(fill="no"):
            #     self.catalog.v1.insert(name, doc)
            #
            # ====THIS SECTION REQUIRES DATABASE SETUP====

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


if __name__ == "__main__":
    import signal
    import sys

    # Run an agent that still closes out the data stream on bad exit.
    agent = DummyAgent()
    signal.signal(signal.SIGINT, agent.signal_handler)
    try:
        agent.start()
        agent.spin()
        agent.stop()
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
    finally:
        agent.print_run()
