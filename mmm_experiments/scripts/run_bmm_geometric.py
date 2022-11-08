import logging
import signal
import time

from mmm_experiments.agents.bmm import GeometricAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    try:
        agent = GeometricAgent(
            origin=(183.818, 122.319),
            relative_bounds=(-28, 37),
            metadata=dict(init_time=time.time(), notes="Agent run on half wafer as initial test"),
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.independent_cache.append(0.0)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
