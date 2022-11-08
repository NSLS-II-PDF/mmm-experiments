import logging
import signal
import time

from mmm_experiments.agents.pdf import GeometricAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = GeometricAgent(
            sample_origin=(105, 88.7),
            relative_bounds=(-68, 0),
            metadata=dict(init_time=time.time(), notes="Agent run on whole wafer as initial test"),
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
