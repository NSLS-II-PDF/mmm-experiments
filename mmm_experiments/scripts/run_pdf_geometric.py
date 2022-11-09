import logging
import signal
import time

from mmm_experiments.agents.pdf import GeometricAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = GeometricAgent(
            sample_origin=(67.2, 92.0),
            relative_bounds=(-30.2, 36.8),
            metadata=dict(init_time=time.time(), notes="Day Of work testing."),
            min_resolution=0.2,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
