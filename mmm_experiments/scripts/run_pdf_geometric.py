import logging
import signal
import time

from mmm_experiments.agents.pdf import GeometricAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = GeometricAgent(
            sample_origin=(104, 92.0),
            relative_bounds=(-68, 0),
            metadata=dict(init_time=time.time(), notes="Overnight passive run on half wafer"),
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
