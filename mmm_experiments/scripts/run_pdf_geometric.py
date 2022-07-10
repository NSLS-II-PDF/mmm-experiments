import logging
import signal

from mmm_experiments.agents.pdf import GeometricAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = GeometricAgent(
            sample_origin=(69.2, 2.0),
            relative_bounds=(-30, 30),
            metadata={"Yogurt": 1, "Froyo": 52},
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
