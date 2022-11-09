import logging
import signal

from mmm_experiments.agents.pdf import SequentialAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = SequentialAgent(
            1.0,
            sample_origin=(70.0, 2.0),
            relative_bounds=(-30, 30),
            metadata={"Yogurt": 1, "Froyo": 52},
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.independent_cache.append(0.0)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
