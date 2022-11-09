import logging
import signal

from mmm_experiments.agents.joint import SequentialMonarchPDF

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = SequentialMonarchPDF(
            step_size=1.0,
            sample_origin=(70.0, 2.0),
            relative_bounds=(-30, 30),
            Cu_origin=(155.390, 83.96),
            Ti_origin=(155.381, 82.169),
            Cu_det_position=205,
            Ti_det_position=20,
            bmm_bounds=(-30, 23),
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.independent_cache.append(0.0)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
