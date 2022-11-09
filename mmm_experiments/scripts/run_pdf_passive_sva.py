import logging
import signal
import time

from mmm_experiments.agents.pdf import ScientificValue

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = ScientificValue(
            device="cuda:1",
            sample_origin=(104, 92.0),
            relative_bounds=(-68.0, 0.0),
            metadata=dict(init_time=time.time(), notes="Overnight passive run on half wafer"),
            ask_on_tell=True,
            direct_to_queue=False,
            report_on_tell=True,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
