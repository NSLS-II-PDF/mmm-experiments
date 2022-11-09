import logging
import signal
import time

from mmm_experiments.agents.pdf import CMFAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = CMFAgent(
            sample_origin=(104, 92.0),
            relative_bounds=(-68, 0),
            metadata=dict(init_time=time.time(), notes="Overnight passive run on half wafer"),
            num_components=4,
            ask_mode="unconstrained",
            ask_on_tell=True,
            report_on_tell=True,
            direct_to_queue=False,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
