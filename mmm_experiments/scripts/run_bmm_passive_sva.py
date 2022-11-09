import logging
import signal
import time

from mmm_experiments.agents.bmm import ScientificValue

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = ScientificValue(
            device="cpu",
            origin=(183.818, 122.319),
            relative_bounds=(-28, 37),
            metadata=dict(init_time=time.time(), notes="Agent run on half wafer as initial test"),
            ask_on_tell=False,
            report_on_tell=True,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
