import logging
import signal
import time

from mmm_experiments.agents.bmm import ScientificValue

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = ScientificValue(
            device="cuda:1",
            origin=(183.818, 122.319),
            relative_bounds=(-28, 37),
            metadata=dict(
                init_time=time.time(),
                notes="Overnight passive run on half wafer. Spectral distance scientific value agent.",
            ),
            ask_on_tell=True,
            direct_to_queue=False,
            report_on_tell=True,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
