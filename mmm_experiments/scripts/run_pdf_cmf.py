import logging
import signal
import time

from mmm_experiments.agents.pdf import CMFAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = CMFAgent(
            sample_origin=(67.2, 92.0),
            relative_bounds=(-30.2, 36.8),
            metadata=dict(init_time=time.time(), notes="Overnight passive run on half wafer"),
            num_components=7,
            ask_mode="unconstrained",
            ask_on_tell=True,
            report_on_tell=True,
            direct_to_queue=False,
            lustre_path="/nsls2/data/pdf/scratch/mmm_stuff",
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
