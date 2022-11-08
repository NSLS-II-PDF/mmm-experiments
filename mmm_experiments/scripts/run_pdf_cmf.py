import logging
import signal

from mmm_experiments.agents.pdf import CMFAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = CMFAgent(
            sample_origin=(69.2, 2.0),
            relative_bounds=(-30, 30),
            metadata={},
            num_components=5,
            ask_mode="unconstrained",
            ask_on_tell=False,
            report_on_tell=False,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
