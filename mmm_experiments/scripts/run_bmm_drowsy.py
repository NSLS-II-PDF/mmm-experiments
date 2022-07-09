import logging
import signal

from mmm_experiments.agents.bmm import DrowsyBMMAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = DrowsyBMMAgent()
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
