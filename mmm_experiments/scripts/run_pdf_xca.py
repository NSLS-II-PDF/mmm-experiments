import logging
import signal
from pathlib import Path

import numpy as np

from mmm_experiments.agents.pdf import XCAActiveAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = XCAActiveAgent(
            sample_origin=(69.2, 2.0),
            relative_bounds=(-30, 30),
            metadata={},
            botorch_device="cuda:3",
            xca_device="cuda:2",
            model_qspace=np.linspace(0.065, 7.89, 3000),
            model_checkpoint=Path(__file__).parent / "models" / "2022-nov" / "low_q_low_fidelity.ckpt",
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=True)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
