import logging
import signal
import time
from pathlib import Path

import numpy as np

from mmm_experiments.agents.pdf import XCAActiveAgent

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        agent = XCAActiveAgent(
            sample_origin=(67.2, 92.0),
            relative_bounds=(-30.2, 36.8),
            metadata=dict(init_time=time.time(), notes="Overnight passive run on half wafer"),
            botorch_device="cuda:2",
            xca_device="cuda:3",
            model_qspace=np.linspace(0.065, 7.89, 3000),
            model_checkpoint=Path(__file__).parents[1] / "models" / "2022-nov" / "low_q_low_fidelity.ckpt",
            ask_on_tell=True,
            direct_to_queue=False,
            report_on_tell=True,
            lustre_path="/nsls2/data/pdf/scratch/mmm_stuff",
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
