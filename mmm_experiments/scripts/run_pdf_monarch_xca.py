"""This agent is designed to run along side a geometric grid, and then after some time take over.
The take over can be accomplished by the user triggering `enable_continuous_suggesting`,
Requests can be made to BMM using `add_suggestions_to_subject_queues`.
"""
import logging
import signal
import time
from pathlib import Path

import numpy as np

from mmm_experiments.agents import bmm
from mmm_experiments.agents.joint import PDFXCAMonarch


class BMMSubject(bmm.RandomAgent):
    """Subject agent when needed"""


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        subject_agent = BMMSubject(
            origin=(183.818, 122.319),
            relative_bounds=(-28, 37),
            metadata=dict(init_time=time.time(), notes="Subject for PDF Monarch."),
            sample_number=9,
            direct_to_queue=True,
        )
        subject_agent.queue_add_position = "front"

        agent = PDFXCAMonarch(
            subjects=[subject_agent],
            direct_subjects_on_tell=False,
            sample_origin=(67.2, 92.0),
            relative_bounds=(-30.2, 36.8),
            botorch_device="cuda:2",
            xca_device="cuda:3",
            model_qspace=np.linspace(0.065, 7.89, 3000),
            model_checkpoint=Path(__file__).parents[1] / "models" / "2022-nov" / "low_q_low_fidelity.ckpt",
            metadata=dict(
                init_time=time.time(),
                notes="BMM Monarch using spectral distance scientific value agent. "
                "Doubles all measurements on PDF.",
            ),
            ask_on_tell=False,
            direct_to_queue=True,
            report_on_tell=True,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
