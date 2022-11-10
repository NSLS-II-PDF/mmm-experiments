"""
This agent is designed to run along side a geometric grid and develop a wholistic vision of the system.
When the CMF decomposition is adequate, it can be used to trigger a batch on BMM representing
the most distinct regions.
Requests can be made to BMM using `add_suggestions_to_subject_queues`.
"""
import logging
import signal
import time

from mmm_experiments.agents import bmm
from mmm_experiments.agents.joint import PDFCMFMonarch


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
            direct_to_queue=True,
            exp_filename="MultimodalMadness",
        )
        subject_agent.queue_add_position = "front"

        agent = PDFCMFMonarch(
            subjects=[subject_agent],
            direct_subjects_on_tell=False,
            sample_origin=(67.2, 92.0),
            relative_bounds=(-30.2, 36.8),
            num_components=7,
            ask_mode="unconstrained",
            metadata=dict(init_time=time.time(), notes="PDF Monarch conducting CMF on the whole dataset."),
            ask_on_tell=False,
            direct_to_queue=False,
            report_on_tell=True,
            lustre_path="/nsls2/data/pdf/scratch/mmm_stuff",
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
