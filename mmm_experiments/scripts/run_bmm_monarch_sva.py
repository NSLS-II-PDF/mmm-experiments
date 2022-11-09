import logging
import signal
import time

from mmm_experiments.agents import pdf
from mmm_experiments.agents.joint import BMMSVAMonarch


class PDFSubject(pdf.RandomAgent):
    """Subject agent with increased measurement time"""

    def measurement_plan_args(self, x_position) -> list:
        args = super().measurement_plan_args(x_position)
        args[-1] = 60
        return args


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    try:
        subject_agent = PDFSubject(
            sample_origin=(67.2, 92.0),
            relative_bounds=(-30.2, 36.8),
            metadata=dict(init_time=time.time(), notes="Subject for BMM Monarch."),
            sample_number=9,
            direct_to_queue=True,
        )

        agent = BMMSVAMonarch(
            subjects=[subject_agent],
            direct_subjects_on_tell=True,
            device="cuda:1",
            origin=(183.818, 122.319),
            relative_bounds=(-28, 37),
            metadata=dict(
                init_time=time.time(),
                notes="BMM Monarch using spectral distance scientific value agent. "
                "Doubles all measurements on PDF.",
            ),
            ask_on_tell=True,
            direct_to_queue=True,
            report_on_tell=True,
        )
        signal.signal(signal.SIGINT, agent.signal_handler)
        agent.start(ask_at_start=False)
    except Exception as e:
        agent.stop(exit_status="fail", reason=f"{e}")
        raise e
