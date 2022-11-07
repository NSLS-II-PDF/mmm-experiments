#!/usr/bin/env python3
from textwrap import dedent

from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run


class PDFMMMSwitchBoard(PVGroup):
    """
    An IOC with three uncoupled read/writable PVs.

    Scalar PVs
    ----------
    A (int)
    B (float)

    Array PVs
    ---------
    C (array of int)
    """

    publish_to_queue = pvproperty(
        value=1,
        dtype=int,
        name="Pub-CMD",
        doc="""A flag to be used to tell the adjudicator to do it's job.

Set high to start the work and the adjudicator will set it back to
0
""",
    )
    adjudicate_mode = pvproperty(
        value="off",
        report_as_string=True,
        name="Adj-Mode",
        doc="""A string PV to control the behavior of the adjudicator.

This may turn into an enum in the future, but for now use this PV to pass
strings through the IOC -> the adjudicator process.
""",
    )
    adjudicate_status = pvproperty(
        value="",
        report_as_string=True,
        name="Adj-STS",
        doc="A string PV for the adjudicator to publish its feelings.",
    )


if __name__ == "__main__":
    ioc_options, run_options = ioc_arg_parser(
        default_prefix="{xf}{{SB:{inst_num}}}",
        desc=dedent(PDFMMMSwitchBoard.__doc__ or ""),
        macros={"xf": None, "inst_num": "1"},
    )
    ioc = PDFMMMSwitchBoard(**ioc_options)
    run(ioc.pvdb, **run_options)
