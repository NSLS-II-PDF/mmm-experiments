from bluesky import plan_stubs as bps


def agent_driven_nap(delay: float, *, delay_kwarg: float = 0):
    """Ensuring we can auto add 'agent_' plans and use args/kwargs"""
    if delay_kwarg:
        yield from bps.sleep(delay_kwarg)
    else:
        yield from bps.sleep(delay)
