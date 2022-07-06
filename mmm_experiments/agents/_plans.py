import bluesky.preprocessors as bpp
from bluesky import plan_stubs as bps


@bpp.run_decorator(md={})
def agent_driven_nap(delay: float, *, delay_kwarg: float = 0):
    """Ensuring we can auto add 'agent_' plans and use args/kwargs"""
    if delay_kwarg:
        yield from bps.sleep(delay_kwarg)
    else:
        yield from bps.sleep(delay)


# ================================== PDF SPECIFIC PLANS =================================== #
# ==================================  99-demo_plans.py  =================================== #

# ================================== PDF SPECIFIC PLANS =================================== #
# ==================================  99-demo_plans.py  =================================== #


# ================================== BMM SPECIFIC PLANS =================================== #
# ================================== 99-agent_plans.py  =================================== #
def xafs(*args, filename, **kwargs):
    """Core plan in BMM startup"""
    ...


def change_edge(*args, **kwargs):
    """Change energy in BMM startup"""
    ...


xafs_det = ...


def agent_move_and_measure(motor, Cu_position, Ti_position, *, Cu_det_position, Ti_det_position, **kwargs):
    """
    A complete XAFS measurement for the Cu/Ti sample.
    Each element edge must have it's own calibrated motor positioning and detector distance.
    The sample is moved into position, edge changed and spectra taken.

    Parameters
    ----------
    motor :
        Positional motor for sample
    Cu_position : float
        Absolute motor position for Cu measurement
    Ti_position : float
        Absolute motor position for Ti measurement
    Cu_det_position : float
        Absolute motor position for the xafs detector for the Cu measurement.
    Ti_det_position : float
        Absolute motor position for the xafs detector for the Ti measurement.
    kwargs :
        All keyword arguments for the xafs plan. Must include  'filename'. Eg below:
            >>> {'filename': 'Cu_PdCuCr_112421_001',
            >>> 'nscans': 1,
            >>> 'start': 'next',
            >>> 'mode': 'fluorescence',
            >>> 'element': 'Cu',
            >>> 'edge': 'K',
            >>> 'sample': 'PdCuCr',
            >>> 'preparation': 'film deposited on something',
            >>> 'comment': 'index = 1, position (x,y) = (-9.04, -31.64), center at (236.98807533, 80.98291381)',
            >>> 'bounds': '-200 -30 -10 25 12k',
            >>> 'steps': '10 2 0.3 0.05k',
            >>> 'times': '0.5 0.5 0.5 0.5'}
    """
    yield from bps.mv(motor, Cu_position)
    yield from bps.mv(xafs_det, Cu_det_position)
    yield from change_edge(["Cu"], focus=True)
    yield from xafs(element="Cu", **kwargs)
    yield from bps.mv(motor, Ti_position)
    yield from bps.mv(xafs_det, Ti_det_position)
    yield from change_edge(["Ti"], focus=True)
    yield from xafs(element="Ti", **kwargs)


# ================================== BBB SPECIFIC PLANS =================================== #
# ================================== 99-agent_plans.py  =================================== #
