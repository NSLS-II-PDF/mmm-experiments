import bluesky.preprocessors as bpp
import msgpack
import redis
from bluesky import plan_stubs as bps
from confluent_kafka import Producer
from nslsii import _read_bluesky_kafka_config_file


@bpp.run_decorator()
def agent_driven_nap(delay: float, *, delay_kwarg: float = 0, md=None):
    """Ensuring we can auto add 'agent_' plans and use args/kwargs"""
    if delay_kwarg:
        yield from bps.sleep(delay_kwarg)
    else:
        yield from bps.sleep(delay)


def agent_directive(tla, name, doc):
    """
    Issue any directive to a listening agent by name/uid.

    Parameters
    ----------
    tla : str
        Beamline three letter acronym
    name : str
        Unique agent name. These are generated using xkcdpass for names like:
        "agent-exotic-farm"
        "xca-clever-table"
    doc : dict
        This is the message to pass to the agent. It must take the form:
        {"action": method_name,
         "args": (arguments,),
         "kwargs: {keyword:argument}
        }

    Returns
    -------

    """
    kafka_config = _read_bluesky_kafka_config_file("/etc/bluesky/kafka.yml")
    producer_config = dict()
    producer_config.update(kafka_config["runengine_producer_config"])
    producer_config["bootstrap.servers"] = ",".join(kafka_config["bootstrap_servers"])
    agent_producer = Producer(producer_config)

    # All 3 steps should happen for each message publication
    agent_producer.produce(topic=f"{tla}.mmm.bluesky.agents", key="", value=msgpack.dumps((name, doc)))
    agent_producer.poll(0)
    agent_producer.flush()
    yield from bps.null()


# ================================== PDF SPECIFIC PLANS =================================== #
# ==================================  99-demo_plans.py  =================================== #
def simple_ct(*args, **kwargs):
    ...


pe1c = ...
Grid_X = ...
Grid_Y = ...
Grid_Z = ...
Det_1_X = ...
Det_1_Y = ...
Det_1_Z = ...
ring_current = ...
BStop1 = ...


def agent_sample_count(motor, position: float, exposure: float, *, sample_number: int, md=None):
    yield from bps.mv(motor, position)
    _md = dict(
        Grid_X=Grid_X.read(),
        Grid_Y=Grid_Y.read(),
        Grid_Z=Grid_Z.read(),
        Det_1_X=Det_1_X.read(),
        Det_1_Y=Det_1_Y.read(),
        Det_1_Z=Det_1_Z.read(),
        ring_current=ring_current.read(),
        BStop1=BStop1.read(),
    )
    _md.update(get_metadata_for_sample_number(bt, sample_number))
    _md.update(md or {})
    yield from simple_ct([pe1c], exposure, md=_md)


@bpp.run_decorator(md={})
def agent_driven_nap(delay: float, *, delay_kwarg: float = 0):
    """Ensuring we can auto add 'agent_' plans and use args/kwargs"""
    if delay_kwarg:
        yield from bps.sleep(delay_kwarg)
    else:
        yield from bps.sleep(delay)


def agent_print_glbl_val(key: str):
    """
    Get common global values from a namespace dictionary.
    Keys
    ---
    frame_acq_time : Frame rate acquisition time
    dk_window : dark window time
    """
    print(glbl[key])
    yield from bps.null()


def agent_set_glbl_val(key: str, val: float):
    """
    Set common global values from a namespace dictionary.
    Keys
    ---
    frame_acq_time : Frame rate acquisition time
    dk_window : dark window time
    """
    glbl[key] = val
    yield from bps.null()


# ================================== PDF SPECIFIC PLANS =================================== #
# ==================================  99-demo_plans.py  =================================== #


# ================================== BMM SPECIFIC PLANS =================================== #
# ================================== BMM/agent_plans.py =================================== #
def xafs(*args, filename, **kwargs):
    """Core plan in BMM startup"""
    ...


def change_edge(*args, **kwargs):
    """Change energy in BMM startup"""
    ...


xafs_det = ...
slits3 = ...


def agent_move_and_measure(
    motor_x,
    Cu_x_position,
    Ti_x_position,
    motor_y,
    Cu_y_position,
    Ti_y_position,
    *,
    Cu_det_position,
    Ti_det_position,
    md=None,
    **kwargs,
):
    """
    A complete XAFS measurement for the Cu/Ti sample.
    Each element edge must have it's own calibrated motor positioning and detector distance.
    The sample is moved into position, edge changed and spectra taken.

    Parameters
    ----------
    motor_x :
        Positional motor for sample in x.
    Cu_x_position : float
        Absolute motor position for Cu measurement (This is the real independent variable)
    Ti_x_position : float
        Absolute motor position for Ti measurement
    motor_y :
        Positional motor for sample in y.
    Cu_y_position : float
        Absolute motor position for Cu measurement
    Ti_y_position : float
        Absolute motor position for Ti measurement
    Cu_det_position : float
        Absolute motor position for the xafs detector for the Cu measurement.
    Ti_det_position : float
        Absolute motor position for the xafs detector for the Ti measurement.
    md : Optional[dict]
        Metadata
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

    def Cu_plan():
        yield from bps.mv(motor_x, Cu_x_position)
        _md = dict(Cu_position=motor_x.position)
        yield from bps.mv(motor_y, Cu_y_position)
        yield from bps.mv(xafs_det, Cu_det_position)
        _md["Cu_det_position"] = xafs_det.position
        _md.update(md or {})
        yield from bps.mv(slits3.vsize, 0.1)
        yield from change_edge("Cu", focus=True)
        # xafs doesn't take md, so stuff it into a comment string to be ast.literal_eval()
        yield from xafs(element="Cu", comment=str(_md), **kwargs)

    def Ti_plan():
        yield from bps.mv(motor_x, Ti_x_position)
        _md = dict(Ti_position=motor_x.position)
        yield from bps.mv(motor_y, Ti_y_position)
        yield from bps.mv(xafs_det, Ti_det_position)
        _md["Ti_det_position"] = xafs_det.position
        _md.update(md or {})
        yield from bps.mv(slits3.vsize, 0.3)
        yield from change_edge("Ti", focus=True)
        yield from xafs(element="Ti", comment=str(_md), **kwargs)

    rkvs = redis.Redis(host="xf06bm-ioc2", port=6379, db=0)
    element = rkvs.get("BMM:pds:element").decode("utf-8")
    # edge = rkvs.get('BMM:pds:edge').decode('utf-8')
    if element == "Ti":
        yield from Ti_plan()
        yield from Cu_plan()
    else:
        yield from Cu_plan()
        yield from Ti_plan()


def agent_move_motor(motor_x, Cu_x_position, *args, **kwargs):
    yield from bps.mv(motor_x, Cu_x_position)


def agent_change_edge(*args, **kwargs):
    yield from change_edge("Cu", focus=True)


def agent_xafs(
    motor_x,
    Cu_x_position,
    Ti_x_position,
    motor_y,
    Cu_y_position,
    Ti_y_position,
    *,
    Cu_det_position,
    Ti_det_position,
    md=None,
    **kwargs,
):
    _md = dict(Cu_position=motor_x.position)
    _md["Cu_det_position"] = xafs_det.position
    _md.update(md or {})
    yield from xafs(element="Cu", **kwargs)


# ================================== BBB SPECIFIC PLANS =================================== #
# ================================== BMM/agent_plans.py =================================== #
