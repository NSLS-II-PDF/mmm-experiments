# Borrowed from https://github.com/NSLS-II-BMM/profile_collection/blob/master/startup/BMM/larch_interface.py
import larch.utils.show as lus
import numpy
from larch import Group, Interpreter
from larch.xafs import autobk, find_e0, pre_edge, xftf

LARCH = Interpreter()


class Pandrosus:
    """A thin wrapper around basic XAS data processing for individual
    data sets as implemented in Larch.

    The plotting capabilities of this class are very similar to the
    orange plot buttons in Athena.

    Attributes
    ----------
    uid : str
        Databroker unique ID of the data set, used to fetch mu(E) from the database
    name : str
        Human-readable name of the data set
    group : str
        Larch group containing the data
    pre : dict
        Dictionary of pre-edge and normalization arguments
    bkg : dict
        Dictionary of background subtraction arguments
    fft : dict   Dictionary of forward Fourier transform arguments
    bft : dict
        Dictionary of backward Fourier transform arguments
    rmax : float
        upper bound of R-space plot

    See http://xraypy.github.io/xraylarch/xafs/preedge.html and
    http://xraypy.github.io/xraylarch/xafs/autobk.html for details
    about these parameters.

    Methods
    ------
    fetch:
        Get data set and prepare for analysis with Larch
    make_xmu:
        method that actually constructs mu(E) from the data source
    put:
        make a Pandrosus object from ndarrays for energy and mu
    show:
        wrapper around Larch's show command, examine the content of the Larch group
    prep:
        normalize, background subtract, and forward transform the data
    do_xftf:
        perform the forward (k->R) transform
    do_xftr:
        perform the reverse (R->q) transform
    """

    def __init__(self, uid=None, name=None):
        self.uid = uid
        self.name = name
        self.group = None
        self.title = ""
        # Larch parameters
        self.pre = {
            "e0": None,
            "pre1": None,
            "pre2": None,
            "norm1": None,
            "norm2": None,
            "nnorm": None,
            "nvict": 0,
        }
        self.bkg = {
            "rbkg": 1,
            "e0": None,
            "kmin": 0,
            "kmax": None,
            "kweight": 2,
        }
        self.fft = {
            "window": "Hanning",
            "kmin": 3,
            "kmax": 12,
            "dk": 2,
        }
        self.bft = {
            "window": "Hanning",
            "rmin": 1,
            "rmax": 3,
            "dr": 0.1,
        }
        # plotting parameters
        self.xe = "energy (eV)"
        self.xk = "wavenumber ($\AA^{-1}$)"  # noqa
        self.xr = "radial distance ($\AA$)"  # noqa
        self.rmax = 6

        # flow control parameters

    def make_xmu(self, run, mode):
        """Load energy and mu(E) arrays into Larch and into this wrapper object.

        ***************************************************************
        This should be the only part of this startup script that needs
        beamline-specific configuration.  What is shown below is
        specific to how data are retrieved from Databroker at
        BMM. Other beamlines -- or reading data from files -- will
        need something different.
        ***************************************************************

        Parameters
        ----------
        run : databroker.client.BlueskyRun
            database identifier (assuming you are using databroker)
        mode : str
            'transmission', 'fluorescence', or 'reference'

        """
        table = run.primary.data
        self.group.energy = numpy.array(table["dcm_energy"])
        self.group.i0 = numpy.array(table["I0"])
        if mode == "flourescence":
            mode = "fluorescence"
        if mode == "reference":
            self.group.mu = numpy.array(numpy.log(table["It"] / table["Ir"]))
            self.group.i0 = numpy.array(table["It"])
            self.group.signal = numpy.array(table["Ir"])

        #######################################################################################
        # CAUTION!!  This only works when BMMuser is correctly set.  This is unlikely to work #
        # on data in past history.  See new '_dtc' element of start document.  9 Sep 2020     #
        #######################################################################################
        elif any(md in mode for md in ("fluo", "flou", "both")):
            columns = run.start["XDI"]["_dtc"]
            self.group.mu = numpy.array(
                (table[columns[0]] + table[columns[1]] + table[columns[2]] + table[columns[3]]) / table["I0"]
            )
            self.group.i0 = numpy.array(table["I0"])
            self.group.signal = numpy.array(
                table[columns[0]] + table[columns[1]] + table[columns[2]] + table[columns[3]]
            )

        elif mode == "xs1":
            columns = run.start["XDI"]["_dtc"]
            self.group.mu = numpy.array(table[columns[0]] / table["I0"])
            self.group.i0 = numpy.array(table["I0"])
            self.group.signal = numpy.array(table[columns[0]])

        elif mode == "xs":
            columns = run.start["XDI"]["_dtc"]
            self.group.mu = numpy.array(
                (table[columns[0]] + table[columns[1]] + table[columns[2]] + table[columns[3]]) / table["I0"]
            )
            self.group.i0 = numpy.array(table["I0"])
            self.group.signal = numpy.array(
                table[columns[0]] + table[columns[1]] + table[columns[2]] + table[columns[3]]
            )

        elif mode == "ref":
            self.group.mu = numpy.array(numpy.log(table["It"] / table["Ir"]))
            self.group.i0 = numpy.array(table["It"])
            self.group.signal = numpy.array(table["Ir"])

        elif mode == "yield":
            self.group.mu = numpy.array(table["Iy"] / table["I0"])
            self.group.i0 = numpy.array(table["I0"])
            self.group.signal = numpy.array(table["Iy"])

        else:
            self.group.mu = numpy.array(numpy.log(table["I0"] / table["It"]))
            self.group.i0 = numpy.array(table["I0"])
            self.group.signal = numpy.array(table["It"])

    def fetch(self, run, name=None, mode="transmission"):
        self.uid = run.start["uid"]
        if name is not None:
            self.name = name
        else:
            self.name = run.start["uid"][-6:]
        self.group = Group(__name__=self.name)
        self.title = run.metadata["start"]["XDI"]["Sample"]["name"]
        self.make_xmu(run, mode=mode)
        self.prep()

    def put(self, energy, mu, name):
        self.name = name
        self.group = Group(__name__=self.name)
        self.group.energy = energy
        self.group.mu = mu
        self.prep()

    def prep(self):
        # the next several lines seem necessary because the version
        # of Larch currently at BMM is not correctly resolving
        # pre1=pre2=None or norm1=norm2=None.  The following
        # approximates Larch's defaults
        if self.pre["e0"] is None:
            find_e0(self.group.energy, mu=self.group.mu, group=self.group, _larch=LARCH)
            ezero = self.group.e0
        else:
            ezero = self.pre["e0"]
        if self.pre["norm2"] is None:
            self.pre["norm2"] = self.group.energy.max() - ezero
        if self.pre["norm1"] is None:
            self.pre["norm1"] = self.pre["norm2"] / 5
        if self.pre["pre1"] is None:
            self.pre["pre1"] = self.group.energy.min() - ezero
        if self.pre["pre2"] is None:
            self.pre["pre2"] = self.pre["pre1"] / 3
        pre_edge(
            self.group.energy,
            mu=self.group.mu,
            group=self.group,
            e0=ezero,
            step=None,
            pre1=self.pre["pre1"],
            pre2=self.pre["pre2"],
            norm1=self.pre["norm1"],
            norm2=self.pre["norm2"],
            nnorm=self.pre["nnorm"],
            nvict=self.pre["nvict"],
            _larch=LARCH,
        )
        autobk(
            self.group.energy,
            mu=self.group.mu,
            group=self.group,
            rbkg=self.bkg["rbkg"],
            e0=self.bkg["e0"],
            kmin=self.bkg["kmin"],
            kmax=self.bkg["kmax"],
            kweight=self.bkg["kweight"],
            _larch=LARCH,
        )
        xftf(
            self.group.k,
            chi=self.group.chi,
            group=self.group,
            window=self.fft["window"],
            kmin=self.fft["kmin"],
            kmax=self.fft["kmax"],
            dk=self.fft["dk"],
            _larch=LARCH,
        )

    def show(self, which=None):
        if which is None:
            lus.show(self.group, _larch=LARCH)
        elif "pre" in which:
            lus.show(self.group.pre_edge_details, _larch=LARCH)
        elif which == "autobk":
            lus.show(self.group.autobk_details, _larch=LARCH)
        elif which == "fft" or which == "xftf":
            lus.show(self.group.xftf_details, _larch=LARCH)
        elif which == "bft" or which == "xftr":
            lus.show(self.group.xftr_details, _larch=LARCH)
        else:
            lus.show(self.group, _larch=LARCH)
