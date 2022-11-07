from ophyd import Component as Cpt
from ophyd import Device, EpicsSignal, EpicsSignalRO
from ophyd.status import SubscriptionStatus


class LatchSignal(EpicsSignal):
    def set(self, value, *, timeout=None, settle_time=None, **kwargs):
        base_st = super().set(value, timeout=timeout, settle_time=settle_time, **kwargs)
        # going low, just be done
        if value == 0:
            return base_st

        # going high, set up a latch
        def make_cb():
            went_high = False

            def cb(*, old_value, value, **kwargs):
                nonlocal went_high
                if value != 0:
                    went_high = True
                if value == 0 and went_high:
                    return True
                return False

            return cb

        my_cb = make_cb()

        st = SubscriptionStatus(self, my_cb, timeout=timeout)
        # questionable choice to override the base
        self._status = st & base_st
        return self._status


class SwitchBoardClient(Device):
    # this has tricking "latching" behavior to wait for other party to work
    publish = Cpt(LatchSignal, "Pub-CMD", put_complete=True, timeout=15)
    adjudicate_mode = Cpt(EpicsSignal, "Adj-Mode", put_complete=True)
    # client can not set status
    adjudicate_status = Cpt(EpicsSignalRO, "Adj-STS")


class SwitchBoardBackend(Device):
    # do not use latching behavior on backend
    publish_to_queue = Cpt(EpicsSignal, "Pub-CMD", put_complete=True)
    # backend can not set its own mode
    adjudicate_mode = Cpt(EpicsSignalRO, "Adj-Mode")
    adjudicate_status = Cpt(EpicsSignal, "Adj-STS", put_complete=True)


if __name__ == "__main__":
    sb = SwitchBoardClient("XF:28ID1-DA{SB:1}", name="sb")
