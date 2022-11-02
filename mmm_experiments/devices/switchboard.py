from ophyd import Device, Component as Cpt, EpicsSignal, EpicsSignalRO
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

        st = SubscriptionStatus(self, make_cb())
        # questionable choice to override the base
        self._status = st & base_st
        return self._status


class SwitchBoard(Device):
    publish = Cpt(LatchSignal, "publish_to_queue", put_complete=True)
    adjudicate_mode = Cpt(EpicsSignal, "adjudicate_mode", put_complete=True)
    adjudicate_status = Cpt(EpicsSignalRO, "adjudicate_status")


sb = SwitchBoard("pdf:switchboard:", name="sb")
