"""Opt-in scale-boundary layout policy.

Resident mode changes storage/layout routing only. It does not change pooling,
projection, quantization, window coordinates, weights, or arithmetic precision.
Debug/capture outviews remain explicit and are never generated implicitly by
the resident hot path.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionPolicy:
    resident: bool = False
    capture_outviews: bool = False


_active = ContextVar("native_transition_policy", default=TransitionPolicy())


def active_transition_policy():
    return _active.get()


@contextmanager
def transition_policy(*, resident=False, capture_outviews=False):
    if type(resident) is not bool or type(capture_outviews) is not bool:
        raise ValueError("transition policy flags must be explicit booleans")
    if capture_outviews and not resident:
        raise ValueError("capture_outviews only describes an opt-in resident run")
    value = TransitionPolicy(resident=resident, capture_outviews=capture_outviews)
    token = _active.set(value)
    try:
        yield value
    finally:
        _active.reset(token)
