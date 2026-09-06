"""Opt-in whole-grid scheduling policy for recovered window families.

This changes submission granularity only.  It deliberately does not change
the window coordinate system, padding, weights, precision or model math.
"""
from contextlib import contextmanager
from contextvars import ContextVar


FAMILIES = frozenset(('pre', 'c32', 'c64', 'c128', 'c256', 'c512', 'vit', 'head'))
_active = ContextVar('native_grid_policy', default=frozenset())


def validate_families(families):
    value = frozenset(families)
    unknown = value - FAMILIES
    if unknown:
        raise ValueError(f'unreviewed whole-grid families: {sorted(unknown)}')
    return value


def whole_grid_enabled(family):
    if family not in FAMILIES:
        raise ValueError(f'unreviewed whole-grid family: {family}')
    return family in _active.get()


def run_windows(module, windows, window_batch, family):
    """Run one full-grid module call or preserve the reviewed batch loop."""
    import torch
    if type(window_batch) is not int or window_batch <= 0:
        raise ValueError('positive integer window batch required')
    if whole_grid_enabled(family):
        return module(windows)
    return torch.cat([module(part) for part in windows.split(window_batch)])


@contextmanager
def grid_policy(families=()):
    value = validate_families(families)
    token = _active.set(value)
    try:
        yield value
    finally:
        _active.reset(token)
