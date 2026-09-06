"""Named execution schedules; neither changes original model bytes or quality gates."""
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceSchedule:
    name:str
    window_batch:int
    query_chunk:int
    compact_layout:bool
    allocator_budget_bytes:int|None=None


def select_schedule(name):
    if name=='baseline12':return InferenceSchedule(name,12,32,False)
    if name=='native_opt2':return InferenceSchedule(name,384,128,True)
    if name=='native_opt3':return InferenceSchedule(name,768,1024,True,5000000000)
    raise ValueError('unknown optimization profile; no silent fallback')
