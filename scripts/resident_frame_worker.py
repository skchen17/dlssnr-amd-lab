"""Local-only, single-client shared-memory worker for bounded game bring-up.

Full surfaces are covered by independent 640x360 tiles, with edge replication
and NO resizing. This is not equivalent to dynamic full-frame attention.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import mmap
import struct
import time
import uuid
from pathlib import Path

import numpy as np

try:
    from .resident_full_graph import ResidentGraph, WIDTH, HEIGHT, replay
except ImportError:
    from resident_full_graph import ResidentGraph, WIDTH, HEIGHT, replay

MAGIC = b'NRLIVE01'
HEADER = struct.Struct('<8sQQIIIIII')
HEADER_BYTES = 256
MAX_BYTES = 3840 * 2160 * 8
MAPPING_BYTES = HEADER_BYTES + MAX_BYTES * 2


def tile_frame(graph, raw: bytes, width: int, height: int, *, budget_seconds=20):
    if not 1 <= width <= 8192 or not 1 <= height <= 8192 or len(raw) != width * height * 8 or len(raw) > MAX_BYTES:
        raise ValueError('frame geometry/byte budget')
    source = np.frombuffer(raw, dtype='<f2').reshape(height, width, 4)
    if not np.isfinite(source).all():
        raise ValueError('nonfinite source')
    result = np.empty_like(source)
    tiles = []
    started = time.perf_counter()
    for y in range(0, height, HEIGHT):
        for x in range(0, width, WIDTH):
            h, w = min(HEIGHT, height - y), min(WIDTH, width - x)
            tile = np.pad(source[y:y+h, x:x+w], ((0, HEIGHT-h), (0, WIDTH-w), (0, 0)), mode='edge')
            remaining = budget_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError('full-surface inference deadline')
            output, info = graph.infer(tile.tobytes(), timeout_seconds=min(10.0, remaining))
            image = np.frombuffer(output, dtype='<f2').reshape(HEIGHT, WIDTH, 4)
            result[y:y+h, x:x+w] = image[:h, :w]
            tiles.append({'xywh': [x, y, w, h], **info})
    # Alpha is a game transport contract, not a learned color component.
    result[:, :, 3] = source[:, :, 3]
    if time.perf_counter() - started > budget_seconds:
        raise TimeoutError('full-surface result expired')
    raw_output = result.tobytes()
    return raw_output, {'width': width, 'height': height, 'tiles': tiles,
        'input_sha256': replay.sha256(raw), 'output_sha256': replay.sha256(raw_output),
        'host_surface_ms': (time.perf_counter() - started) * 1000,
        'whole_surface_covered': True, 'resized': False, 'alpha_preserved': True,
        'independent_fixed_shape_tiles': True, 'seam_quality_verified': False,
        'equivalent_full_frame_attention': False, 'temporal_inputs_bound': False}


class WinEvents:
    def __init__(self):
        self.k = ct.WinDLL('kernel32', use_last_error=True)
        self.k.CreateEventW.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_wchar_p]
        self.k.CreateEventW.restype = ct.c_void_p
        self.k.SetEvent.argtypes = [ct.c_void_p]
        self.k.SetEvent.restype = ct.c_int
        self.k.WaitForSingleObject.argtypes = [ct.c_void_p, ct.c_uint32]
        self.k.WaitForSingleObject.restype = ct.c_uint32
        self.k.CloseHandle.argtypes = [ct.c_void_p]
        self.k.CloseHandle.restype = ct.c_int
        self.handles = []

    def create(self, name):
        handle = self.k.CreateEventW(None, False, False, name)
        if not handle:
            raise ct.WinError(ct.get_last_error())
        self.handles.append(handle)
        if ct.get_last_error() == 183:
            raise RuntimeError('IPC event name already exists')
        return handle

    def close(self):
        for h in self.handles:
            self.k.CloseHandle(h)


def validate_request(header, previous):
    magic, request, response, width, height, size, status, client, version = header
    if magic != MAGIC or version != 1 or request != previous + 1 or response != previous or status != 2 or not client:
        raise ValueError('invalid/stale IPC request identity')
    if not 1 <= width <= 8192 or not 1 <= height <= 8192 or size != width * height * 8 or size > MAX_BYTES:
        raise ValueError('IPC request dimensions')
    return request, width, height, size, client


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--nvcuda', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-frames', type=int, default=12)
    p.add_argument('--idle-seconds', type=int, default=600)
    args = p.parse_args()
    if not 1 <= args.max_frames <= 1000 or not 1 <= args.idle_seconds <= 3600:
        p.error('bounded frames/idle timeout required')
    args.output.mkdir(parents=True, exist_ok=False)
    graph = None
    events = WinEvents()
    shared = None
    report = {'status': 'STARTING', 'frames': [], 'game_session_verified': False}
    path = args.output / 'worker.json'
    try:
        graph = ResidentGraph(args.plan.resolve(), args.nvcuda.resolve(),
                              progress=lambda item: print(json.dumps(item), flush=True))
        # Warm before exposing ready: no compilation or immutable-weight IO in game callbacks.
        graph.infer(bytes(WIDTH * HEIGHT * 8))
        name = 'Local\\DLSSNRLive_' + uuid.uuid4().hex
        request_event = events.create(name + '_request')
        response_event = events.create(name + '_response')
        shared = mmap.mmap(-1, MAPPING_BYTES, tagname=name)
        shared[:HEADER.size] = HEADER.pack(MAGIC, 0, 0, 0, 0, 0, 1, 0, 1)
        # Tiny ASCII name file consumed by the separately validated native bridge.
        (args.output / 'worker_name.txt').write_text(name + '\n', encoding='ascii')
        report.update(status='READY', runtime=graph.info, ipc_name=name)
        replay.write_progress(path, report)
        previous, owner = 0, None
        for _ in range(args.max_frames):
            status = events.k.WaitForSingleObject(request_event, args.idle_seconds * 1000)
            if status == 258:
                report['status'] = 'IDLE_TIMEOUT'
                break
            if status != 0:
                raise RuntimeError('request event failed')
            header = HEADER.unpack(shared[:HEADER.size])
            seq, w, h, size, client = validate_request(header, previous)
            if owner is not None and owner != client:
                raise ValueError('IPC client PID changed')
            owner = client
            data = bytes(shared[HEADER_BYTES:HEADER_BYTES + size])
            output, info = tile_frame(graph, data, w, h)
            # Client owns input while request is in flight; never publish if mutated.
            if HEADER.unpack(shared[:HEADER.size]) != header:
                raise ValueError('IPC request changed during inference')
            shared[HEADER_BYTES + MAX_BYTES:HEADER_BYTES + MAX_BYTES + size] = output
            shared[:HEADER.size] = HEADER.pack(MAGIC, seq, seq, w, h, size, 1, client, 1)
            if not events.k.SetEvent(response_event):
                raise ct.WinError(ct.get_last_error())
            previous = seq
            info.update(request_id=seq, client_pid=client)
            report['frames'].append(info)
            report['status'] = 'SERVING'
            # Only bounded audit outputs; no per-slot checkpoint files.
            if seq <= 2:
                (args.output / f'input{seq}.raw').write_bytes(data)
                (args.output / f'output{seq}.raw').write_bytes(output)
            replay.write_progress(path, report)
            print(json.dumps({'request': seq, 'width': w, 'height': h,
                              'host_ms': info['host_surface_ms']}), flush=True)
        else:
            report['status'] = 'FRAME_LIMIT'
    except BaseException as exc:
        report.update(status='FAIL', failure=str(exc))
        if shared is not None:
            struct.pack_into('<I', shared, 36, 3)
            events.k.SetEvent(response_event)
        raise
    finally:
        replay.write_progress(path, report)
        if shared is not None:
            shared.close()
        events.close()
        if graph:
            graph.close()


if __name__ == '__main__':
    main()
