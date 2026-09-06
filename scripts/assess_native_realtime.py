"""Fail-closed assessment of the user-selected 3840x2160/60 FPS milestone.

Consumes normalized GAME evidence, not subgraph capability reports. This is an
assessment tool, not a capture tool: it cannot create missing visual evidence.
NVIDIA-matching quality is explicitly deferred; native full-frame, live output,
SDR/HDR, stability, synchronization and actual performance are NOT deferred.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

WIDTH, HEIGHT, FPS, MIN_FRAMES = 3840, 2160, 60, 1000


def assess(report):
    reasons, metrics = [], {}
    def require(ok, reason):
        if not ok:
            reasons.append(reason)
    require(report.get('kind') == 'native_full_frame_game_run', 'not a complete native game run')
    require(report.get('device') == 'AMD Radeon RX 9070 XT', 'target GPU not evidenced')
    require(report.get('native_graph_complete') is True, 'native graph incomplete')
    for key in ('cpu_neural_fallback', 'cpu_image_transfer_on_frame_path', 'nvidia_runtime_dependency',
                'independent_image_tiles', 'silent_input_downscale', 'device_fault_observed'):
        require(report.get(key) is False, f'{key} must be explicitly false')
    require(report.get('pre_hud_output_visible') is True, 'pre-HUD visible neural output unverified')
    require(report.get('resource_generation_and_fence_validation') is True, 'resource/fence identity unverified')
    require(report.get('dynamic_size_reset_validation') is True, 'size/scene reset handling unverified')
    require(report.get('game_settings_restored') is True, 'game settings restoration unverified')
    for mode in ('sdr', 'hdr'):
        session = report.get('sessions', {}).get(mode, {})
        require(session.get('resolution') == [WIDTH, HEIGHT], f'{mode}: wrong resolution')
        require(session.get('visual_review') == 'PASS_NO_BLACK_FRAMES_SEAMS_FLICKER_OR_HUD_DAMAGE',
                f'{mode}: stable image/HUD review missing')
        require(bool(session.get('visual_evidence_paths')), f'{mode}: missing visual evidence links')
        frames = session.get('frames', [])
        require(len(frames) >= MIN_FRAMES, f'{mode}: fewer than {MIN_FRAMES} frames')
        if len(frames) < 2:
            continue
        frame_ids, times, inference = [], [], []
        valid = True
        for frame in frames:
            frame_ids.append(frame.get('frame_id'))
            require(frame.get('network_applied') is True and frame.get('fallback_used') is False,
                    f'{mode}: frame without applied network output')
            require(frame.get('source_frame_id') == frame.get('frame_id') and frame.get('frame_id') is not None,
                    f'{mode}: stale or unidentified output')
            for key, values in [('present_ms', times), ('inference_complete_host_ms', inference)]:
                value = frame.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    valid = False
                else:
                    values.append(value)
        require(all(type(i) is int for i in frame_ids) and len(set(frame_ids)) == len(frame_ids),
                f'{mode}: missing/duplicate frame identities')
        require(valid, f'{mode}: invalid timing evidence')
        if not valid:
            continue
        intervals = [b-a for a, b in zip(times, times[1:])]
        require(all(v > 0 for v in intervals), f'{mode}: nonmonotonic presentation times')
        if not all(v > 0 for v in intervals):
            continue
        actual_fps = 1000*len(intervals)/sum(intervals)
        p95 = sorted(intervals)[math.ceil(.95*len(intervals))-1]
        metrics[mode] = {'applied_frames': len(frames), 'actual_fps': actual_fps,
                         'present_interval_p95_ms': p95,
                         'inference_host_median_ms': statistics.median(inference),
                         'inference_host_max_ms': max(inference)}
        require(actual_fps >= FPS, f'{mode}: measured FPS below {FPS}')
        require(p95 <= 1000/FPS, f'{mode}: p95 frame interval exceeds 60 FPS budget')
    return {'status': 'REALTIME_GAME_MILESTONE_PASS' if not reasons else 'NOT_ACCEPTED',
            'pass': not reasons, 'target': {'width': WIDTH, 'height': HEIGHT, 'fps': FPS},
            'reasons': list(dict.fromkeys(reasons)), 'metrics': metrics,
            'nvidia_matching_quality_required': False, 'dlss5_quality_equivalence_claimed': False,
            'note': 'Requires genuine trace and visual-review artifacts; analyzer output alone is not observational proof.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assess(json.loads(args.input.read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['pass'] else 1)
