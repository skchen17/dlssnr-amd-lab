"""Bind one color image to both preblock and postblock in a standalone replay plan."""
import argparse
import json
from pathlib import Path

try:
    from scripts.run_full_graph_integrated import load_color_input, sha256
except ModuleNotFoundError:
    from run_full_graph_integrated import load_color_input, sha256


def make_plan(base: Path, color: Path, post_ptx: Path, post_audit: Path, output: Path) -> dict:
    plan = json.loads(base.read_text(encoding='utf-8-sig'))
    if [s['slot'] for s in plan['slots']] != list(range(156)) or plan.get('diagnostic_injections'):
        raise ValueError('requires contiguous injection-free graph')
    payload, color_info = load_color_input(color)
    ptx = post_ptx.read_bytes()
    audit = json.loads(post_audit.read_text(encoding='utf-8-sig'))
    if audit.get('status') != 'PASS' or audit['output_sha256'].upper() != sha256(ptx) or b'tex.2d.' not in ptx:
        raise ValueError('postblock texture-preserving PTX provenance mismatch')

    def rebase(value):
        if isinstance(value, dict):
            return {k: str((base.parent / v).resolve()) if k in ('path', 'ptx') and isinstance(v, str)
                    else rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        return value

    plan = rebase(plan)
    prior = {k: plan['slots'][154][k] for k in ('ptx', 'ptx_sha256', 'special')}
    plan['slots'][154].update(ptx=str(post_ptx.resolve()), ptx_sha256=sha256(ptx), special='post_linear_surface_texture')
    plan['post_texture'] = {**color_info, 'path': str(color.resolve()),
                            'sampler': 'point_border_normalized_coordinates'}
    plan['color_input'] = color_info
    plan['experiment'] = 'real_color_single_frame_diagnostic'
    plan['color_plan_provenance'] = {'base': str(base.resolve()), 'base_sha256': sha256(base.read_bytes()),
        'replaced_postblock': prior, 'postblock_audit': str(post_audit.resolve()),
        'zero_reference_applicable': False, 'temporal_inputs_bound': False,
        'full_frame_dynamic_resolution_verified': False}
    checked = {}
    for slot in plan['slots']:
        path = Path(slot['ptx'])
        if path not in checked:
            checked[path] = sha256(path.read_bytes())
        if checked[path] != slot['ptx_sha256'].upper():
            raise ValueError(f"slot {slot['slot']} PTX provenance mismatch")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    return {'status': 'PLAN_PREPARED', 'modules_verified': len(checked), 'input': color_info,
            'postblock_samples_preserved': True, 'output': str(output.resolve())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'color', 'post-ptx', 'post-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_plan(args.base, args.color, args.post_ptx, args.post_audit, args.output), indent=2))
