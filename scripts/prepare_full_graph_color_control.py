"""Create a zero-neural-color control while preserving the postblock base image."""
import argparse
import json
from pathlib import Path

try:
    from scripts.run_full_graph_integrated import load_color_input, sha256
except ModuleNotFoundError:
    from run_full_graph_integrated import load_color_input, sha256


def prepare(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    if not plan.get('color_input', {}).get('external') or not plan.get('post_texture'):
        raise ValueError('requires a real-color plan with a separately bound post texture')
    if plan.get('diagnostic_injections'):
        raise ValueError('reference injections not allowed')
    # Keep all assets valid when the control lives in a new directory.
    def rebase(value):
        if isinstance(value, dict):
            return {k: str((plan_path.parent / v).resolve()) if k in ('path', 'ptx') and isinstance(v, str)
                    else rebase(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        return value
    plan = rebase(plan)
    original = plan['color_input']
    output.mkdir(parents=True, exist_ok=False)
    zero = output / 'zero_rgba16f.raw'
    zero.write_bytes(bytes(640 * 360 * 8))
    _, plan['color_input'] = load_color_input(zero)
    plan['experiment'] = 'zero_preblock_color_fixed_real_post_control'
    plan['control_provenance'] = {
        'parent_plan': str(plan_path.resolve()), 'parent_sha256': sha256(plan_path.read_bytes()),
        'original_neural_color': original,
        'only_plan_binding_changed': 'slot1_color_input', 'post_texture_unchanged': True,
        'rtx_quality_reference': False}
    target = output / 'plan.json'
    target.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    return {'status': 'CONTROL_PREPARED', 'plan': str(target.resolve())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.output), indent=2))
