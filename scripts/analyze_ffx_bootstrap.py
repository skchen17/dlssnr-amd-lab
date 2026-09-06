"""Check external bootstrap fixture, forwarding, and scoped failure cleanup."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def analyze(root):
    positive = root / 'positive with spaces'
    boot, host = read(positive / 'bootstrap.json'), read(positive / 'host.json')
    negative, checks = read(root / 'negative' / 'bootstrap.json'), read(root / 'checks.json')
    events = [json.loads(line) for line in (positive / 'events.jsonl').read_text().splitlines() if line]
    gates = dict(
        bootstrap=boot.get('status') == 'BOOTSTRAP_PASS' and boot.get('entry_instruction_restored') is True and boot.get('child_exit_code') == 0,
        header_only=boot.get('observer_profile') == 0 and all(e.get('profile') == 0 and not e.get('body_decoded', False) for e in events),
        original_forwarding=host.get('status') == 'HOST_PASS' and host.get('original_calls') == 5 and host.get('last_error') == 0xFACE,
        concurrent_log_read=host.get('concurrent_log_read') is True,
        five_imports=[e.get('event') for e in events] == ['ffxCreateContext', 'ffxQuery', 'ffxConfigure', 'ffxDispatch', 'ffxDestroyContext'],
        original_statuses=[e.get('return_code') for e in events] == [0, 31, 23, 37, 29],
        existing_result_preserved=checks.get('existing_result_preserved') is True,
        failed_child_cleanup=negative.get('status') == 'BOOTSTRAP_FAIL' and checks.get('negative_child_exited') is True,
        image_unchanged=bool(checks.get('fixture_sha256_before')) and checks.get('fixture_sha256_before') == checks.get('fixture_sha256_after'),
        no_game=boot.get('game_launched') is False and boot.get('texture_capture_enabled') is False,
    )
    return dict(status='PASS' if all(gates.values()) else 'FAIL', checks=gates,
                game_runtime_ready=False, texture_capture_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
