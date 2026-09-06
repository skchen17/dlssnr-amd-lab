#!/usr/bin/env python3
"""Strictly lower every decoder entry used by graph slots 99-154."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# name, tag, init, bulk, wide, release, relaxed, shared, discard,
# e4m3, movmatrix, fp8 mma, f16 mma, vector red, release fence, surface, null texture
ENTRIES = [
    ("cc_dec_input_upsample_1024_512_tilesync_fp8", "dec_input", 2, 2, 24, 2, 1, 6, 0, 146, 0, 64, 0, 8, 0, 0, 0),
    ("cc_split_swin_16h_ffwd_512_chained_fp8", "split_ffwd", 2, 2, 4, 1, 2, 10, 0, 82, 0, 96, 0, 0, 0, 0, 0),
    ("cc_split_swin_16h_ffwd_proj_512_chained_fp8", "split_ffwd_proj", 3, 8, 8, 1, 2, 16, 0, 80, 0, 64, 0, 0, 0, 0, 0),
    ("cc_split_swin_16h_qkv_512_chained_fp8", "split_qkv", 2, 4, 4, 1, 2, 20, 0, 196, 32, 256, 0, 0, 0, 0, 0),
    ("cc_split_swin_16h_proj_512_chained_fp8", "split_proj", 3, 4, 4, 1, 3, 8, 0, 40, 0, 32, 0, 0, 0, 0, 0),
    ("cc_split_swin_16h_proj_512_outview_wait_fp8", "split_proj_outview", 3, 8, 0, 0, 3, 16, 0, 80, 0, 64, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_8h_256_8_upsample_tilesync_fp8", "swin8_upsample", 0, 0, 4, 1, 0, 68, 0, 356, 32, 292, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_8h_256_8_chained_fp8", "swin8_chained", 0, 0, 4, 1, 1, 68, 0, 324, 32, 288, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_8h_256_8_outview_wait_fp8", "swin8_outview", 0, 0, 0, 0, 1, 68, 0, 324, 32, 288, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_4h_128_4_upsample_tilesync_fp8", "swin4_upsample", 0, 0, 4, 1, 0, 64, 0, 356, 32, 276, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_4h_128_4_chained_fp8", "swin4_chained", 0, 0, 4, 1, 1, 64, 0, 324, 32, 272, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_4h_128_4_outview_wait_fp8", "swin4_outview", 0, 0, 0, 0, 1, 64, 0, 324, 32, 272, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_2h_64_2_upsample_tilesync_fp8", "swin2_upsample", 0, 0, 4, 1, 0, 56, 0, 356, 32, 276, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_2h_64_2_chained_fp8", "swin2_chained", 0, 0, 4, 1, 1, 28, 0, 372, 32, 304, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_2h_64_2_outview_wait_fp8", "swin2_outview", 0, 0, 0, 0, 1, 28, 0, 372, 32, 304, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8", "swin1_upsample", 0, 0, 4, 1, 0, 0, 0, 420, 32, 260, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_1h_32_1_chained_fp8", "swin1_chained", 0, 0, 4, 1, 1, 0, 0, 388, 32, 256, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_swin_1h_32_1_outview_wait_fp8", "swin1_outview", 0, 0, 0, 0, 1, 0, 0, 388, 32, 256, 0, 0, 0, 0, 0),
    ("cc_tinlayout_fused_post_block_swin_1h_32_fp8", "post_block", 0, 0, 0, 0, 0, 0, 6, 388, 32, 256, 16, 0, 0, 2, 14),
]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    for entry in ENTRIES:
        (name, tag, init, bulk, wide, release, relaxed, shared, discard,
         e4m3_count, mov_count, fp8_count, f16_count, vector_red, fence,
         surface_count, null_texture_count) = entry
        source = source_root / f"{name}.ptx"
        stages = [output / f"{tag}_{suffix}.ptx" for suffix in
                  ("mbarrier", "compat", "e4m3", "mov", "fp8", "f16",
                   "decoder_compat", "surface", "full")]
        reports = [output / f"{tag}_{suffix}.json" for suffix in
                   ("mbarrier", "compat", "e4m3", "mov", "fp8", "f16",
                    "decoder_compat", "surface", "null_texture")]
        waits = 2 if init else 0
        run([sys.executable, str(scripts / "lower_ptx_mbarrier.py"), str(source),
             str(stages[0]), str(reports[0]), "--expected-init", str(init),
             "--expected-bulk", str(bulk), "--expected-elect", str(bulk),
             "--expected-arrive", str(waits), "--expected-try-wait", str(waits)])
        run([sys.executable, str(scripts / "lower_ptx_zluda_compat.py"), str(stages[0]),
             str(stages[1]), str(reports[1]), "--expected-discard", str(discard),
             "--expected-wide-store", str(wide), "--expected-release-store", str(release),
             "--expected-relaxed-load", str(relaxed), "--expected-shared-cta-scope", str(shared)])
        run([sys.executable, str(scripts / "lower_ptx_e4m3.py"), str(stages[1]),
             str(stages[2]), str(reports[2]), "--expected-count", str(e4m3_count)])
        run([sys.executable, str(scripts / "lower_ptx_movmatrix.py"), str(stages[2]),
             str(stages[3]), str(reports[3]), "--expected-count", str(mov_count)])
        run([sys.executable, str(scripts / "lower_ptx_fp8_mma.py"), str(stages[3]),
             str(stages[4]), str(reports[4]), "--expected-count", str(fp8_count), "--compact"])
        run([sys.executable, str(scripts / "lower_ptx_f16_mma.py"), str(stages[4]),
             str(stages[5]), str(reports[5]), "--expected-count", str(f16_count)])
        run([sys.executable, str(scripts / "lower_ptx_vit_compat.py"), str(stages[5]),
             str(stages[6]), str(reports[6]), "--expected-vector-red", str(vector_red),
             "--expected-release-fence", str(fence)])
        run([sys.executable, str(scripts / "lower_ptx_surface_compat.py"), str(stages[6]),
             str(stages[7]), str(reports[7]), "--expected-count", str(surface_count)])
        run([sys.executable, str(scripts / "lower_ptx_null_texture.py"), str(stages[7]),
             str(stages[8]), str(reports[8]), "--expected-count", str(null_texture_count)])

    reports = []
    for path in output.glob("*.json"):
        reports.append(json.loads(path.read_text(encoding="utf-8-sig")))
    status = "PASS" if len(reports) == len(ENTRIES) * 9 and all(
        report.get("status") == "PASS" for report in reports) else "FAIL"
    manifest = {
        "schema": 1,
        "experiment": "decoder_slots99_154_strict_lowering",
        "status": status,
        "entry_count": len(ENTRIES),
        "report_count": len(reports),
        "entries": [{"function": entry[0], "tag": entry[1]} for entry in ENTRIES],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
