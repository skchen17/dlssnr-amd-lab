"""Static byte/layout audit for encoder and decoder scale boundaries.

This is a storage-plan audit, not a GPU timing tool. Byte totals are explicit
FP16 feature materializations visible in the Python implementation; allocator
internals, weight reads and temporary implementation details of torch kernels
are intentionally excluded and must be measured by the GPU benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SIZES={"1080p":(1920,1080),"1440p":(2560,1440),"4k":(3840,2160)}


def padded(width,height):
    return (width+127)//128*128,(height+127)//128*128


def encoder_row(width,height,channels):
    source_bytes=width*height*channels*2
    target_bytes=source_bytes//2
    legacy_materializations={
        "scatter_packed":source_bytes,
        "unpack_current_logical":source_bytes,
        "pooled_fp16":source_bytes//4,
        "channel_permuted_pool":source_bytes//4,
        "projected_quantized_down":target_bytes,
        "pack_image_downsampled":target_bytes,
        "pack_outview":target_bytes,
        "quantized_skip":source_bytes,
        "next_unpack_outview":target_bytes,
        "next_pack_image":target_bytes,
    }
    resident_materializations={key:value for key,value in legacy_materializations.items()
        if key not in ("pack_outview","next_unpack_outview","next_pack_image")}
    direct_outputs={"quantized_skip":source_bytes,"next_resident":target_bytes}
    bounded_native={
        "preceding_swin_scatter":source_bytes,
        "pool_permute_matrix_input":source_bytes//4,
        "quantized_skip":source_bytes,
        "projection_output":target_bytes,
        "quantized_resident":target_bytes,
    }
    return {
        "transition":f"C{channels}->C{channels*2}",
        "source_geometry":[width,height,channels],
        "target_geometry":[width//2,height//2,channels*2],
        "source_bytes":source_bytes,
        "target_bytes":target_bytes,
        "unused_downsampled_hot_path_bytes":target_bytes,
        "legacy_named_tensor_count":len(legacy_materializations),
        "legacy_named_tensor_write_bytes":sum(legacy_materializations.values()),
        "resident_routing_named_tensor_count":len(resident_materializations),
        "resident_routing_named_tensor_write_bytes":sum(resident_materializations.values()),
        "direct_transition_output_bytes":sum(direct_outputs.values()),
        "bounded_native_named_tensor_count":len(bounded_native),
        "bounded_native_named_tensor_write_bytes":sum(bounded_native.values()),
        "bounded_native_authored_dispatches":2,
        "bounded_native_library_projection_calls":1,
        "legacy_layout_steps":["scatter","unpack_image","channel_permutation","pack_image",
            "pack_outview","next_unpack_outview","next_pack_image","next_gather"],
        "resident_layout_steps":["scatter","unpack_image","channel_permutation","pack_image","next_gather"],
        "target_direct_layout_steps":["direct_transition_to_skip_and_resident","next_native_consume"],
        "direct_outputs":direct_outputs,
        "bounded_native_materializations":bounded_native,
    }


def decoder_row(width,height,channels):
    destination_bytes=width*height*channels*2
    low_bytes=destination_bytes//2
    projected_low_bytes=destination_bytes//4
    legacy={
        "unpack_low_outview":low_bytes,
        "unpack_skip":destination_bytes,
        "permute_low":low_bytes,
        "project_low":projected_low_bytes,
        "repeat_interleave":destination_bytes,
        "scaled_skip":destination_bytes,
        "logical_fused":destination_bytes,
        "pack_image":destination_bytes,
        "gather_windows":destination_bytes,
    }
    direct_outputs={"next_swin_resident_windows":destination_bytes}
    bounded_native={
        "unpack_permute_matrix_input":low_bytes,
        "projection_output":projected_low_bytes,
        "expanded_skip_fused_resident":destination_bytes,
        "next_gather_windows":destination_bytes,
    }
    return {
        "transition":f"C{channels*2}->C{channels}",
        "low_geometry":[width//2,height//2,channels*2],
        "skip_and_target_geometry":[width,height,channels],
        "low_bytes":low_bytes,
        "skip_bytes":destination_bytes,
        "target_bytes":destination_bytes,
        "legacy_named_tensor_count":len(legacy),
        "legacy_named_tensor_write_bytes":sum(legacy.values()),
        "direct_transition_output_bytes":sum(direct_outputs.values()),
        "bounded_native_named_tensor_count":len(bounded_native),
        "bounded_native_named_tensor_write_bytes":sum(bounded_native.values()),
        "bounded_native_authored_dispatches":2,
        "bounded_native_library_projection_calls":1,
        "legacy_layout_steps":["unpack_outview","unpack_skip","channel_permutation","projection",
            "repeat_interleave","skip_scale_add","pack_image","gather"],
        "target_direct_layout_steps":["resident_low_and_skip_to_next_swin_layout"],
        "direct_outputs":direct_outputs,
        "bounded_native_materializations":bounded_native,
    }


def find_downsampled_consumers(root):
    consumers=[]
    for path in sorted((root/"scripts").glob("*.py")):
        if path.name==Path(__file__).name:continue
        for number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
            if "downsampled" in line:
                consumers.append({"file":str(path.relative_to(root)).replace("\\","/"),"line":number,
                    "text":line.strip(),"hot_path":path.name=="native_whole_frame.py"})
    return consumers


def audit(root):
    sizes={}
    for name,(width,height) in SIZES.items():
        pw,ph=padded(width,height)
        enc=[]
        for level,channels in enumerate((32,64,128,256),1):
            enc.append(encoder_row(pw//(2**level),ph//(2**level),channels))
        dec=[]
        for divisor,channels in ((16,256),(8,128),(4,64),(2,32)):
            dec.append(decoder_row(pw//divisor,ph//divisor,channels))
        sizes[name]={"input_geometry":[width,height],"padded_geometry":[pw,ph],
            "encoder":enc,"decoder":dec}
    consumers=find_downsampled_consumers(root)
    return {
        "schema":1,
        "scope":"Static visible FP16 feature materializations; excludes weights, allocator internals and GPU timing.",
        "sizes":sizes,
        "downsampled_consumers":consumers,
        "downsampled_used_by_runtime_hot_path":any(row["hot_path"] for row in consumers),
        "conclusion":"downsampled is diagnostic-only; resident should replace captured_outview in the runtime chain",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=audit(root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"downsampled_used_by_runtime_hot_path":report["downsampled_used_by_runtime_hot_path"]}))


if __name__=="__main__":main()
