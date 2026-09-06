#!/usr/bin/env python3
"""Lower m16n8k32 E4M3 MMA to warp shuffles and scalar FP32 accumulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path


REG = r"%[A-Za-z_][A-Za-z0-9_]*"
FP8_MMA = re.compile(
    rf"mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*"
    rf"\{{\s*(?P<d0>{REG})\s*,\s*(?P<d1>{REG})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<a0>{REG})\s*,\s*(?P<a1>{REG})\s*,\s*(?P<a2>{REG})\s*,\s*(?P<a3>{REG})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<b0>{REG})\s*,\s*(?P<b1>{REG})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<c0>{REG})\s*,\s*(?P<c1>{REG})\s*\}}\s*;",
    re.DOTALL,
)


def e4m3_constructed(code: int) -> float:
    magnitude = code & 0x7F
    exponent, mantissa = magnitude >> 3, magnitude & 7
    if magnitude == 0x7F:
        return math.nan
    numerator = mantissa if exponent == 0 else mantissa + 8
    scale_exponent = 118 if exponent == 0 else exponent + 117
    scale = struct.unpack("<f", struct.pack("<I", scale_exponent << 23))[0]
    value = float(numerator) * scale
    return -value if code & 0x80 else value


def cd_coord(lane: int, element: int) -> tuple[int, int]:
    group, thread = lane >> 2, lane & 3
    return group + (8 if element >= 2 else 0), thread * 2 + (element & 1)


def a_source(lane: int, element: int, k: int) -> tuple[int, int]:
    row, _ = cd_coord(lane, element)
    source_lane = (row & 7) * 4 + ((k & 15) >> 2)
    source_element = (k & 3) + (4 if row >= 8 else 0) + (8 if k >= 16 else 0)
    return source_lane, source_element


def b_source(lane: int, element: int, k: int) -> tuple[int, int]:
    _, column = cd_coord(lane, element)
    source_lane = column * 4 + ((k & 15) >> 2)
    source_element = (k & 3) + (4 if k >= 16 else 0)
    return source_lane, source_element


def emit_decode(code: str, output: str, r, p,
                subnormal_mode: str = "preserve") -> list[str]:
    lines = [
        f"and.b32 {r('magnitude')}, {code}, 127;",
        f"shr.u32 {r('exponent')}, {r('magnitude')}, 3;",
        f"and.b32 {r('mantissa')}, {r('magnitude')}, 7;",
        f"add.u32 {r('normal_numerator')}, {r('mantissa')}, 8;",
        f"setp.eq.u32 {p('subnormal')}, {r('exponent')}, 0;",
        f"selp.b32 {r('numerator')}, {r('mantissa')}, {r('normal_numerator')}, {p('subnormal')};",
        f"cvt.rn.f32.u32 {output}, {r('numerator')};",
        f"add.u32 {r('normal_scale_exp')}, {r('exponent')}, 117;",
        f"selp.b32 {r('scale_exp')}, 118, {r('normal_scale_exp')}, {p('subnormal')};",
        f"shl.b32 {r('scale_bits')}, {r('scale_exp')}, 23;",
        f"mul.rn.f32 {output}, {output}, {r('scale_bits')};",
        f"and.b32 {r('sign_bits')}, {code}, 128;",
        f"shl.b32 {r('sign_bits')}, {r('sign_bits')}, 24;",
        f"xor.b32 {output}, {output}, {r('sign_bits')};",
        f"setp.eq.u32 {p('nan')}, {r('magnitude')}, 127;",
        f"selp.b32 {output}, 2143289344, {output}, {p('nan')};",
    ]
    if subnormal_mode == "flush_inputs":
        lines.append(f"selp.b32 {output}, 0, {output}, {p('subnormal')};")
    elif subnormal_mode != "preserve":
        raise ValueError(f"unsupported subnormal mode {subnormal_mode}")
    return lines


def accumulator_rounding(accumulator: str, mode: str, rounding: str, r, h) -> list[str]:
    if mode == "f32":
        return []
    if mode == "f16":
        return [
            f"cvt.rn.f16.f32 {h('acc_round')}, {accumulator};",
            f"cvt.f32.f16 {accumulator}, {h('acc_round')};",
        ]
    if not mode.startswith("mantissa"):
        raise ValueError(f"unsupported accumulation mode {mode}")
    mantissa_bits = int(mode.removeprefix("mantissa"))
    if not 1 <= mantissa_bits <= 22:
        raise ValueError("intermediate mantissa bits must be in 1..22")
    shift = 23 - mantissa_bits
    mask = (0xFFFFFFFF << shift) & 0xFFFFFFFF
    lines = [
        f"mov.b32 {r('round_bits')}, {accumulator};",
    ]
    if rounding == "rn":
        bias = (1 << (shift - 1)) - 1
        lines += [
            f"shr.u32 {r('round_lsb')}, {r('round_bits')}, {shift};",
            f"and.b32 {r('round_lsb')}, {r('round_lsb')}, 1;",
            f"add.u32 {r('round_bits')}, {r('round_bits')}, {bias};",
            f"add.u32 {r('round_bits')}, {r('round_bits')}, {r('round_lsb')};",
        ]
    elif rounding != "rz":
        raise ValueError(f"unsupported accumulator rounding {rounding}")
    lines += [
        f"and.b32 {r('round_bits')}, {r('round_bits')}, {mask};",
        f"mov.b32 {accumulator}, {r('round_bits')};",
    ]
    return lines


def replacement(match: re.Match[str], index: int, accumulation_mode: str = "f32",
                accumulation_order: str = "sequential_c_first",
                accumulation_rounding_mode: str = "rn",
                subnormal_mode: str = "preserve",
                product_mode: str = "preserve") -> str:
    d = [match.group("d0"), match.group("d1")]
    a = [match.group(f"a{i}") for i in range(4)]
    b = [match.group(f"b{i}") for i in range(2)]
    c = [match.group(f"c{i}") for i in range(2)]
    r = lambda name: f"%__fp8mma_{index}_r_{name}"
    h = lambda name: f"%__fp8mma_{index}_h_{name}"
    p = lambda name: f"%__fp8mma_{index}_p_{name}"
    interleaved = (int(accumulation_order.removeprefix("interleaved"))
                   if accumulation_order.startswith("interleaved") else 0)
    pairwise = accumulation_order.startswith("pairwise")
    partial_count = 33 if pairwise and accumulation_order.endswith("c_first") else (
        32 if pairwise else interleaved
    )
    b32 = [r(name) for name in (
        "lane", "group_base", "thread_x8", "source_a", "source_b0", "source_b1",
        "word_a0", "word_a1", "word_b0", "word_b1", "code",
        "magnitude", "exponent", "mantissa", "normal_numerator", "numerator",
        "normal_scale_exp", "scale_exp", "scale_bits", "sign_bits",
        "a0", "a1", "b0", "b1", "acc0", "acc1", "acc2", "acc3", "bits",
        "round_bits", "round_lsb", "zero",
        "cval0", "cval1", "cval2", "cval3",
        "product0", "product1", "product2", "product3",
        "product_abs0", "product_abs1", "product_abs2", "product_abs3",
        *(f"partial{element}_{partial}" for element in range(4)
          for partial in range(partial_count))
    )]
    b16 = [h(name) for name in (
        "c", "out0", "out1", "out2", "out3", "acc_round", "product_round"
    )]
    predicates = [p(name) for name in (
        "valid_a0", "valid_a1", "valid_b0", "valid_b1", "subnormal", "nan",
        "product_sub0", "product_sub1", "product_sub2", "product_sub3"
    )]
    lines = [
        f".reg .b32 {', '.join(b32)};",
        f".reg .b16 {', '.join(b16)};",
        f".reg .pred {', '.join(predicates)};",
        f"mov.u32 {r('lane')}, %laneid;",
        f"and.b32 {r('group_base')}, {r('lane')}, 28;",
        f"and.b32 {r('thread_x8')}, {r('lane')}, 3;",
        f"shl.b32 {r('thread_x8')}, {r('thread_x8')}, 3;",
        f"mov.b32 {r('zero')}, 0;",
    ]
    c_last = (accumulation_order in ("sequential_c_last", "reverse_c_last")
              or bool(interleaved) or accumulation_order.endswith("c_last")
              or accumulation_order.startswith("dot_f16"))
    for element in range(4):
        c_word = c[element // 2]
        lines.append(
            f"shr.u32 {r('bits')}, {c_word}, 16;" if element & 1
            else f"and.b32 {r('bits')}, {c_word}, 65535;"
        )
        lines.extend([
            f"cvt.u16.u32 {h('c')}, {r('bits')};",
            f"cvt.f32.f16 {r(f'cval{element}')}, {h('c')};",
            (f"mov.b32 {r(f'acc{element}')}, 0;" if c_last
             else f"mov.b32 {r(f'acc{element}')}, {r(f'cval{element}')};"),
        ])
    if interleaved or pairwise:
        for element in range(4):
            for partial in range(partial_count):
                lines.append(f"mov.b32 {r(f'partial{element}_{partial}')}, 0;")
            if pairwise and accumulation_order.endswith("c_first"):
                lines.append(f"mov.b32 {r(f'partial{element}_0')}, {r(f'cval{element}')};")
    k_values = (range(31, -1, -1) if accumulation_order.startswith("reverse") else range(32))
    for k in k_values:
        source_thread = (k & 15) >> 2
        byte_shift = (k & 3) * 8
        a_low_register = a[2 if k >= 16 else 0]
        a_high_register = a[3 if k >= 16 else 1]
        b_register = b[1 if k >= 16 else 0]
        lines.extend([
            f"add.u32 {r('source_a')}, {r('group_base')}, {source_thread};",
            f"add.u32 {r('source_b0')}, {r('thread_x8')}, {source_thread};",
            f"add.u32 {r('source_b1')}, {r('source_b0')}, 4;",
            f"shfl.sync.idx.b32 {r('word_a0')}|{p('valid_a0')}, {a_low_register}, {r('source_a')}, 31, -1;",
            f"shfl.sync.idx.b32 {r('word_a1')}|{p('valid_a1')}, {a_high_register}, {r('source_a')}, 31, -1;",
            f"shfl.sync.idx.b32 {r('word_b0')}|{p('valid_b0')}, {b_register}, {r('source_b0')}, 31, -1;",
            f"shfl.sync.idx.b32 {r('word_b1')}|{p('valid_b1')}, {b_register}, {r('source_b1')}, 31, -1;",
        ])
        for word, output in (
            (r("word_a0"), r("a0")), (r("word_a1"), r("a1")),
            (r("word_b0"), r("b0")), (r("word_b1"), r("b1")),
        ):
            lines.append(f"shr.u32 {r('code')}, {word}, {byte_shift};" if byte_shift
                         else f"mov.b32 {r('code')}, {word};")
            lines.append(f"and.b32 {r('code')}, {r('code')}, 255;")
            lines.extend(emit_decode(r("code"), output, r, p, subnormal_mode))
        if interleaved:
            targets = [r(f"partial{element}_{k % interleaved}") for element in range(4)]
        elif pairwise:
            partial = k + (1 if accumulation_order.endswith("c_first") else 0)
            targets = [r(f"partial{element}_{partial}") for element in range(4)]
        else:
            targets = [r(f"acc{element}") for element in range(4)]
        pairs = ((r("a0"), r("b0")), (r("a0"), r("b1")),
                 (r("a1"), r("b0")), (r("a1"), r("b1")))
        if product_mode == "preserve":
            for element, (left, right) in enumerate(pairs):
                lines.append(f"fma.rn.f32 {targets[element]}, {left}, {right}, {targets[element]};")
        elif product_mode == "f16":
            for element, (left, right) in enumerate(pairs):
                product = r(f"product{element}")
                lines += [
                    f"mul.rn.f32 {product}, {left}, {right};",
                    f"cvt.rn.f16.f32 {h('product_round')}, {product};",
                    f"cvt.f32.f16 {product}, {h('product_round')};",
                    f"add.rn.f32 {targets[element]}, {targets[element]}, {product};",
                ]
        elif product_mode == "flush_f16_subnormals":
            for element, (left, right) in enumerate(pairs):
                product = r(f"product{element}")
                absolute = r(f"product_abs{element}")
                predicate = p(f"product_sub{element}")
                lines += [
                    f"mul.rn.f32 {product}, {left}, {right};",
                    f"abs.f32 {absolute}, {product};",
                    f"setp.lt.f32 {predicate}, {absolute}, 0f38800000;",
                    f"selp.f32 {product}, 0f00000000, {product}, {predicate};",
                    f"add.rn.f32 {targets[element]}, {targets[element]}, {product};",
                ]
        else:
            raise ValueError(f"unsupported product mode {product_mode}")
        if not pairwise:
            for element in range(4):
                lines.extend(accumulator_rounding(
                    targets[element], accumulation_mode, accumulation_rounding_mode, r, h
                ))
    if pairwise:
        for element in range(4):
            active = [r(f"partial{element}_{partial}") for partial in range(partial_count)]
            while len(active) > 1:
                next_active = []
                for position in range(0, len(active), 2):
                    left = active[position]
                    right = active[position + 1] if position + 1 < len(active) else r("zero")
                    lines.append(f"add.rn.f32 {left}, {left}, {right};")
                    lines.extend(accumulator_rounding(
                        left, accumulation_mode, accumulation_rounding_mode, r, h
                    ))
                    next_active.append(left)
                active = next_active
            lines.append(f"mov.b32 {r(f'acc{element}')}, {active[0]};")
            if accumulation_order.endswith("c_last"):
                lines.append(f"add.rn.f32 {r(f'acc{element}')}, {r(f'acc{element}')}, {r(f'cval{element}')};")
                lines.extend(accumulator_rounding(
                    r(f"acc{element}"), accumulation_mode, accumulation_rounding_mode, r, h
                ))
    elif interleaved:
        for element in range(4):
            active = [r(f"partial{element}_{partial}") for partial in range(interleaved)]
            while len(active) > 1:
                next_active = []
                for left, right in zip(active[0::2], active[1::2]):
                    lines.append(f"add.rn.f32 {left}, {left}, {right};")
                    next_active.append(left)
                active = next_active
            lines.append(f"add.rn.f32 {r(f'acc{element}')}, {active[0]}, {r(f'cval{element}')};")
    elif accumulation_order.startswith("dot_f16"):
        for element in range(4):
            lines += [
                f"cvt.rn.f16.f32 {h('acc_round')}, {r(f'acc{element}')};",
                f"cvt.f32.f16 {r(f'acc{element}')}, {h('acc_round')};",
                f"add.rn.f32 {r(f'acc{element}')}, {r(f'acc{element}')}, {r(f'cval{element}')};",
            ]
            if accumulation_order == "dot_f16_then_f16_add_c":
                lines += [
                    f"cvt.rn.f16.f32 {h('acc_round')}, {r(f'acc{element}')};",
                    f"cvt.f32.f16 {r(f'acc{element}')}, {h('acc_round')};",
                ]
    elif c_last:
        for element in range(4):
            lines.append(f"add.rn.f32 {r(f'acc{element}')}, {r(f'acc{element}')}, {r(f'cval{element}')};")
    for element in range(4):
        lines.append(f"cvt.rn.f16.f32 {h(f'out{element}')}, {r(f'acc{element}')};")
    lines.extend([
        f"mov.b32 {d[0]}, {{{h('out0')}, {h('out1')}}};",
        f"mov.b32 {d[1]}, {{{h('out2')}, {h('out3')}}};",
    ])
    return "{\n" + "\n".join(lines) + "\n}"


def lower(text: str, accumulation_mode: str = "f32",
          accumulation_order: str = "sequential_c_first",
          accumulation_rounding_mode: str = "rn",
          subnormal_mode: str = "preserve",
          product_mode: str = "preserve") -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        result = replacement(match, count, accumulation_mode, accumulation_order,
                             accumulation_rounding_mode, subnormal_mode, product_mode)
        count += 1
        return result

    return FP8_MMA.sub(replace, text), count


COMPACT_HELPER_NAME = "__dlssnr_fp8_m16n8k32"


def compact_call(match: re.Match[str], index: int) -> str:
    inputs = [match.group(f"a{i}") for i in range(4)]
    inputs += [match.group(f"b{i}") for i in range(2)]
    inputs += [match.group(f"c{i}") for i in range(2)]
    outputs = [match.group("d0"), match.group("d1")]
    args = [f"__fp8mma_call_{index}_arg_{i}" for i in range(8)]
    rets = [f"__fp8mma_call_{index}_ret_{i}" for i in range(2)]
    lines = ["{"]
    lines += [f".param .b32 {name};" for name in args + rets]
    lines += [f"st.param.b32 [{name}], {value};" for name, value in zip(args, inputs)]
    lines.append(
        f"call.uni ({', '.join(rets)}), {COMPACT_HELPER_NAME}, "
        f"({', '.join(args)});"
    )
    lines += [f"ld.param.b32 {value}, [{name}];" for value, name in zip(outputs, rets)]
    lines.append("}")
    return "\n".join(lines)


def compact_helper(accumulation_mode: str = "f32",
                   accumulation_order: str = "sequential_c_first",
                   accumulation_rounding_mode: str = "rn",
                   subnormal_mode: str = "preserve",
                   product_mode: str = "preserve") -> str:
    dummy = (
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%__d0, %__d1}, {%__a0, %__a1, %__a2, %__a3}, "
        "{%__b0, %__b1}, {%__c0, %__c1};"
    )
    match = FP8_MMA.search(dummy)
    assert match is not None
    scalar_body = replacement(match, "helper", accumulation_mode, accumulation_order,
                              accumulation_rounding_mode, subnormal_mode, product_mode)
    params = [f"__arg_{i}" for i in range(8)]
    signature = ",\n".join(f"    .param .b32 {name}" for name in params)
    loads = "\n".join(
        f"ld.param.b32 %__{kind}{slot}, [{param}];"
        for param, (kind, slot) in zip(
            params,
            [("a", 0), ("a", 1), ("a", 2), ("a", 3),
             ("b", 0), ("b", 1), ("c", 0), ("c", 1)],
        )
    )
    return (
        f".func (.param .b32 __ret_0, .param .b32 __ret_1)\n"
        f"{COMPACT_HELPER_NAME}(\n{signature}\n)\n{{\n"
        ".reg .b32 %__a0, %__a1, %__a2, %__a3, %__b0, %__b1, "
        "%__c0, %__c1, %__d0, %__d1;\n"
        f"{loads}\n{scalar_body}\n"
        "st.param.b32 [__ret_0], %__d0;\n"
        "st.param.b32 [__ret_1], %__d1;\n"
        "ret;\n}\n"
    )


def lower_compact(text: str, accumulation_mode: str = "f32",
                  accumulation_order: str = "sequential_c_first",
                  accumulation_rounding_mode: str = "rn",
                  subnormal_mode: str = "preserve",
                  product_mode: str = "preserve") -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        result = compact_call(match, count)
        count += 1
        return result

    lowered = FP8_MMA.sub(replace, text)
    if count:
        entry = re.search(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+", lowered)
        if not entry:
            raise ValueError("PTX entry declaration not found for compact helper insertion")
        lowered = (lowered[:entry.start()]
                   + compact_helper(accumulation_mode, accumulation_order,
                                    accumulation_rounding_mode, subnormal_mode, product_mode)
                   + "\n" + lowered[entry.start():])
    return lowered, count


def candidate_model_options(name: str) -> dict[str, str]:
    options = {
        "accumulation_mode": "f32",
        "accumulation_order": "sequential_c_first",
        "accumulation_rounding_mode": "rn",
        "subnormal_mode": "preserve",
        "product_mode": "preserve",
    }
    direct_orders = {
        "f32_sequential_c_first": "sequential_c_first",
        "f32_sequential_c_last": "sequential_c_last",
        "f32_reverse_c_first": "reverse_c_first",
        "f32_reverse_c_last": "reverse_c_last",
        "f32_interleaved2": "interleaved2",
        "f32_interleaved4": "interleaved4",
        "f32_interleaved8": "interleaved8",
        "f32_pairwise_c_first": "pairwise_c_first",
        "f32_pairwise_c_last": "pairwise_c_last",
    }
    if name in direct_orders:
        options["accumulation_order"] = direct_orders[name]
    elif name == "f16_each_fma":
        options["accumulation_mode"] = "f16"
    elif match := re.fullmatch(r"mantissa(1[0-9]|2[0-2])(_rz)?_each_fma", name):
        options["accumulation_mode"] = f"mantissa{match.group(1)}"
        options["accumulation_rounding_mode"] = "rz" if match.group(2) else "rn"
    elif name == "f16_products_f32_acc":
        options["product_mode"] = "f16"
    elif name == "f16_products_f16_acc":
        options["product_mode"] = "f16"
        options["accumulation_mode"] = "f16"
    elif name == "f16_dot_then_f32_add_c":
        options["accumulation_order"] = "dot_f16_then_f32_add_c"
    elif name == "f16_dot_then_f16_add_c":
        options["accumulation_order"] = "dot_f16_then_f16_add_c"
    elif name == "flush_e4m3_inputs":
        options["subnormal_mode"] = "flush_inputs"
    elif name == "flush_f16_subnormal_products":
        options["product_mode"] = "flush_f16_subnormals"
    elif name == "exact_fsum":
        raise ValueError("exact_fsum is a CPU diagnostic and has no finite PTX rewrite")
    else:
        raise ValueError(f"unsupported candidate model {name}")
    return options


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-count", type=int, default=256)
    parser.add_argument("--compact", action="store_true",
                        help="emit one scalar helper function and compact call sites")
    parser.add_argument("--candidate-model",
                        help="apply a candidate name emitted by analyze_slot3_mma_trace.py")
    parser.add_argument("--accumulation-mode", default="f32",
                        choices=["f32", "f16", *(f"mantissa{bits}" for bits in range(10, 23))],
                        help="round each internal FP8 MMA accumulation step")
    parser.add_argument("--accumulation-rounding", default="rn", choices=["rn", "rz"],
                        help="round-to-nearest-even or truncate limited mantissas")
    parser.add_argument("--accumulation-order", default="sequential_c_first",
                        choices=["sequential_c_first", "sequential_c_last",
                                 "reverse_c_first", "reverse_c_last",
                                 "interleaved2", "interleaved4", "interleaved8",
                                 "pairwise_c_first", "pairwise_c_last",
                                 "dot_f16_then_f32_add_c", "dot_f16_then_f16_add_c"],
                        help="select the internal dot-product reduction order")
    parser.add_argument("--subnormal-mode", default="preserve",
                        choices=["preserve", "flush_inputs"],
                        help="select the FP8 MMA subnormal-input policy")
    parser.add_argument("--product-mode", default="preserve",
                        choices=["preserve", "f16", "flush_f16_subnormals"],
                        help="select the FP8 product subnormal policy")
    args = parser.parse_args()
    try:
        options = (candidate_model_options(args.candidate_model) if args.candidate_model else {
            "accumulation_mode": args.accumulation_mode,
            "accumulation_order": args.accumulation_order,
            "accumulation_rounding_mode": args.accumulation_rounding,
            "subnormal_mode": args.subnormal_mode,
            "product_mode": args.product_mode,
        })
    except ValueError as error:
        parser.error(str(error))
    source = args.input.read_bytes()
    lowering = lower_compact if args.compact else lower
    lowered, count = lowering(source.decode("utf-8"), **options)
    output = lowered.encode("utf-8")
    remaining = len(FP8_MMA.findall(lowered))
    status = "PASS" if count == args.expected_count and remaining == 0 else "FAIL"
    report = {
        "schema": 1,
        "experiment": ("ptx_fp8_m16n8k32_compact_scalar_lowering" if args.compact
                       else "ptx_fp8_m16n8k32_scalar_lowering"),
        "status": status,
        "classification": "RTX_AMD_ORACLE_VALIDATED_PTX_REWRITE",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "fp8_mma_lowered": count,
        "expected_fp8_mma": args.expected_count,
        "remaining_fp8_mma": remaining,
        "candidate_model": args.candidate_model,
        "accumulation": ("f32_fma_then_single_f16_round"
                         if options["accumulation_mode"] == "f32"
                         else f"{options['accumulation_mode']}_round_after_each_fma_then_f16"),
        **options,
        "compact_helper": args.compact,
        "translator_execution_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
