import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_scale_transitions import audit,decoder_row,encoder_row,padded


def test_padding_and_encoder_byte_accounting():
    assert padded(1920,1080)==(1920,1152)
    row=encoder_row(960,576,32)
    assert row["target_geometry"]==[480,288,64]
    assert row["target_bytes"]==row["source_bytes"]//2
    assert row["legacy_named_tensor_count"]==10
    assert row["resident_routing_named_tensor_count"]==7
    assert row["direct_transition_output_bytes"]==row["source_bytes"]+row["target_bytes"]
    assert row["bounded_native_authored_dispatches"]==2
    assert row["bounded_native_named_tensor_write_bytes"]<row["resident_routing_named_tensor_write_bytes"]


def test_decoder_byte_accounting():
    row=decoder_row(960,576,32)
    assert row["transition"]=="C64->C32"
    assert row["low_bytes"]==row["target_bytes"]//2
    assert row["legacy_named_tensor_count"]==9
    assert row["bounded_native_authored_dispatches"]==2
    assert row["bounded_native_named_tensor_write_bytes"]<row["legacy_named_tensor_write_bytes"]


def test_runtime_does_not_consume_downsampled_key():
    root=Path(__file__).resolve().parents[1]
    report=audit(root)
    assert not report["downsampled_used_by_runtime_hot_path"]
    assert len(report["sizes"]["4k"]["encoder"])==4
    assert len(report["sizes"]["4k"]["decoder"])==4
