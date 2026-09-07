from pathlib import Path


def test_null_hip_stream_is_accepted_as_the_default_stream():
    source=(Path(__file__).resolve().parents[1]/'tools/native_fp8_chain/resident_fp8_chain.hip').read_text()
    assert '||!stream' not in source
    assert 'reinterpret_cast<hipStream_t>(stream)' in source
