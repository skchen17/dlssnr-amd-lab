import sys
from pathlib import Path
import struct
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_ffx_live_entries import entry_jump, provider_route_snapshot, provider_metadata_snapshot, PINNED_FFX_SHA256, DISPATCH_CODE, RESOLVER_CODE, FORWARD_CODE


def no_read(address, size):
    raise AssertionError('unexpected indirect read')


def test_relative_entry_jump_signed():
    assert entry_jump(b'\xe9' + struct.pack('<i', -20), 100, no_read)['target'] == 85


def test_indirect_entry_jump():
    calls = []
    def read(address, size):
        calls.append((address, size))
        return struct.pack('<Q', 0x123456789)
    result = entry_jump(b'\xff\x25' + struct.pack('<i', 10), 100, read)
    assert calls == [(116, 8)]
    assert result['target'] == 0x123456789


def test_absolute_entry_jump():
    result = entry_jump(b'\x48\xb8' + struct.pack('<Q', 0x987654321) + b'\xff\xe0', 100, no_read)
    assert result['target'] == 0x987654321


def test_unknown_or_truncated_is_not_a_hook_claim():
    for data in (b'', b'\xe9', b'\x48\x89\x5c\x24\x08', b'\xff\x15' + bytes(4)):
        assert entry_jump(data, 100, no_read) is None


def route_memory(forward=False):
    base, context, provider, table = 0x100000, 0x200000, 0x210000, 0x220000
    values = {base + 0x1360: DISPATCH_CODE, base + 0x15c0: RESOLVER_CODE,
              context: struct.pack('<Q', provider), provider: struct.pack('<Q', table),
              table + 0x40: struct.pack('<Q', base + 0x15b0 if forward else 0x300000)}
    if forward:
        values[base + 0x15b0] = FORWARD_CODE
        values[provider + 0x40] = struct.pack('<Q', 0x400000)
    def read(address, size):
        value = values[address]
        assert len(value) == size
        return value
    return base, context, values, read


@pytest.mark.parametrize('forward', [False, True])
def test_exact_private_provider_route(forward):
    base, context, _, read = route_memory(forward)
    route = provider_route_snapshot(read, base, context, PINNED_FFX_SHA256)
    assert route['provider_object'] == 0x210000
    assert route['external_callback'] == (0x400000 if forward else None)


def test_unknown_image_rejected_before_memory_read():
    with pytest.raises(ValueError, match='unsupported'):
        provider_route_snapshot(no_read, 0, 0, 'unknown')


def test_changed_dispatch_code_rejected():
    base, context, values, read = route_memory()
    values[base + 0x1360] = bytes(len(DISPATCH_CODE))
    with pytest.raises(ValueError, match='signature'):
        provider_route_snapshot(read, base, context, PINNED_FFX_SHA256)


def test_null_provider_rejected():
    base, context, values, read = route_memory()
    values[context] = bytes(8)
    with pytest.raises(ValueError, match='null'):
        provider_route_snapshot(read, base, context, PINNED_FFX_SHA256)


def test_unstable_route_rejected():
    base, context, values, read = route_memory()
    seen = 0
    def changing(address, size):
        nonlocal seen
        if address == 0x220040:
            seen += 1
            if seen == 2:
                return struct.pack('<Q', 0x300100)
        return read(address, size)
    with pytest.raises(ValueError, match='changed'):
        provider_route_snapshot(changing, base, context, PINNED_FFX_SHA256)


@pytest.mark.parametrize('invalid', [False, True])
def test_metadata_requires_known_getters(invalid):
    base = 0x100000
    values = {0x220010: struct.pack('<Q', base + 0x1790), 0x220018: struct.pack('<Q', base + 0x1980),
              base + 0x1790: bytes.fromhex('488b4108c3'), base + 0x1980: bytes.fromhex('488b4110c3'),
              0x210008: struct.pack('<QQ', 123, 0x400000)}
    for i, c in enumerate(b'4.1.1 *\0'):
        values[0x400000 + i] = bytes([c])
    if invalid:
        values[base + 0x1980] = bytes(5)
    def read(address, size):
        assert len(values[address]) == size
        return values[address]
    route = dict(provider_object=0x210000, provider_vtable=0x220000)
    if invalid:
        with pytest.raises(ValueError, match='getters'):
            provider_metadata_snapshot(read, base, route)
    else:
        assert provider_metadata_snapshot(read, base, route) == dict(version_id=123, version_name='4.1.1 *')
