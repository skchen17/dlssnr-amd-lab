"""Bounded read-only entry-point audit of the exact game process in provenance.

No injection, memory writes, context queries, capture arming, or GPU work.
This snapshot detects some entry jumps, not all hooks or execution paths.
"""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import os
from pathlib import Path
import struct
from datetime import datetime

PINNED_FFX_SHA256 = '77809405A0FF464B63654F1264F0EC0FCF8F243DAC7C15B5F5C032615520D143'
# Exact disassembly of this pinned image, not a general/public context ABI.
DISPATCH_CODE = bytes.fromhex(
    '48895c2408574883ec20488bfa488bd94885d274274885c97422e841020000'
    '488bd3488bc84c8b004d8b48404c8bc7488b5c24304883c4205f49ffe1'
    '488b5c2430b8060000004883c4205fc3')
RESOLVER_CODE = bytes.fromhex('488b01488b00c3')
FORWARD_CODE = bytes.fromhex('488b41404c8bca498bc9498bd048ffe0')


def provider_route_snapshot(read, base, context, digest):
    """Read pointers only after verifying the exact private-layout machine code.

    The candidate stores *ffxContext, so only one load is needed for the
    provider object. No remote function calls and no lifetime guarantee.
    """
    if digest != PINNED_FFX_SHA256:
        raise ValueError('unsupported provider image for private route audit')
    if read(base + 0x1360, len(DISPATCH_CODE)) != DISPATCH_CODE or read(base + 0x15c0, len(RESOLVER_CODE)) != RESOLVER_CODE:
        raise ValueError('live dispatch/resolver signature mismatch')

    def pointer(address):
        value = struct.unpack('<Q', read(address, 8))[0]
        if not value:
            raise ValueError('null route pointer')
        return value

    def snapshot():
        provider = pointer(context)
        table = pointer(provider)
        dispatch = pointer(table + 0x40)
        callback = None
        if dispatch == base + 0x15b0 and read(dispatch, len(FORWARD_CODE)) == FORWARD_CODE:
            callback = pointer(provider + 0x40)
        return dict(context=context, provider_object=provider, provider_vtable=table,
                    provider_dispatch=dispatch, external_callback=callback)

    first = snapshot()
    if snapshot() != first:
        raise ValueError('route changed during snapshot')
    return first


def provider_metadata_snapshot(read, base, route):
    """Decode verified getter fields; never invoke driver code or guess offsets."""
    table, provider = route['provider_vtable'], route['provider_object']
    if (struct.unpack('<Q', read(table + 0x10, 8))[0] != base + 0x1790 or
            struct.unpack('<Q', read(table + 0x18, 8))[0] != base + 0x1980 or
            read(base + 0x1790, 5) != bytes.fromhex('488b4108c3') or
            read(base + 0x1980, 5) != bytes.fromhex('488b4110c3')):
        raise ValueError('unsupported provider metadata getters')
    fields = read(provider + 8, 16)
    version, name_pointer = struct.unpack('<QQ', fields)
    if not version or not name_pointer:
        raise ValueError('null provider metadata')
    name = bytearray()
    for offset in range(128):
        char = read(name_pointer + offset, 1)
        if char == b'\0':
            break
        if len(char) != 1 or not 32 <= char[0] < 127:
            raise ValueError('provider name is not bounded printable ASCII')
        name += char
    else:
        raise ValueError('provider name cap')
    if not name or read(provider + 8, 16) != fields:
        raise ValueError('empty/changed provider metadata')
    return dict(version_id=version, version_name=name.decode('ascii'))


def entry_jump(data, address, read):
    """Recognize only unambiguous x64 entry jump encodings; never execute them."""
    if len(data) >= 5 and data[0] == 0xE9:
        return dict(encoding='jmp_rel32', target=address + 5 + struct.unpack_from('<i', data, 1)[0])
    if len(data) >= 6 and data[:2] == b'\xff\x25':
        slot = address + 6 + struct.unpack_from('<i', data, 2)[0]
        return dict(encoding='jmp_rip_indirect', slot=slot, target=struct.unpack('<Q', read(slot, 8))[0])
    if len(data) >= 12 and data[:2] == b'\x48\xb8' and data[10:12] == b'\xff\xe0':
        return dict(encoding='mov_rax_jmp_rax', target=struct.unpack_from('<Q', data, 2)[0])
    return None


class ReadOnlyProcess:
    def __init__(self, pid):
        self.kernel = ct.WinDLL('kernel32', use_last_error=True)
        self.psapi = ct.WinDLL('psapi', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        self.kernel.OpenProcess.restype = wt.HANDLE
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]
        self.kernel.ReadProcessMemory.argtypes = [wt.HANDLE, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
        self.kernel.ReadProcessMemory.restype = wt.BOOL
        self.psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ct.POINTER(wt.HMODULE), wt.DWORD, ct.POINTER(wt.DWORD), wt.DWORD]
        self.psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, wt.HMODULE, wt.LPWSTR, wt.DWORD]
        self.kernel.GetProcessTimes.argtypes = [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4
        self.handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        self.bytes_read = 0

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def created(self):
        values = [wt.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *(ct.byref(v) for v in values)):
            raise ct.WinError(ct.get_last_error())
        return ((values[0].dwHighDateTime << 32) + values[0].dwLowDateTime) / 10000000 - 11644473600

    def read(self, address, size):
        if not 0 < size <= 4096 or self.bytes_read + size > 262144:
            raise ValueError('read budget exceeded')
        self.bytes_read += size
        data, got = ct.create_string_buffer(size), ct.c_size_t()
        if not self.kernel.ReadProcessMemory(self.handle, address, data, size, ct.byref(got)) or got.value != size:
            raise ct.WinError(ct.get_last_error())
        return data.raw

    def modules(self):
        handles = (wt.HMODULE * 1024)()
        needed = wt.DWORD()
        if not self.psapi.EnumProcessModulesEx(self.handle, handles, ct.sizeof(handles), ct.byref(needed), 3):
            raise ct.WinError(ct.get_last_error())
        if needed.value > ct.sizeof(handles):
            raise ValueError('module cap exceeded')
        result = []
        for handle in handles[:needed.value // ct.sizeof(wt.HMODULE)]:
            path = ct.create_unicode_buffer(32768)
            count = self.psapi.GetModuleFileNameExW(self.handle, handle, path, len(path))
            if not 0 < count < len(path) - 1:
                raise ValueError('module path unavailable')
            # Read only the image-size field from each mapped PE header.
            dos = self.read(handle, 64)
            offset = struct.unpack_from('<I', dos, 60)[0]
            if dos[:2] != b'MZ' or not 0 < offset <= 1048576:
                raise ValueError('module DOS header')
            nt = self.read(handle + offset, 84)
            if nt[:4] != b'PE\0\0':
                raise ValueError('module PE header')
            result.append(dict(path=path.value, base=handle, size=struct.unpack_from('<I', nt, 80)[0]))
        return result


def audit(provenance, log, provider_route=False):
    import pefile
    import psutil
    proof = json.loads(provenance.read_text(encoding='utf-8-sig'))
    if proof.get('observation_deferred') is not True:
        raise ValueError('requires deferred-session provenance')
    if log.resolve() != provenance.resolve().parent / 'session.jsonl' or log.stat().st_size > 16777216:
        raise ValueError('requires the same bounded session log')
    expected_exe = Path(r'C:\DATA\GAME\GODOFWAR\GoWR.exe')
    expected_start = datetime.fromisoformat(proof['process_start']).timestamp()
    process = psutil.Process(proof['pid'])
    if Path(process.exe()).resolve() != expected_exe.resolve() or abs(process.create_time() - expected_start) > 0.002:
        raise ValueError('process identity/start mismatch')
    remote = ReadOnlyProcess(proof['pid'])
    try:
        if abs(remote.created() - expected_start) > 0.002:
            raise ValueError('opened handle start mismatch')
        modules = remote.modules()
        session_path = Path(proof['validation_run']).parent / 'ffx_capture_session.dll'
        sessions = [m for m in modules if Path(m['path']).resolve() == session_path.resolve()]
        if len(sessions) != 1:
            raise ValueError('exact pinned session module not found')
        with session_path.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest().upper() != proof['inspector_sha256']:
                raise ValueError('session disk hash changed')

        def owner(address):
            matches = [m for m in modules if m['base'] <= address < m['base'] + m['size']]
            return dict(module=matches[0]['path'], rva=hex(address - matches[0]['base'])) if len(matches) == 1 else None

        def inspect(address):
            data = remote.read(address, 32)
            jump = entry_jump(data, address, remote.read)
            if jump:
                jump['target_owner'] = owner(jump['target'])
                jump['target'] = hex(jump['target'])
                if 'slot' in jump:
                    jump['slot'] = hex(jump['slot'])
            return dict(address=hex(address), owner=owner(address), entry_hex=data.hex(), recognized_entry_jump=jump)

        targets = [m for m in modules if Path(m['path']).resolve() == expected_exe.parent / 'amd_fidelityfx_dx12.dll']
        if len(targets) != 1:
            raise ValueError('exact FFX module not found')
        target = targets[0]
        path = Path(target['path'])
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest().upper()
        if digest != proof['before']['amd_fidelityfx_dx12.dll']:
            raise ValueError('FFX disk hash changed')
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(directories=[0])
        exports = []
        for name in (b'ffxCreateContext', b'ffxDestroyContext', b'ffxDispatch', b'ffxQuery'):
            symbols = [s for s in pe.DIRECTORY_ENTRY_EXPORT.symbols if s.name == name]
            if len(symbols) != 1 or symbols[0].forwarder:
                raise ValueError('export absent/ambiguous/forwarded')
            rva = symbols[0].address
            row = inspect(target['base'] + rva)
            disk = pe.get_data(rva, 32)
            row.update(export=name.decode(), disk_entry_hex=disk.hex(),
                       raw_disk_bytes_equal=row['entry_hex'] == disk.hex(),
                       relocation_normalized=False)
            exports.append(row)
        pe.close()
        events = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        # These lists are held by the bounded session; no broad heap scanning.
        candidates = [e for e in events if e.get('event') == 'candidate']
        route = None
        if provider_route:
            if not candidates:
                raise ValueError('no context candidate')
            pointers = provider_route_snapshot(remote.read, target['base'], int(candidates[-1]['context'], 16), digest)
            metadata = provider_metadata_snapshot(remote.read, target['base'], pointers)
            control_matches = []
            for name in ('fsr3', 'fsr4'):
                manifest_path = Path(proof['validation_run']) / name / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                if manifest.get('requested_provider_id') == metadata['version_id'] and manifest.get('requested_provider') == metadata['version_name']:
                    control_matches.append(str(manifest_path))
            route = dict(pointers={name: hex(value) if value else None for name, value in pointers.items()},
                         provider_metadata=metadata, matching_independent_controls=control_matches,
                         provider_object_owner=owner(pointers['provider_object']),
                         provider_vtable_owner=owner(pointers['provider_vtable']),
                         dispatch_entry=inspect(pointers['provider_dispatch']),
                         callback_entry=inspect(pointers['external_callback']) if pointers['external_callback'] else None,
                         scope='Exact-image private layout, double-read pointer snapshot; no execution/lifetime proof',
                         selected_public_provider_id_verified=False,
                         private_metadata_matches_independent_provider=len(control_matches) == 1)
        addresses = list(dict.fromkeys(e['list'] for e in reversed(candidates)))[:2]
        lists = []
        for address in addresses:
            table = struct.unpack('<Q', remote.read(int(address, 16), 8))[0]
            entries = {}
            for index, name in ((9, 'Close'), (10, 'Reset'), (14, 'Dispatch'), (26, 'ResourceBarrier')):
                pointer = struct.unpack('<Q', remote.read(table + index * 8, 8))[0]
                entries[name] = inspect(pointer)
            lists.append(dict(list=address, vtable=hex(table), vtable_owner=owner(table), entries=entries))
        if not process.is_running() or abs(remote.created() - expected_start) > 0.002:
            raise ValueError('process exited/changed during audit')
        return dict(status='READ_ONLY_ENTRY_EVIDENCE', pid=proof['pid'], process_start=proof['process_start'],
                    ffx_sha256=digest, exports=exports, provider_route=route, command_lists=lists, bytes_read=remote.bytes_read,
                    process_access='QUERY_INFORMATION | VM_READ', process_writes=0,
                    limitations=['Snapshot only; not a complete executed call graph',
                                 'Only three entry-jump encodings recognized',
                                 'Raw disk comparison is not relocation-normalized',
                                 'Alternate COM interfaces/enhanced barriers not audited'],
                    capture_authorized=False, game_runtime_ready=False)
    finally:
        remote.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provider-route', action='store_true', help='Audit exact-image internal context forwarding pointers (read-only)')
    args = parser.parse_args()
    if os.name != 'nt' or ct.sizeof(ct.c_void_p) != 8:
        raise SystemExit('Requires 64-bit Windows Python')
    if args.output.exists():
        raise SystemExit('Refusing to overwrite report')
    report = audit(args.provenance, args.log, args.provider_route)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))
