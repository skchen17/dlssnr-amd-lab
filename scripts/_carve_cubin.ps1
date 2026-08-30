# Carve the first two CUBIN ELFs out of nvngx_dlssnr.dll (metadata work: extracted
# copies live in results/, never committed) and dump their headers with llvm-readobj.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repo 'results\20260830_170305'
$src = 'C:\DATA\Python_File\Qoder_workspace\file\nvngx_dlssnr.dll'

# offsets from binary_manifest_dlssnr.json (bytes)
$offsets = @(2069024, 5690584)
# ELF total size: e_shoff + e_shnum*e_shentsize; read generously then trim via readobj later.
$buf = [byte[]]::new(24 * 1024 * 1024)
$fs = [System.IO.File]::OpenRead($src)
for ($i = 0; $i -lt $offsets.Count; $i++) {
    $off = $offsets[$i]
    $fs.Seek($off, 'Begin') | Out-Null
    $n = $fs.Read($buf, 0, $buf.Length)
    # parse minimal ELF64 header to get real size (max of shoff-table end and phoff-table end)
    $ehsize   = [BitConverter]::ToUInt16($buf, 52)
    $phentsz  = [BitConverter]::ToUInt16($buf, 54)
    $phnum    = [BitConverter]::ToUInt16($buf, 56)
    $shentsz  = [BitConverter]::ToUInt16($buf, 58)
    $shnum    = [BitConverter]::ToUInt16($buf, 60)
    $phoff    = [BitConverter]::ToUInt64($buf, 32)
    $shoff    = [BitConverter]::ToUInt64($buf, 40)
    $endSh = $shoff + [uint64]$shnum * [uint64]$shentsz
    $endPh = $phoff + [uint64]$phnum * [uint64]$phentsz
    # section extents: walk section headers for max(sh_offset+sh_size)
    $maxEnd = [uint64]([Math]::Max($endSh, $endPh))
    for ($s = 0; $s -lt $shnum; $s++) {
        $base = [int]($shoff + [uint64]$s * [uint64]$shentsz)
        if ($base + 64 -gt $n) { break }
        $soff = [BitConverter]::ToUInt64($buf, $base + 24)
        $ssz  = [BitConverter]::ToUInt64($buf, $base + 32)
        $shType = [BitConverter]::ToUInt32($buf, $base + 4)
        if ($shType -ne 8) { $e2 = $soff + $ssz; if ($e2 -gt $maxEnd) { $maxEnd = $e2 } }
    }
    $total = [int]$maxEnd
    Write-Host ("module[{0}] @0x{1:x}: shoff=0x{2:x} shnum={3} phoff=0x{4:x} phnum={5} total={6} bytes" -f $i, $off, $shoff, $shnum, $phoff, $phnum, $total)
    $dst = Join-Path $runDir ("carved_cubin_$i.elf")
    [System.IO.File]::WriteAllBytes($dst, $buf[0..($total - 1)])
    Write-Host "  carved -> carved_cubin_$i.elf"
}
$fs.Close()
Write-Host 'done'
