# Read e_flags from the carved CUBINs to identify the target SM architecture.
$runDir = 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\20260830_170305'
foreach ($i in 0, 1) {
    $p = Join-Path $runDir ("carved_cubin_$i.elf")
    $b = [System.IO.File]::ReadAllBytes($p)[0..63]
    $eType    = [BitConverter]::ToUInt16($b, 16)
    $eMachine = [BitConverter]::ToUInt16($b, 18)
    $eFlags   = [BitConverter]::ToUInt32($b, 48)
    # ELF64 CUDA: arch = e_flags & 0xFF (historically), or per modern cubin: bits encode sm target
    $archBits = $eFlags -band 0xFF
    Write-Host ("cubin_{0}: e_machine=0x{1:x} e_type=0x{2:x} e_flags=0x{3:x8} archBits=0x{4:x2} ({5})" -f $i, $eMachine, $eType, $eFlags, $archBits, $archBits)
}
