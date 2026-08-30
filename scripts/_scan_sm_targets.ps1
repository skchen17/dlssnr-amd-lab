# Scan both DLLs for ASCII sm_* target markers (PTX '.target sm_' or fatbin arch strings).
$ErrorActionPreference = 'Continue'
$files = @(
    'C:\DATA\Python_File\Qoder_workspace\file\nvngx_dlss.dll',
    'C:\DATA\Python_File\Qoder_workspace\file\nvngx_dlssnr.dll'
)
$rx = [regex]'sm_[0-9]{2}[a-z0-9]*'
$chunk = 64MB
foreach ($f in $files) {
    Write-Host "== $f =="
    $fs = [System.IO.File]::OpenRead($f)
    $buf = [byte[]]::new($chunk + 16)
    $counts = @{}
    $examples = @{}
    $carry = [byte[]]::new(16)
    $pos = [long]0
    while ($true) {
        [Array]::Copy($carry, 0, $buf, 0, 16)
        $n = $fs.Read($buf, 16, $chunk)
        if ($n -le 0) { break }
        $s = [System.Text.Encoding]::ASCII.GetString($buf, 0, $n + 16)
        foreach ($m in $rx.Matches($s)) {
            $v = $m.Value
            if (-not $counts.ContainsKey($v)) { $counts[$v] = 0; $examples[$v] = [long]($pos + $m.Index) }
            $counts[$v]++
        }
        [Array]::Copy($buf, $n, $carry, 0, 16)
        $pos += $n
    }
    $fs.Close()
    if ($counts.Count -eq 0) { Write-Host '  (no sm_* markers found)' }
    foreach ($k in ($counts.Keys | Sort-Object)) {
        Write-Host ("  {0}: count={1} first@0x{2:x}" -f $k, $counts[$k], $examples[$k])
    }
}
Write-Host 'done'
