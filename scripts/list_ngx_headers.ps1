$inc = 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\third_party\nvidia-dlss\include'
Get-ChildItem $inc | ForEach-Object {
    $lines = (Get-Content $_.FullName | Measure-Object -Line).Lines
    Write-Host ('{0,-42} {1,8} B  {2} lines' -f $_.Name, $_.Length, $lines)
}
Write-Host '--- lib tree:'
Get-ChildItem 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\third_party\nvidia-dlss\lib' -Recurse -File |
    Select-Object -First 30 | ForEach-Object { Write-Host ('{0}  {1} B' -f $_.FullName, $_.Length) }
Write-Host '--- LICENSE head:'
Get-Content 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\third_party\nvidia-dlss\LICENSE.txt' -TotalCount 12
