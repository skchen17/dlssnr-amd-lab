function Get-NativePreviewSourceHashes([string]$Repository) {
    $files=@(Get-ChildItem -LiteralPath (Join-Path $Repository 'scripts') -Filter 'native_*.py' -File)
    $files+=Get-Item -LiteralPath (Join-Path $Repository 'tools/native_reshade/NativePreview.fx')
    @($files | Sort-Object FullName | Get-FileHash -Algorithm SHA256)
}

function Assert-NativePreviewSources($Expected,$Actual) {
    if(-not $Expected -or -not $Actual){throw 'Native source proof missing; fresh standalone proof required'}
    $expectedPaths=@($Expected | ForEach-Object Path)
    $actualPaths=@($Actual | ForEach-Object Path)
    if(($expectedPaths | Sort-Object -Unique).Count -ne $expectedPaths.Count -or
       @(Compare-Object $expectedPaths $actualPaths).Count){throw 'Native source proof coverage differs'}
    foreach($entry in $Actual){
        $old=@($Expected | Where-Object Path -eq $entry.Path)
        if($old.Count -ne 1 -or $old[0].Hash -ne $entry.Hash){throw 'Native source changed; fresh standalone proof required'}
    }
}
