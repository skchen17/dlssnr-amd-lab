param(
    [Parameter(Mandatory=$true)][string]$FreshOutputDirectory,
    [string]$Rdp = '',
    [string]$QuantDll = 'results\20260906_head_qkv_build_v1\native_fusion_quantize.dll',
    [string]$MatrixDll = 'results\20260906_c64_c128_attention_wmma_build_v1\head_wmma.dll'
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$root=[IO.Path]::GetFullPath($FreshOutputDirectory)
if(Test-Path -LiteralPath $root){throw 'Fresh output directory already exists; refusing overwrite'}
New-Item -ItemType Directory -Path $root | Out-Null
if(-not $Rdp){$Rdp=Join-Path $repo '.tools\radeon-developer-suite\RadeonDeveloperToolSuite-2026-05-28-1806\RadeonDeveloperPanelCLI.exe'}
$rdf=Join-Path $repo 'results\20260906_rgp_rdf_dump_build_v7\rgp_rdf_dump.exe'
foreach($path in @($Rdp,$rdf,(Join-Path $repo $QuantDll),(Join-Path $repo $MatrixDll))){
    if(-not(Test-Path -LiteralPath $path)){throw "Missing profiler input: $path"}
}
$names=@('input_pack','ffn','qkv_project','qkv_norm','qk','softmax','pv','projection','tail')
Push-Location $repo
try {
    for($index=1;$index -le $names.Count;$index++){
        $capture=Join-Path $root ("{0:D2}_{1}" -f $index,$names[$index-1])
        & .\.venv-rocm\Scripts\python.exe scripts\capture_rgp.py --rdp $Rdp --output $capture `
            --process python.exe --dispatch-start 1 --dispatch-count 1 --ready-token RGP_DISPATCH_READY --timeout-seconds 90 -- `
            .\.venv-rocm\Scripts\python.exe scripts\profile_head_dispatch_target.py `
            --quant-dll $QuantDll --matrix-dll $MatrixDll --output "$capture\target" --stage $names[$index-1] `
            --arm-seconds 2 --post-seconds 5
        if($LASTEXITCODE -ne 0){throw "STOP GPU: RGP dispatch $index failed; no automatic retry"}
        & $rdf "$capture\capture.rgp" --spm-summary | Set-Content "$capture\spm_summary.json" -Encoding utf8
        if($LASTEXITCODE -ne 0){throw "RDF summary failed for dispatch $index"}
    }
    & .\.venv-rocm\Scripts\python.exe scripts\summarize_head_dispatch_sweep.py --root $root --output "$root\summary.json"
    if($LASTEXITCODE -ne 0){throw 'Head dispatch summary failed'}
} finally {Pop-Location}
