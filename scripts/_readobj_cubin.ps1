# Dump carved CUBIN headers with llvm-objdump (readobj is absent in this LLVM).
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$llvm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-objdump.exe'
$rd = 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\20260830_170305'
foreach ($i in 0, 1) {
    $elf = Join-Path $rd ("carved_cubin_$i.elf")
    $o = & $llvm -f -h --syms $elf 2>&1 | Out-String
    Set-Content (Join-Path $rd "carved_cubin_$i.readobj.log") $o
    Write-Host ("wrote carved_cubin_$i.readobj.log (" + $o.Length + " chars)")
}
