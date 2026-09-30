$exps = @('r3_spectral_15ep','r4_spec_swd9_15ep','r3_llw2_10ep','r3_swd6_llw1_10ep','hp_simple_swd12_15ep','d1_gram_hf1_15ep','hp_rgb_s05','630_random20_heun_5ep')
foreach ($e in $exps) {
    $p = "G:\GitHub\Latent_Style\SchrodingerBridge\exp\$e"
    if (Test-Path $p) {
        $pts = Get-ChildItem $p -Filter '*.pt' -ErrorAction SilentlyContinue
        if ($pts) {
            Write-Output "LOCAL_HAS_CKPT: $e"
            $pts | ForEach-Object { Write-Output "  $($_.Name) ($([math]::Round($_.Length/1MB,1))MB)" }
        } else {
            Write-Output "LOCAL_NO_CKPT: $e (dir exists but no .pt)"
        }
    } else {
        Write-Output "MISSING: $e"
    }
}
