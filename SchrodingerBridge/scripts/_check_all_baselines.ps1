# Check all baseline image dirs and counts
"=== wikiarts20 baseline images ==="
$base20 = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20"
foreach ($m in @('identity','adain','wct','samam','sdturbo')) {
    $d = "$base20\$m\images"
    if (Test-Path $d) {
        $c = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        "  $m => $c png"
    } else {
        "  $m => NO DIR"
    }
}

"=== wikiarts20_eval (WEAVE) images ==="
$wdvf = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\images"
if (Test-Path $wdvf) {
    $c = (Get-ChildItem $wdvf -Filter *.png -ErrorAction SilentlyContinue).Count
    "  WEAVE wiki20 => $c png"
}

"=== 256 baseline images ==="
$base256 = "I:\exp_256_photo2art"
foreach ($m in @('adain_256','identity_256','samam_256','samst_256','wct_256')) {
    $d = "$base256\$m"
    if (Test-Path $d) {
        $c = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        "  $m => $c png"
    } else {
        "  $m => NO DIR"
    }
}

"=== 256 extra (sdturbo/styleid/cut/seedream) ==="
foreach ($m in @('sdturbo_256','styleid_256','cut_256','seedream_256')) {
    $d = "$base256\$m"
    if (Test-Path $d) {
        $c = (Get-ChildItem $d -Filter *.png -ErrorAction SilentlyContinue).Count
        "  $m => $c png"
    } else {
        "  $m => NO DIR"
    }
}

"=== distinct5_512 baseline_v2 images ==="
$b512 = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images"
if (Test-Path $b512) {
    Get-ChildItem $b512 -Directory | ForEach-Object {
        $c = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
        "  $($_.Name) => $c png"
    }
} else {
    "  baseline_v2 NO DIR"
}

"=== legacy256 test images per style ==="
$test256 = "I:\datasets\legacy256_overfit50\test"
foreach ($s in @('cezanne','Hayao','monet','photo','vangogh')) {
    $d = "$test256\$s"
    if (Test-Path $d) {
        $c = (Get-ChildItem $d -Filter *.jpg -ErrorAction SilentlyContinue).Count
        "  $s => $c jpg"
    }
}
