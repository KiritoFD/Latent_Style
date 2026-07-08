$ErrorActionPreference = 'Continue'

# Local results directory
$local_root = "g:\GitHub\Latent_Style\SchrodingerBridge\results"

# Create directory structure
$datasets = @("D5-512", "P256", "R5-WikiArt")
$methods = @("identity", "adain", "wct", "sdturbo", "cut", "samst", "samam", "seedream", "weave")

foreach ($ds in $datasets) {
    foreach ($m in $methods) {
        $dir = Join-Path $local_root "$ds\$m"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

Write-Host "Created directory structure at $local_root"

# Check local disk space
$drive = (Get-PSDrive -Name "g")
$freeGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "Local G: drive free space: ${freeGB} GB"

# Remote image directory mapping
# Format: @{ local_method = "remote_path" }
$remote_map = @{
    "D5-512" = @{
        "identity" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\identity"
        "adain" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\adain"
        "wct" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\wct_v32k"
        "sdturbo" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\sdturbo"
        "cut" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\cut"
        "samst" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\samst"
        "samam" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\images\samam"
        "seedream" = "I:\Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\images"
        "weave" = "I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\images"
    }
    "P256" = @{
        "identity" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256\identity\images"
        "adain" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256\adain\images"
        "wct" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts15_256\wct\images"
    }
    "R5-WikiArt" = @{
        "identity" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\identity\images"
        "adain" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\adain\images"
        "wct" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\wct\images"
        "sdturbo" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\sdturbo\images"
        "samst" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samst\images"
        "samam" = "I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20\samam\images"
        "weave" = "I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval\images"
    }
}

# Save the mapping for use by the transfer script
$map_path = "$local_root\_remote_map.json"
$remote_map | ConvertTo-Json -Depth 3 | Out-File -FilePath $map_path -Encoding UTF8
Write-Host "Remote mapping saved to $map_path"

# Print summary
Write-Host "`n=== Transfer Plan ==="
foreach ($ds in $datasets) {
    Write-Host "`n[$ds]"
    foreach ($m in $methods) {
        $remote = $remote_map[$ds][$m]
        if ($remote) {
            Write-Host "  $m -> $remote"
        } else {
            Write-Host "  $m -> (not available)"
        }
    }
}

# Also check remote disk space
Write-Host "`n=== Remote I: drive space ==="
# This will be done via SSH in the transfer script
