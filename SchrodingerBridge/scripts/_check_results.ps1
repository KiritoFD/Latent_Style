$root = "g:\GitHub\Latent_Style\SchrodingerBridge\results"
if (Test-Path $root) {
    Write-Host "=== Existing local results ==="
    Get-ChildItem $root -Directory | ForEach-Object {
        $ds = $_.Name
        Write-Host "[$ds]"
        Get-ChildItem $_.FullName -Directory | ForEach-Object {
            $cnt = (Get-ChildItem $_.FullName -Filter *.png -ErrorAction SilentlyContinue).Count
            if ($cnt -gt 0) {
                Write-Host ("  " + $_.Name + ": " + $cnt + " png")
            }
        }
    }
} else {
    Write-Host "results dir not found"
}
