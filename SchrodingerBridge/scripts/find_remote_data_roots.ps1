param(
    [string[]]$Roots = @("I:\", "F:\", "G:\")
)

$ErrorActionPreference = "Continue"
$patterns = @("*wikiart*", "*latent*", "*distinct*", "*samam*")

foreach ($root in $Roots) {
    if (-not (Test-Path -LiteralPath $root)) {
        continue
    }
    "== $root =="
    foreach ($pat in $patterns) {
        Get-ChildItem -LiteralPath $root -Directory -Recurse -Filter $pat -ErrorAction SilentlyContinue |
            Select-Object -First 80 FullName |
            ForEach-Object { $_.FullName }
    }
}
