param(
  [string]$Root = "I:\Github",
  [string]$OutFile = "$env:TEMP\latent_style_remote_roots.txt"
)

$ErrorActionPreference = "SilentlyContinue"
if (Test-Path -LiteralPath $Root) {
  Get-ChildItem -LiteralPath $Root -Directory -Force |
    Select-Object -ExpandProperty FullName |
    Set-Content -Encoding UTF8 -Path $OutFile
} else {
  "MISSING $Root" | Set-Content -Encoding UTF8 -Path $OutFile
}
Get-Content -LiteralPath $OutFile
