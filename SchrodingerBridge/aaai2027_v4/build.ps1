# Build the AAAI 2027 paper (paper.pdf) from source, self-contained in this folder.
# Usage:  .\build.ps1   (run from this directory, or it cd's automatically)
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $dir
try {
    Write-Host "=== pdflatex (1/3) ==="
    pdflatex -interaction=nonstopmode paper.tex | Out-Null
    Write-Host "=== bibtex ==="
    bibtex paper
    Write-Host "=== pdflatex (2/3) ==="
    pdflatex -interaction=nonstopmode paper.tex | Out-Null
    Write-Host "=== pdflatex (3/3) ==="
    pdflatex -interaction=nonstopmode paper.tex | Out-Null
    if (Test-Path paper.pdf) {
        Write-Host "Build OK -> paper.pdf ($([math]::Round((Get-Item paper.pdf).Length/1KB,1)) KB)"
    } else {
        Write-Error "paper.pdf was not produced."
    }
} finally {
    Pop-Location
}
