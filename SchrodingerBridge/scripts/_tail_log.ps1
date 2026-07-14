param(
    [string]$Path = "C:\Users\Administrator\logs\sty_inject\nonzero_only_train.out",
    [int]$Tail = 30
)
if (Test-Path $Path) {
    Get-Content $Path -Tail $Tail
} else {
    Write-Output "FILE NOT FOUND: $Path"
}
