$ErrorActionPreference='SilentlyContinue'
"=== REMOTE DRIVES ==="
Get-PSDrive -PSProvider FileSystem | ForEach-Object { "$($_.Name): root=$($_.Root)" }
"=== CANDIDATE DATA DIRS (depth<=2, name match) ==="
$roots = (Get-PSDrive -PSProvider FileSystem).Root
foreach ($r in $roots) {
  Get-ChildItem -Path $r -Depth 2 -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'Dataset|dataset|wikiart|distinct|fewshot|latent|style_data|feats|vavae|Schrodinger' } | ForEach-Object { $_.FullName }
}
