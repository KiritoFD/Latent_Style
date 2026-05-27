$ErrorActionPreference='Continue'
$repo='I:\Github\Latent_Style\SchrodingerBridge'
$exp=Join-Path $repo 'exp'
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$archive=Join-Path $repo "archives\exp_archive_$stamp"
$manifestDir=Join-Path $exp '_cleanup_manifests'
New-Item -ItemType Directory -Force -Path $archive,$manifestDir | Out-Null
$moves=@()
function TargetFor([string]$name){
  if($name -eq 'diffeomorphic_tangent_sweep'){ return $null }
  if($name -like 'vae_backend_256*' -or $name -eq 'vae_scale_decode_sweep'){ return "vae_backend\$name" }
  if($name -like 'inference_*'){ return "inference\$name" }
  if($name -in @('frontier_decision_tree_8h','orthogonal_budget36','stagewise_meeting','dynamic_metric_probe','t01_large_patch_probe','t01_patch1_probe','t01_patch36')){ return "frontier\$name" }
  if($name -in @('forward_diagnostics','layer_diagnostics')){ return "diagnostics\$name" }
  return "ARCHIVE:$name"
}
$rootDirs=Get-ChildItem -LiteralPath $exp -Directory | Where-Object { $_.Name -notin @('_cleanup_manifests','vae_backend','paper','video','inference','frontier','diagnostics') }
foreach($d in $rootDirs){
  $target=TargetFor $d.Name
  if($null -eq $target){ continue }
  if($target.StartsWith('ARCHIVE:')){ $dest=Join-Path $archive ("remote_exp_legacy\" + $d.Name) }
  else { $dest=Join-Path $exp $target }
  if(Test-Path -LiteralPath $dest){ $dest = "$dest`__moved_$stamp" }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
  try {
    Move-Item -LiteralPath $d.FullName -Destination $dest -Force -ErrorAction Stop
    $moves += [pscustomobject]@{source=$d.FullName; destination=$dest; name=$d.Name; category= if($target.StartsWith('ARCHIVE:')){'archive'}else{($target -split '\\')[0]}; moved=$true; error=$null}
  } catch {
    $moves += [pscustomobject]@{source=$d.FullName; destination=$dest; name=$d.Name; category= if($target.StartsWith('ARCHIVE:')){'archive'}else{($target -split '\\')[0]}; moved=$false; error=$_.Exception.Message}
  }
}
$rootFiles=Get-ChildItem -LiteralPath $exp -File
foreach($f in $rootFiles){
  $dest=Join-Path $archive ("remote_exp_root_files\" + $f.Name)
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
  try {
    Move-Item -LiteralPath $f.FullName -Destination $dest -Force -ErrorAction Stop
    $moves += [pscustomobject]@{source=$f.FullName; destination=$dest; name=$f.Name; category='root_files'; moved=$true; error=$null}
  } catch {
    $moves += [pscustomobject]@{source=$f.FullName; destination=$dest; name=$f.Name; category='root_files'; moved=$false; error=$_.Exception.Message}
  }
}
$manifest=[pscustomobject]@{timestamp=$stamp; repo=$repo; exp=$exp; archive=$archive; moves=$moves}
$manifestPath=Join-Path $manifestDir "remote_exp_reorg_finish_$stamp.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8
$readme=Join-Path $exp 'README.md'
@(
'# exp layout',
'',
"Reorganized on $stamp. Move manifest: $manifestPath.",
'',
'- `vae_backend/`: active VAE backend runs and status outputs.',
'- `inference/`: inference parameter sweeps.',
'- `frontier/`: frontier/patch/stagewise sweeps.',
'- `diagnostics/`: diagnostic probes.',
'- `diffeomorphic_tangent_sweep/`: kept at top level because scripts use it as the t01 base config.',
"- archived legacy clutter: $archive."
) | Set-Content -Path $readme -Encoding UTF8
[pscustomobject]@{manifest=$manifestPath; moved=($moves | Where-Object moved).Count; failed=($moves | Where-Object { -not $_.moved }).Count; top=(Get-ChildItem -LiteralPath $exp -Directory | Select-Object -ExpandProperty Name)} | ConvertTo-Json -Compress
