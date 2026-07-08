$ErrorActionPreference = 'SilentlyContinue'
function SizeGB($p){ $s=(Get-ChildItem $p -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum; if($s){'{0:N2}GB' -f ($s/1GB)}else{'?'} }
Write-Host '=== I:\ root ==='
Get-ChildItem I:\ | ForEach-Object {
  if($_.PSIsContainer){ '{0,-32} DIR  {1}' -f $_.Name,(SizeGB $_.FullName) }
  else { '{0,-32} FILE {0:N1}MB' -f $_.Name,($_.Length/1MB) }
}
Write-Host '=== I:\datasets (current target) ==='
Get-ChildItem I:\datasets | ForEach-Object {
  if($_.PSIsContainer){ '{0,-32} DIR  {1}' -f $_.Name,(SizeGB $_.FullName) }
  else { '{0,-32} FILE {0:N1}MB' -f $_.Name,($_.Length/1MB) }
}
Write-Host '=== candidate scattered dataset dirs on I:\ ==='
foreach($cand in @('exp_*','latent_style_remote_*','Github\Latent_Style\latent-*','Github\Latent_Style\style_data','Github\Latent_Style\clip-feats-vitb32','Github\Latent_Style\latent_cyclegan','Github\Latent_Style\vavae_exp','Github\Latent_Style\Cycle-NCE','wikiart*','latent*','style_data')){
  Get-ChildItem I:\ -Directory -Filter $cand -ErrorAction SilentlyContinue | ForEach-Object {
    '{0,-55} {1}' -f $_.FullName,(SizeGB $_.FullName)
  }
}
Write-Host '=== nested datasets inside I:\Github\Latent_Style (depth2) ==='
Get-ChildItem I:\Github\Latent_Style -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $d=$_.FullName
  Get-ChildItem $d -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'latent|style_data|wikiart|clip|cycle|vavae|datasets' } | ForEach-Object {
    '{0,-60} {1}' -f $_.FullName,(SizeGB $_.FullName)
  }
}
