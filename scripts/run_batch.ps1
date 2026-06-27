$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$tmp = "outputs\tmp_batch"
New-Item -ItemType Directory -Path $tmp -Force > $null
Write-Host "=== Running 314 queries ==="

for ($i = 0; $i -le 313; $i++) {
  $qid = "q_{0:D3}" -f $i
  $d = "$tmp\$qid"
  New-Item -ItemType Directory -Path $d -Force > $null
  Write-Host $qid
  python -m query_generation_loop.pipeline --query_id $qid --mode methodology_baseline --output "$d\result.json" --claims_final_output "$d\claims.json" 2>&1 | Out-Null
}

Write-Host "=== Aggregating ==="
python scripts/aggregate_results.py

Remove-Item -Recurse -Force $tmp
Write-Host "=== Done -> outputs/batch_summary.json ==="
