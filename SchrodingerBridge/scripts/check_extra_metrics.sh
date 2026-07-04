#!/bin/bash
echo "=== Process check ==="
ps -ef | grep batch_compute | grep -v grep
echo ""
echo "=== Full log ==="
cat /mnt/i/exp_extra_metrics.log 2>/dev/null
echo ""
echo "=== Results so far ==="
cat /mnt/i/exp_extra_metrics_results.json 2>/dev/null
