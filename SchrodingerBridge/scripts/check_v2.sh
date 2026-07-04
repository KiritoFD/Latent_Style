#!/bin/bash
echo "=== Process check ==="
ps -ef | grep batch_compute | grep -v grep
echo ""
echo "=== Log ==="
tail -80 /mnt/i/exp_extra_metrics_v2.log 2>/dev/null
echo ""
echo "=== Results ==="
cat /mnt/i/exp_extra_metrics_v2_results.json 2>/dev/null
