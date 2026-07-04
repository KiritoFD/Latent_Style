#!/bin/bash
F="/mnt/c/Users/Administrator/v4_final.txt"
{
  echo "V4_FINAL $(date)"
  tmux list-sessions 2>&1 || echo "TMUX_DEAD"
  echo "=== CKPTS ==="
  for d in fc_sb_kernel7 fc_sb_floor0 fc_sb_curriculum fc_sb_fiber_ep fc_sb_wavelet; do
    echo -n "$d: "
    ls /home/xy/Latent_Style/SchrodingerBridge/$d/checkpoints/*.pt 2>/dev/null | xargs -I{} basename {} 2>/dev/null | tr '\n' ' '
    echo ""
  done
  echo "=== LOG LAST 40 ==="
  tail -40 /home/xy/Latent_Style/SchrodingerBridge/exp/p3_remote_10h/v4_train.log 2>/dev/null || echo "NO_LOG"
  echo "=== PYTHON ==="
  ps aux | grep "run.py" | grep -v grep | head -3 || echo "NO_RUN_PY"
} > "$F" 2>&1
