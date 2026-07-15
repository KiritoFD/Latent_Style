#!/usr/bin/env bash
# Probe what is holding the GPU and why python processes keep respawning.
echo "===== nvidia-smi compute apps ====="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || nvidia-smi

echo
echo "===== python processes (top 20 by etime) ====="
ps -eo pid,ppid,user,etime,cmd --sort=-etime 2>/dev/null | head -1
ps -eo pid,ppid,user,etime,cmd --sort=-etime 2>/dev/null | grep -i python | grep -v grep | head -20

echo
echo "===== parent processes of python procs ====="
for pid in $(pgrep -f python 2>/dev/null | head -10); do
  echo "--- pid=$pid ---"
  ps -o pid,ppid,user,etime,cmd -p "$pid" 2>/dev/null
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ -n "$ppid" ] && [ "$ppid" != "1" ]; then
    echo "  parent (ppid=$ppid):"
    ps -o pid,ppid,user,etime,cmd -p "$ppid" 2>/dev/null
  fi
done

echo
echo "===== tmux sessions ====="
tmux ls 2>/dev/null || echo "(no tmux server)"

echo
echo "===== screen sessions ====="
screen -ls 2>/dev/null || echo "(no screen)"

echo
echo "===== systemd user services (running) ====="
systemctl --user list-units --type=service --state=running 2>/dev/null | head -30 || echo "(no systemd user)"

echo
echo "===== cron jobs (user xy) ====="
crontab -l 2>/dev/null || echo "(no user crontab)"

echo
echo "===== /etc/cron.* ====="
ls -la /etc/cron.d/ /etc/cron.daily/ /etc/cron.hourly/ 2>/dev/null

echo
echo "===== keepalive / samam-related files in /tmp and home ====="
ls -la /tmp/*keepalive* /tmp/*samam* /home/xy/*keepalive* /home/xy/*samam* 2>/dev/null | head -20

echo
echo "===== .bashrc / .profile tail (auto-start lines) ====="
grep -nE 'samam|keepalive|train_SaMam|tmux|screen|nohup|setsid' /home/xy/.bashrc /home/xy/.profile /home/xy/.bash_profile 2>/dev/null | head -30

echo
echo "===== recently modified scripts in SchrodingerBridge/tools ====="
ls -lt /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/*.sh 2>/dev/null | head -10

echo
echo "===== any train_SaMam or eval process tree ====="
pstree -ap $(pgrep -f train_SaMam 2>/dev/null | head -1) 2>/dev/null | head -20 || echo "(no train_SaMam process)"
