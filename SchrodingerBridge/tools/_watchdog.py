"""Watchdog that auto-chains the 3 baseline schtasks on the remote 3060.

Runs every 15 min via schtasks (Watchdog_Baselines). Logic:
  - While StyleAligned_Runs is running  -> wait.
  - Else if Z-STAR not yet launched     -> launch ZSTAR_Runs (flag .zstar_launched).
  - Else if Z-STAR finished (produced >0 outputs) and StyleShot not launched
                                         -> launch StyleShot_Runs (flag .styleshot_launched).
  - Else                                 -> all done.

This survives SSH disconnect because it is a system scheduled task.
"""
import subprocess, os, time
from pathlib import Path

EXP = Path("I:/GitHub/Latent_Style/SchrodingerBridge/exp")
SA, ZS, SS = "StyleAligned_Runs", "ZSTAR_Runs", "StyleShot_Runs"


def task_state(name):
    # schtasks /FO LIST uses the "模式" (mode) field: 正在运行 / 就绪
    out = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "LIST"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("模式") or line.lstrip().startswith("模式"):
            return line.split(":", 1)[-1].strip()
    return "UNKNOWN"


def run_task(name):
    subprocess.run(["schtasks", "/Run", "/TN", name], capture_output=True, text=True)


def flag(p):
    return Path(p).exists()


def write_flag(p):
    Path(p).write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def count_images(sub):
    d = EXP / sub / "images"
    return len([f for f in os.listdir(d) if f.endswith(".png")]) if d.is_dir() else 0


def main():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    sa = task_state(SA)
    lines = [f"[{now}] SA={sa}"]
    if "运行" in sa:
        lines.append("  StyleAligned still running -> wait.")
    elif not flag(EXP / ".zstar_launched"):
        lines.append("  >> Launching Z-STAR")
        run_task(ZS)
        write_flag(EXP / ".zstar_launched")
    else:
        zs = task_state(ZS)
        lines.append(f"  ZS={zs}")
        if "运行" in zs:
            lines.append("  Z-STAR running -> wait.")
        elif not flag(EXP / ".styleshot_launched"):
            zc = count_images("baseline_zstar/D5") + count_images("baseline_zstar/P2A") + count_images("baseline_zstar/R5")
            lines.append(f"  Z-STAR outputs={zc}")
            if zc > 0:
                lines.append("  >> Launching StyleShot")
                run_task(SS)
                write_flag(EXP / ".styleshot_launched")
            else:
                lines.append("  Z-STAR produced 0 outputs -> NOT launching StyleShot (check zstar.log)")
        else:
            ss = task_state(SS)
            lines.append(f"  SS={ss}")
            if "运行" not in ss:
                lines.append("  ALL DONE.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
