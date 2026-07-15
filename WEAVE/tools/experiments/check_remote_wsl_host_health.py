from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


DEFAULT_HOST = "100.115.18.62"
DEFAULT_PORT = 2222
DEFAULT_USER = "administrator"
DEFAULT_WSL_DISTRO = "Ubuntu-26.04"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _ssh(host: str, port: int, user: str, remote_command: str) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-p",
            str(port),
            "-T",
            "-o",
            "LogLevel=ERROR",
            f"{user}@{host}",
            remote_command,
        ]
    )


def _normalize(text: str) -> str:
    return str(text or "").replace("\x00", "").strip()


def _looks_like_hcs_failure(text: str) -> bool:
    upper = _normalize(text).upper()
    return "HCS_E_SERVICE_NOT_AVAILABLE" in upper or "WSL/SERVICE/CREATEINSTANCE/CREATEVM/HCS/" in upper


def _parse_service_state(text: str) -> str:
    for line in _normalize(text).splitlines():
        if "STATE" in line:
            parts = line.split(":")[-1].strip().split()
            if len(parts) >= 2:
                return parts[1]
    return ""


def _feature_state_map(text: str) -> dict[str, str]:
    states: dict[str, str] = {}
    parsed = _try_json(text)
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            states[str(key)] = str(value)
        return states
    for raw in _normalize(text).splitlines():
        line = raw.strip()
        if "|" not in line:
            continue
        left, right = line.split("|", 1)
        name = left.strip()
        state = right.strip()
        if name:
            states[name] = state
    return states


def _try_json(text: str) -> Any:
    raw = _normalize(text)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_hypervisorlaunchtype(text: str) -> str:
    for line in _normalize(text).splitlines():
        if "hypervisorlaunchtype" in line.lower():
            return line.split()[-1]
    return ""


def _parse_wsl_list(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _normalize(text).splitlines():
        line = raw.replace("\u0000", "").strip()
        if not line or "NAME" in line and "STATE" in line and "VERSION" in line:
            continue
        active = line.startswith("*")
        line = line.lstrip("*").strip()
        parts = [part for part in line.split() if part]
        if len(parts) < 3:
            continue
        rows.append(
            {
                "active": active,
                "name": " ".join(parts[:-2]),
                "state": parts[-2],
                "version": parts[-1],
            }
        )
    return rows


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Check remote Windows/WSL2 host health for 3060 experiment relaunches.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    args = parser.parse_args()

    ping = _ssh(args.host, args.port, args.user, "echo SSH_OK")
    lxss = _ssh(args.host, args.port, args.user, "sc query LxssManager")
    hyper = _ssh(args.host, args.port, args.user, "bcdedit /enum {current}")
    features = _ssh(
        args.host,
        args.port,
        args.user,
        (
            "powershell -NoProfile -Command "
            "\"$out=[ordered]@{"
            "'Microsoft-Windows-Subsystem-Linux'=(Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux).State.ToString();"
            "'VirtualMachinePlatform'=(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State.ToString();"
            "'Microsoft-Hyper-V-Hypervisor'=(Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-Hypervisor).State.ToString()"
            "}; $out | ConvertTo-Json -Compress\""
        ),
    )
    wsl_list = _ssh(args.host, args.port, args.user, "wsl -l -v")
    wsl_exec = _ssh(
        args.host,
        args.port,
        args.user,
        f"wsl -d {args.wsl_distro} --exec /bin/echo WSL_OK",
    )

    feature_states = _feature_state_map(features.stdout)
    lxss_state = _parse_service_state(lxss.stdout)
    hypervisorlaunchtype = _parse_hypervisorlaunchtype(hyper.stdout)
    wsl_rows = _parse_wsl_list(wsl_list.stdout)
    hcs_failure = _looks_like_hcs_failure(wsl_exec.stdout)

    vmp_state = feature_states.get("VirtualMachinePlatform", "")
    wsl_state = feature_states.get("Microsoft-Windows-Subsystem-Linux", "")
    hyperv_hv_state = feature_states.get("Microsoft-Hyper-V-Hypervisor", "")

    def _state_enabled(value: str) -> bool:
        text = str(value or "").strip().lower()
        return text in {"enabled", "enablepending"} or ("启用" in text and "禁" not in text)

    reboot_required = (
        _state_enabled(vmp_state)
        and hypervisorlaunchtype.lower() == "auto"
        and hcs_failure
    )

    report = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "wsl_distro": args.wsl_distro,
        "ssh_ok": ping.returncode == 0 and "SSH_OK" in ping.stdout,
        "lxssmanager_state": lxss_state,
        "hypervisorlaunchtype": hypervisorlaunchtype,
        "feature_states": {
            "Microsoft-Windows-Subsystem-Linux": wsl_state,
            "VirtualMachinePlatform": vmp_state,
            "Microsoft-Hyper-V-Hypervisor": hyperv_hv_state,
        },
        "wsl_distros": wsl_rows,
        "wsl_exec_ok": wsl_exec.returncode == 0 and "WSL_OK" in wsl_exec.stdout,
        "remote_wsl_hcs_failure": hcs_failure,
        "reboot_required_for_wsl2": reboot_required,
        "raw": {
            "lxssmanager": _normalize(lxss.stdout),
            "hypervisor": _normalize(hyper.stdout),
            "features": _normalize(features.stdout),
            "wsl_list": _normalize(wsl_list.stdout),
            "wsl_exec": _normalize(wsl_exec.stdout),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
