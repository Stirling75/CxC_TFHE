"""Process isolation, source identity, and portable run records."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from cases import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record():
    paths = []
    for directory in ("code", "scripts", "config"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                     and not {"target", "__pycache__", ".git"}.intersection(p.parts)
                     and p.name != ".DS_Store")
    for directory in ("vendor", ".cargo"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def source_id(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def serializable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def available_cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))


def host():
    record = {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "available_cpu_ids": available_cpus(),
            "cpu_affinity_supported": hasattr(os, "sched_setaffinity")}
    if sys.platform == "linux":
        record["cpu_topology"] = []
        for cpu in available_cpus():
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            try:
                record["cpu_topology"].append({"cpu": cpu,
                    "package": int((path / "physical_package_id").read_text()),
                    "core": int((path / "core_id").read_text())})
            except (OSError, ValueError):
                record["cpu_topology"].append({"cpu": cpu, "package": None, "core": None})
        for name in ("cpuinfo", "meminfo"):
            try:
                record[name] = Path("/proc", name).read_text()
            except OSError:
                record[name] = None
    return record


def environment(overrides=None):
    prefixes = ("CBS_", "DIRECT_", "PBS_", "KS_", "CMUX_", "CACHED_MULT_", "FUSED_", "RAYON_", "CARGO_TARGET_DIR")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env.update(overrides or {})
    return env


def execute(command, cwd, env, logfile, cpus=None):
    if cpus is not None:
        command = [sys.executable, str(ROOT / "scripts/runtime.py"), ",".join(map(str, cpus)), *command]
    start = time.monotonic()
    with logfile.open("w") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        except BaseException:
            import signal
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            raise
    return {"exit_code": child.returncode, "process_elapsed_seconds": time.monotonic() - start,
            "peak_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
            "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime}


if __name__ == "__main__":
    cpus = set(map(int, sys.argv[1].split(",")))
    os.sched_setaffinity(0, cpus)
    if os.sched_getaffinity(0) != cpus:
        raise RuntimeError("CPU affinity was not applied")
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
