#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


def read_proc_stat(pid):
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        close = data.rfind(")")
        before = data[: close + 1]
        after = data[close + 2 :].split()
        comm = before.split("(", 1)[1][:-1]
        ppid = int(after[1])
        utime = int(after[11])
        stime = int(after[12])
        rss_pages = int(after[21])
        return {
            "pid": pid,
            "comm": comm,
            "ppid": ppid,
            "cpu_seconds": (utime + stime) / CLK_TCK,
            "rss_kb": max(0, rss_pages * PAGE_KB),
        }
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return None


def descendants(root_pid):
    children = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        stat = read_proc_stat(int(proc.name))
        if stat is None:
            continue
        children.setdefault(stat["ppid"], []).append(stat["pid"])

    found = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.append(pid)
        stack.extend(children.get(pid, []))
    return found


def proc_sample(root_pid):
    pids = descendants(root_pid)
    total_rss_kb = 0
    total_cpu = 0.0
    live_pids = []
    for pid in pids:
        stat = read_proc_stat(pid)
        if stat is None:
            continue
        live_pids.append(pid)
        total_rss_kb += stat["rss_kb"]
        total_cpu += stat["cpu_seconds"]
    return live_pids, total_rss_kb, total_cpu


def gpu_sample(root_pid):
    if not shutil_which("nvidia-smi"):
        return 0, "", []

    pids = set(descendants(root_pid))
    try:
        proc_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        gpu_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0, "", []

    uuid_to_index = {}
    gpu_rows = []
    for line in gpu_out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            uuid_to_index[parts[1]] = parts[0]
            gpu_rows.append(parts)

    total = 0
    details = []
    for line in proc_out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            used = int(parts[2])
        except ValueError:
            continue
        if pid in pids:
            total += used
            details.append(f"pid={pid}:gpu={uuid_to_index.get(parts[1], parts[1])}:mem_mb={used}")
    return total, ";".join(details), gpu_rows


def shutil_which(name):
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory) / name
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def terminate_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run a command and sample elapsed time, process RSS, and GPU memory."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--", dest="cmd_marker", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("Missing command to run")

    Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stdout).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stderr).parent.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    max_rss_kb = 0
    max_gpu_mem_mb = 0
    max_cpu_seconds = 0.0
    sample_count = 0
    first_gpu_rows = []

    with open(args.stdout, "w") as out, open(args.stderr, "w") as err:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            start_new_session=True,
            text=True,
        )

        try:
            with open(args.metrics_csv, "w", newline="") as metric_handle:
                writer = csv.writer(metric_handle)
                writer.writerow(
                    [
                        "timestamp",
                        "elapsed_sec",
                        "pid_count",
                        "rss_mb",
                        "cpu_seconds",
                        "gpu_mem_mb",
                        "gpu_details",
                    ]
                )
                while proc.poll() is None:
                    elapsed = time.time() - start_wall
                    live_pids, rss_kb, cpu_seconds = proc_sample(proc.pid)
                    gpu_mem_mb, gpu_details, gpu_rows = gpu_sample(proc.pid)
                    if gpu_rows and not first_gpu_rows:
                        first_gpu_rows = gpu_rows
                    max_rss_kb = max(max_rss_kb, rss_kb)
                    max_gpu_mem_mb = max(max_gpu_mem_mb, gpu_mem_mb)
                    max_cpu_seconds = max(max_cpu_seconds, cpu_seconds)
                    sample_count += 1
                    writer.writerow(
                        [
                            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            f"{elapsed:.3f}",
                            len(live_pids),
                            f"{rss_kb / 1024:.3f}",
                            f"{cpu_seconds:.3f}",
                            gpu_mem_mb,
                            gpu_details,
                        ]
                    )
                    metric_handle.flush()
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            terminate_process_group(proc)
            raise

        return_code = proc.wait()

    elapsed = time.time() - start_wall
    summary = {
        "name": args.name,
        "command": cmd,
        "return_code": return_code,
        "start_epoch": start_wall,
        "end_epoch": time.time(),
        "elapsed_sec": elapsed,
        "max_rss_mb": max_rss_kb / 1024,
        "max_gpu_mem_mb": max_gpu_mem_mb,
        "max_cpu_seconds": max_cpu_seconds,
        "sample_count": sample_count,
        "metrics_csv": str(Path(args.metrics_csv).resolve()),
        "stdout": str(Path(args.stdout).resolve()),
        "stderr": str(Path(args.stderr).resolve()),
        "gpu_inventory": first_gpu_rows,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.exit(return_code)


if __name__ == "__main__":
    main()
