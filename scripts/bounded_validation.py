"""Run a command in an isolated process group with a persistent 55-minute budget.

All elapsed child time (including test/setup overhead) counts conservatively.
An interrupted supervisor reserves its full allowance before starting the child.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path("results/validation")
    root.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with (root / "budget.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = root / "budget.json"
        entries = json.loads(ledger.read_text()) if ledger.exists() else []
        used = sum(e["elapsed_seconds"] for e in entries)
        allowance = min(args.seconds, 600, 3300 - used)
        if allowance <= 0:
            raise SystemExit("55-minute cumulative budget exhausted")
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        entry = dict(
            name=args.name,
            command=command,
            start=dt.datetime.now(dt.UTC).isoformat(),
            allowance_seconds=allowance,
            elapsed_seconds=allowance,
            status="reserved",
        )
        entries.append(entry)
        ledger.write_text(json.dumps(entries, indent=2) + "\n")
        env = dict(
            os.environ,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MPLCONFIGDIR=str(root.resolve() / "matplotlib"),
            WANDB_MODE="disabled",
            TMPDIR=str(root.resolve() / "tmp"),
        )
        start = time.monotonic()
        deadline = start + allowance
        process = None
        code = 1
        try:
            with (root / f"{args.name}.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
                entry["child_pid"] = process.pid
                try:
                    code = process.wait(timeout=max(0, deadline - time.monotonic()))
                    entry["status"] = "completed"
                except subprocess.TimeoutExpired:
                    entry["status"] = "timeout"
                    code = 124
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    # macOS may deny signaling an already reaped process group.
                    # Never let cleanup hide the result or discard budget accounting.
                    entry["cleanup_error"] = str(error)
                    if process.poll() is None:
                        process.kill()
                process.wait()
            entry.update(
                end=dt.datetime.now(dt.UTC).isoformat(),
                elapsed_seconds=time.monotonic() - start,
                returncode=code,
            )
            ledger.write_text(json.dumps(entries, indent=2) + "\n")
            print(json.dumps(entry))
            print(
                f'Cumulative charged time: {sum(e["elapsed_seconds"] for e in entries):.2f}/3300 s'
            )
        raise SystemExit(code)


if __name__ == "__main__":
    main()
