"""workload.py — generate a 30-second scripted workload arc to drive compose.py.

Without artificial load, the captured metrics tend to be flat (idle laptop)
and the song comes out monotonous. This script runs a deliberate 30s arc
of varying CPU / network / disk activity. When compose.py samples it, the
resulting song has clear dynamics:

    seconds  section       cpu%   net KB/s   disk MB/s
      0–3    intro           10          0         –
      3–6    build           30         20         –
      6–12   verse           50         60         –
     12–18   rising          80        150         –        ← 16th hats start at >100 KB/s
     18–25   chorus          95        250         –
     20–21   ↳ disk spike    95        250       3.0        ← tom hit on this bar
     25–28   breakdown       20         20         –
     28–30   outro           10          0         –

Usage:

    # In one terminal, after starting compose.py manually:
    python workload.py

    # Or have workload.py run compose_v2.py for you (recommended):
    python workload.py --render               # writes song_v2.wav
    python workload.py --render --out my.wav  # custom output filename

CPU is driven by spawning one busy-loop process per core, each with a
duty-cycled spin. Network is real UDP packets to localhost (loopback shows
up in psutil's net counters). Disk is real writes to /tmp.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import subprocess
import sys
import threading
import time
from multiprocessing import Process, Value

WORKLOAD_DURATION = 30.0

# Each entry: (t_start, t_end, cpu_percent_target, net_kbps_target, label).
# Aligned with the v3 song structure: each bar = 2 s.
#   bars  0–1   intro   (0–4 s)
#   bars  2–5   verse   (4–12 s)
#   bars  6–8   build   (12–18 s)
#   bars  9–12  chorus  (18–26 s)
#   bar  13     break   (26–28 s)
#   bar  14     outro   (28–30 s)
SCORE = [
    (0.0,   4.0, 10,   0, "intro"),
    (4.0,  12.0, 45,  60, "verse"),
    (12.0, 18.0, 80, 200, "build"),
    (18.0, 26.0, 95, 300, "chorus"),
    (26.0, 28.0, 20,  20, "break"),
    (28.0, 30.0, 10,   0, "outro"),
]

# Disk spikes overlay any CPU/net section. We land them at bar boundaries so
# the captured spike falls into the snapshot the song's tom-hit logic reads.
DISK_SPIKES = [
    (10.0, 11.0, 1.5),   # bar 5 (end of verse) → tom hit lifting into build
    (20.0, 21.0, 3.0),   # bar 10 (chorus) → tom hit on the chorus body
]

DISK_TMPFILE = "/tmp/cpu_sonifier_workload.bin"


def section_at(t: float) -> tuple[int, int, str]:
    for t0, t1, cpu, net, label in SCORE:
        if t0 <= t < t1:
            return cpu, net, label
    return 0, 0, "idle"


def disk_at(t: float) -> float:
    for t0, t1, mb in DISK_SPIKES:
        if t0 <= t < t1:
            return mb
    return 0.0


# ---- Workers ----------------------------------------------------------------

def cpu_burner(cpu_target, stop_val):
    """Subprocess: spin one core at `cpu_target.value` percent duty cycle.

    We need multiprocessing (not threading) because Python's GIL would
    serialize threaded busy-loops onto one core. With one process per core,
    each can independently saturate its own core when busy.
    """
    while stop_val.value == 0:
        # Clamp into [0, 1]. The whole loop runs in a 20ms window.
        target = max(0.0, min(1.0, cpu_target.value / 100.0))
        window = 0.020
        busy = window * target
        idle = window - busy
        if busy > 0:
            end = time.monotonic() + busy
            x = 1.0
            # Run some math the optimizer can't elide.
            while time.monotonic() < end:
                x = math.sqrt(x + 12345.6789) + 0.0001
        if idle > 0:
            time.sleep(idle)


def net_burner(net_target, stop_event):
    """Thread: send UDP packets to localhost to drive net counters.

    psutil.net_io_counters() includes loopback (lo0) by default, so packets
    sent to 127.0.0.1 show up as bytes-sent. We don't need a listener; the
    bytes count regardless of whether anything's bound on the port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    chunk = b"x" * 1024  # 1 KB per send
    while not stop_event.is_set():
        rate_kbps = net_target.value
        if rate_kbps <= 0:
            time.sleep(0.05)
            continue
        try:
            sock.sendto(chunk, ("127.0.0.1", 9999))
        except OSError:
            pass
        # 1 KB per send → `rate_kbps` sends/sec → sleep 1/rate seconds.
        time.sleep(1.0 / max(1.0, rate_kbps))


def disk_burner(disk_target, stop_event):
    """Thread: when disk_target > 0, write real bytes to /tmp."""
    while not stop_event.is_set():
        rate_mbps = disk_target.value
        if rate_mbps <= 0:
            time.sleep(0.05)
            continue
        # Write half-second chunks; the loop spins fast when active so the
        # 1s spike window gets multiple writes for steady throughput.
        chunk_size = max(64 * 1024, int(rate_mbps * 1024 * 1024 * 0.5))
        try:
            with open(DISK_TMPFILE, "wb") as f:
                f.write(os.urandom(chunk_size))
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
        time.sleep(0.5)
    # Best-effort cleanup of the tmp file when we shut down.
    try:
        os.remove(DISK_TMPFILE)
    except OSError:
        pass


# ---- Orchestration ----------------------------------------------------------

def run_workload(render: str | None, out_path: str) -> None:
    cpu_target = Value("d", 0.0)
    net_target = Value("d", 0.0)
    disk_target = Value("d", 0.0)
    stop_val = Value("i", 0)

    n_cores = os.cpu_count() or 4
    print(f"Spawning {n_cores} CPU burner processes...")
    cpu_procs = [Process(target=cpu_burner, args=(cpu_target, stop_val), daemon=True)
                 for _ in range(n_cores)]
    for p in cpu_procs:
        p.start()

    stop_event = threading.Event()
    net_thread = threading.Thread(target=net_burner,
                                  args=(net_target, stop_event), daemon=True)
    disk_thread = threading.Thread(target=disk_burner,
                                   args=(disk_target, stop_event), daemon=True)
    net_thread.start()
    disk_thread.start()

    compose_proc = None
    if render:
        # Start the chosen compose script in the same Python that ran
        # workload.py (i.e. the venv) so dependencies line up. --no-play
        # because we'll tell the user to afplay after the file is ready.
        print(f"Spawning {render} to capture in parallel → {out_path}")
        compose_proc = subprocess.Popen(
            [sys.executable, render, "--no-play", "--out", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    print(f"\n{'t':>5} {'cpu%':>5} {'net KB/s':>9} {'disk MB/s':>10}  section")
    print("-" * 50)

    start = time.monotonic()
    last_label = None
    try:
        while True:
            t = time.monotonic() - start
            if t >= WORKLOAD_DURATION:
                break

            cpu, net, label = section_at(t)
            disk = disk_at(t)
            cpu_target.value = float(cpu)
            net_target.value = float(net)
            disk_target.value = float(disk)

            # Print on label change or every full second otherwise.
            line = f"{t:5.1f} {cpu:5d} {net:9d} {disk:10.1f}  {label}"
            if label != last_label or disk > 0:
                print(line)
                last_label = label
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        # Wind everything down: zero the targets first so workers stop
        # producing load, then signal stop and join.
        cpu_target.value = 0.0
        net_target.value = 0.0
        disk_target.value = 0.0
        stop_val.value = 1
        stop_event.set()
        for p in cpu_procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        net_thread.join(timeout=1)
        disk_thread.join(timeout=1)

    if compose_proc is not None:
        print(f"\nWaiting for {render} to finish writing the file...")
        compose_proc.wait(timeout=10)
        print(f"Done. Play it:  afplay {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--render", nargs="?", const="compose_v3.py", default=None,
                        metavar="SCRIPT",
                        help="Also run the given compose script in parallel "
                             "(default: compose_v3.py). e.g. --render or "
                             "--render compose_v2.py")
    parser.add_argument("--out", default="song_v3.wav",
                        help="Output WAV path (only used with --render).")
    args = parser.parse_args()
    run_workload(render=args.render, out_path=args.out)


if __name__ == "__main__":
    main()
