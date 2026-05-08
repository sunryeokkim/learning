"""CPU sonifier: turns live system stats into ambient music.

Mapping:
    CPU%  -> pitch of a melody note (C major pentatonic, two octaves).
    Mem%  -> volume of a low drone underneath everything.
    Net   -> short noise "tick" whenever bytes-per-second crosses a threshold.

Run with:
    python sonify.py

Deps: sounddevice, numpy, psutil (see requirements.txt).
On macOS the first run will ask for microphone/audio permission for the terminal.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import psutil
import sounddevice as sd


# ---- Audio constants ---------------------------------------------------------

SAMPLE_RATE = 44_100        # CD-quality; sounddevice picks this up by default
BLOCK_SIZE = 1024           # ~23 ms per audio callback at 44.1 kHz
POLL_INTERVAL = 0.25        # how often we re-read psutil, in seconds


# ---- Musical scale -----------------------------------------------------------
# A pentatonic scale has no "wrong" notes against itself, so even random
# CPU values land on something that sounds intentional. We use C major
# pentatonic across two octaves: C3 D3 E3 G3 A3  C4 D4 E4 G4 A4.
PENTATONIC_MIDI = [48, 50, 52, 55, 57, 60, 62, 64, 67, 69]


def midi_to_hz(midi_note: int) -> float:
    """Convert a MIDI note number to its frequency in Hz (A4=440 by convention)."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


PENTATONIC_HZ = [midi_to_hz(m) for m in PENTATONIC_MIDI]
DRONE_HZ = midi_to_hz(36)   # C2, deep bass drone


# ---- Shared state between the poll thread and the audio callback ------------
# The audio callback runs in a real-time thread; we keep state minimal and
# mutate it under a lock so we never tear a value mid-read.

@dataclass
class SharedState:
    target_freq: float = PENTATONIC_HZ[0]   # melody note we're gliding toward
    drone_gain: float = 0.0                 # 0..1, driven by memory %
    tick_energy: float = 0.0                # decays each block; spikes on net activity
    lock: threading.Lock = field(default_factory=threading.Lock)


state = SharedState()


# ---- Audio synthesis ---------------------------------------------------------

# Persistent oscillator phases. Keeping a running phase across blocks avoids
# clicks at block boundaries (resetting phase every block would cause pops).
_melody_phase = 0.0
_drone_phase = 0.0
_smoothed_freq = PENTATONIC_HZ[0]   # glides toward target_freq, no zipper noise
_rng = np.random.default_rng()


def audio_callback(outdata, frames, time_info, status):
    """Called by sounddevice on its real-time thread for each audio block.

    We must be fast and allocate as little as possible. Numpy vector ops are fine.
    """
    global _melody_phase, _drone_phase, _smoothed_freq

    if status:
        # Underruns/overruns get printed but we keep playing.
        print(f"[audio] {status}")

    with state.lock:
        target = state.target_freq
        drone_gain = state.drone_gain
        tick = state.tick_energy
        # Decay the tick so a single network spike rings out and fades.
        state.tick_energy *= 0.55

    # Glide smoothly toward the target frequency. A one-pole filter:
    # new = old + alpha * (target - old). Alpha near 0 = slow glide.
    alpha = 0.05
    _smoothed_freq += alpha * (target - _smoothed_freq)

    # --- Melody: a sine wave at the smoothed frequency ---
    # We advance phase by 2π * f / sr per sample. Using cumulative sum keeps
    # the phase continuous from the last block.
    phase_increment = 2.0 * math.pi * _smoothed_freq / SAMPLE_RATE
    phases = _melody_phase + phase_increment * np.arange(frames)
    melody = np.sin(phases) * 0.18
    _melody_phase = (phases[-1] + phase_increment) % (2.0 * math.pi)

    # --- Drone: deep sine, volume from memory % ---
    drone_inc = 2.0 * math.pi * DRONE_HZ / SAMPLE_RATE
    drone_phases = _drone_phase + drone_inc * np.arange(frames)
    drone = np.sin(drone_phases) * drone_gain * 0.25
    _drone_phase = (drone_phases[-1] + drone_inc) % (2.0 * math.pi)

    # --- Tick: short white-noise burst, scaled by tick_energy ---
    # Even though tick decays per-block, within a block it's effectively
    # constant; that's fine — at 23 ms per block it sounds like a quick click.
    tick_signal = _rng.standard_normal(frames) * tick * 0.4 if tick > 0.01 else 0.0

    # --- Mix and soft-clip so peaks don't distort harshly ---
    mixed = melody + drone + tick_signal
    mixed = np.tanh(mixed)

    # sounddevice expects shape (frames, channels). We're mono.
    outdata[:, 0] = mixed.astype(np.float32)


# ---- Stat polling ------------------------------------------------------------

def cpu_to_note_hz(cpu_percent: float) -> float:
    """Map 0..100 CPU% to a note in our pentatonic scale.

    Low CPU -> low note, high CPU -> high note. We bucket continuously so
    small CPU changes still re-pitch the melody.
    """
    idx = int(cpu_percent / 100.0 * (len(PENTATONIC_HZ) - 1))
    idx = max(0, min(len(PENTATONIC_HZ) - 1, idx))
    return PENTATONIC_HZ[idx]


def poll_loop():
    """Poll psutil and push values into the shared state. Runs in its own thread."""
    last_net = psutil.net_io_counters()
    last_time = time.monotonic()

    # First call to cpu_percent() returns 0.0 — prime it.
    psutil.cpu_percent(interval=None)

    while True:
        time.sleep(POLL_INTERVAL)

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        now = psutil.net_io_counters()
        elapsed = time.monotonic() - last_time
        bytes_per_sec = ((now.bytes_sent + now.bytes_recv)
                         - (last_net.bytes_sent + last_net.bytes_recv)) / max(elapsed, 1e-6)
        last_net = now
        last_time = time.monotonic()

        # Tick threshold: anything above ~50 KB/s makes an audible click,
        # bigger bursts make louder clicks (capped so a download doesn't blast).
        tick_kick = 0.0
        if bytes_per_sec > 50_000:
            tick_kick = min(1.0, math.log10(bytes_per_sec / 50_000 + 1) * 0.7)

        with state.lock:
            state.target_freq = cpu_to_note_hz(cpu)
            state.drone_gain = mem / 100.0
            # Add to tick energy rather than overwriting — multiple spikes stack.
            state.tick_energy = min(1.0, state.tick_energy + tick_kick)

        # Console readout so you can see what's driving the sound.
        bar = "#" * int(cpu / 5)
        print(f"\rCPU {cpu:5.1f}% [{bar:<20}]  MEM {mem:5.1f}%  NET {bytes_per_sec/1024:7.1f} KB/s",
              end="", flush=True)


# ---- Entry point -------------------------------------------------------------

def main():
    print("CPU sonifier — Ctrl+C to stop.\n")

    # Daemon thread so it dies with the process.
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

    # OutputStream with a callback gives us continuous, glitch-free audio.
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nbye.")


if __name__ == "__main__":
    main()
