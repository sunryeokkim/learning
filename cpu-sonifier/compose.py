"""compose.py — capture 30s of system metrics and render them as a 30s song.

Unlike sonify.py (which streams a continuous tone driven by live CPU), this
script first samples psutil once per second for 30s, then *composes* an offline
song. The musical structure is fixed: 120 BPM, 4/4, with a I–vi–IV–V chord
progression. The metrics drive how the song is played within that structure,
not the raw frequencies — so the result has melody, rhythm, and a key.

Per-snapshot mappings (one snapshot = 2 beats = a half-bar):
    CPU%             → which 4-note arpeggio pattern over the chord
    per-core spread  → adds a passing scale tone on beat 3 when cores diverge
    Memory%          → melody octave (above 65% lifts melody up an octave)
    Swap usage       → swaps progression to a minor variant (vi-IV-I-V)
    Net bytes/sec    → doubles hi-hat to 16ths when traffic is high
    Disk write delta → tom hit on beat 1 of "busy" snapshots
    Load avg 1m      → overall drum kit gain
    Process count    → arpeggio direction (even=up, odd=down)
    Ctx switches     → captured but only printed (not mapped to audio)

Run:
    python compose.py            # capture, compose, save song.wav, play it
    python compose.py --no-play  # skip playback (just write the file)
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np
import psutil
import sounddevice as sd


# ---- Constants ---------------------------------------------------------------

SAMPLE_RATE = 44_100
BPM = 120
BEAT_SEC = 60.0 / BPM            # 0.5 s
EIGHTH_SEC = BEAT_SEC / 2.0      # 0.25 s
SIXTEENTH_SEC = BEAT_SEC / 4.0   # 0.125 s

NUM_SNAPSHOTS = 30
SNAPSHOT_INTERVAL = 1.0          # 1 snapshot per second
# 30 snapshots × 1 s = 30 s. At 120 BPM, each snapshot = 2 beats = half-bar.
# So the song is 30 × 2 = 60 beats = 15 bars of 4/4.

# Chord changes happen every BAR (i.e. every 2 snapshots), giving 7.5 cycles
# of a 4-chord progression over the 15 bars.
PROGRESSION_MAJOR = ["C", "Am", "F", "G"]   # I  vi  IV  V — happy
PROGRESSION_MINOR = ["Am", "F", "C", "G"]   # vi IV  I  V  — wistful

# Chord = (root_midi, third_midi, fifth_midi). We keep the chord low and
# voice the melody an octave above.
CHORDS = {
    "C":  (60, 64, 67),
    "Am": (57, 60, 64),
    "F":  (53, 57, 60),
    "G":  (55, 59, 62),
}


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


# ---- Metric capture ----------------------------------------------------------

@dataclass
class Snapshot:
    cpu: float
    cpu_per_max_min: float   # max-min spread across cores
    mem: float
    swap: float
    net_bps: float           # bytes/sec since previous snapshot
    disk_write_delta: float  # bytes since previous snapshot
    load_1m: float
    pids: int
    ctx_switch_delta: int


def capture(n: int = NUM_SNAPSHOTS, interval: float = SNAPSHOT_INTERVAL) -> list[Snapshot]:
    print(f"Capturing {n} snapshots over {n * interval:.0f}s...")
    print("(open a browser tab, run 'yes >/dev/null' in another terminal — "
          "anything that varies CPU/net/disk will color the song)\n")

    # Prime psutil's diff-based counters so the first reading isn't 0.
    psutil.cpu_percent(interval=None)
    prev_net = psutil.net_io_counters()
    prev_disk = psutil.disk_io_counters()
    prev_ctx = psutil.cpu_stats().ctx_switches
    prev_t = time.monotonic()

    snaps: list[Snapshot] = []
    for i in range(n):
        time.sleep(interval)

        cpu = psutil.cpu_percent(interval=None)
        per = psutil.cpu_percent(percpu=True)
        spread = max(per) - min(per) if per else 0.0
        mem = psutil.virtual_memory().percent
        swap = psutil.swap_memory().percent

        now_net = psutil.net_io_counters()
        now_disk = psutil.disk_io_counters()
        now_ctx = psutil.cpu_stats().ctx_switches
        now_t = time.monotonic()
        elapsed = max(now_t - prev_t, 1e-6)

        net_bps = ((now_net.bytes_sent + now_net.bytes_recv)
                   - (prev_net.bytes_sent + prev_net.bytes_recv)) / elapsed
        disk_write_delta = (now_disk.write_bytes - prev_disk.write_bytes) if now_disk and prev_disk else 0
        ctx_delta = now_ctx - prev_ctx

        try:
            load_1m = psutil.getloadavg()[0]
        except (AttributeError, OSError):
            load_1m = 0.0

        snap = Snapshot(
            cpu=cpu,
            cpu_per_max_min=spread,
            mem=mem,
            swap=swap,
            net_bps=net_bps,
            disk_write_delta=disk_write_delta,
            load_1m=load_1m,
            pids=len(psutil.pids()),
            ctx_switch_delta=ctx_delta,
        )
        snaps.append(snap)

        print(f"  [{i + 1:2d}/{n}] cpu {cpu:5.1f}% spread {spread:5.1f}  "
              f"mem {mem:5.1f}%  swap {swap:4.1f}%  "
              f"net {net_bps / 1024:7.1f} KB/s  "
              f"disk_w {disk_write_delta / 1024:7.1f} KB  "
              f"load {load_1m:.2f}  procs {len(psutil.pids())}  "
              f"ctx Δ{ctx_delta}")

        prev_net, prev_disk, prev_ctx, prev_t = now_net, now_disk, now_ctx, now_t

    return snaps


# ---- Synthesis primitives ----------------------------------------------------

def adsr(n_samples: int, sr: int, a: float, d: float, s_level: float, r: float) -> np.ndarray:
    """Build an attack-decay-sustain-release amplitude envelope of length n_samples."""
    n_a = max(1, int(a * sr))
    n_d = max(1, int(d * sr))
    n_r = max(1, int(r * sr))
    n_s = max(0, n_samples - n_a - n_d - n_r)
    parts = [
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, s_level, n_d, endpoint=False),
        np.full(n_s, s_level),
        np.linspace(s_level, 0.0, n_r, endpoint=True),
    ]
    env = np.concatenate(parts)
    if len(env) < n_samples:
        env = np.concatenate([env, np.zeros(n_samples - len(env))])
    return env[:n_samples]


def synth_tone(freq: float, dur: float, sr: int, waveform: str,
               env_kw: dict) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    if waveform == "sine":
        wave_arr = np.sin(2 * np.pi * freq * t)
    elif waveform == "tri":
        # Triangle from a phase ramp: 4|x - floor(x+0.5)| - 1, scaled.
        ramp = freq * t
        wave_arr = 2.0 * np.abs(2.0 * (ramp - np.floor(ramp + 0.5))) - 1.0
    elif waveform == "saw":
        ramp = freq * t
        wave_arr = 2.0 * (ramp - np.floor(ramp + 0.5))
    elif waveform == "square":
        wave_arr = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        raise ValueError(waveform)
    env = adsr(n, sr, **env_kw)
    return wave_arr * env


def synth_kick(dur: float, sr: int) -> np.ndarray:
    """Pitch-swept sine: thumpy kick drum."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    # Frequency drops fast from ~120 Hz to ~45 Hz.
    freq = 45.0 + 80.0 * np.exp(-t * 30.0)
    # Cumulative phase so the dropping freq stays continuous.
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 11.0)


def synth_snare(dur: float, sr: int) -> np.ndarray:
    """Noise + a tonal blip → backbeat snare."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n)
    tone = np.sin(2 * np.pi * 200 * t)
    return (0.7 * noise + 0.3 * tone) * np.exp(-t * 18.0)


def synth_hat(dur: float, sr: int, seed: int = 0) -> np.ndarray:
    """High-passed-ish noise burst → hi-hat."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # Approximate high-pass via first-difference (emphasizes high freqs).
    noise = np.diff(noise, prepend=0.0)
    t = np.arange(n) / sr
    return noise * np.exp(-t * 60.0)


def synth_tom(dur: float, sr: int) -> np.ndarray:
    """Mid-frequency drum hit, slower decay than kick."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = 110.0 + 60.0 * np.exp(-t * 18.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 8.0)


# ---- Composition -------------------------------------------------------------

def add_at(track: np.ndarray, sig: np.ndarray, t_sec: float, sr: int, gain: float = 1.0) -> None:
    """Mix `sig * gain` into `track` starting at second `t_sec`. Clips at end."""
    i0 = int(t_sec * sr)
    i1 = min(len(track), i0 + len(sig))
    if i1 > i0:
        track[i0:i1] += sig[: i1 - i0] * gain


# Four arpeggio patterns over a chord (root, third, fifth). Each entry is an
# index into the chord tuple (or 7 for "scale tone above fifth" — handled
# specially below).
ARPEGGIO_PATTERNS = [
    (0, 1, 2, 1),  # root, 3rd, 5th, 3rd       — calm
    (1, 2, 0, 2),  # 3rd, 5th, root↑, 5th      — lifted
    (2, 1, 0, 1),  # 5th, 3rd, root, 3rd       — descending
    (0, 2, 1, 2),  # root, 5th, 3rd, 5th       — open
]


def compose(snaps: list[Snapshot]) -> np.ndarray:
    sr = SAMPLE_RATE
    total_samples = int(NUM_SNAPSHOTS * SNAPSHOT_INTERVAL * sr)

    # Three buses we'll mix at the end, so we can balance them independently.
    melody = np.zeros(total_samples, dtype=np.float32)
    bass = np.zeros(total_samples, dtype=np.float32)
    drums = np.zeros(total_samples, dtype=np.float32)

    # Decide major or minor based on whether swap is being used at all.
    avg_swap = sum(s.swap for s in snaps) / len(snaps)
    progression = PROGRESSION_MINOR if avg_swap > 5.0 else PROGRESSION_MAJOR
    print(f"\nMode: {'minor' if progression is PROGRESSION_MINOR else 'major'} "
          f"(avg swap = {avg_swap:.1f}%)")

    # Pre-render some drum hits — reuse the same buffers.
    kick_buf = synth_kick(0.30, sr)
    snare_buf = synth_snare(0.25, sr)
    tom_buf = synth_tom(0.30, sr)

    for i, s in enumerate(snaps):
        t0 = i * SNAPSHOT_INTERVAL  # snapshot covers [t0, t0+1s) = 2 beats

        # --- Pick this snapshot's chord ---
        # Chord changes every BAR = every 2 snapshots.
        chord_idx = (i // 2) % len(progression)
        chord_name = progression[chord_idx]
        chord_midi = CHORDS[chord_name]   # (root, 3rd, 5th)

        # --- Arpeggio pattern: which of 4 patterns, decided by CPU bucket ---
        pat_idx = min(3, int(s.cpu / 25.0))
        pattern = ARPEGGIO_PATTERNS[pat_idx]

        # Octave shift on melody when memory is high (busy machine = bright).
        mel_octave = 12 if s.mem > 65 else 0

        # Process count parity flips arpeggio direction (subtle flavor).
        if s.pids % 2 == 1:
            pattern = pattern[::-1]

        # --- Bass: 2 quarter notes per snapshot (root, 5th) ---
        for beat_i, midi in enumerate([chord_midi[0] - 12, chord_midi[2] - 12]):
            t = t0 + beat_i * BEAT_SEC
            sig = synth_tone(midi_to_hz(midi), BEAT_SEC * 0.95, sr, "tri",
                             dict(a=0.005, d=0.06, s_level=0.7, r=0.05))
            add_at(bass, sig, t, sr, gain=0.45)

        # --- Melody: 4 eighth notes per snapshot ---
        for j in range(4):
            t = t0 + j * EIGHTH_SEC
            tone_idx = pattern[j]
            note_midi = chord_midi[tone_idx] + 12 + mel_octave
            # If cores are diverging a lot, replace the 3rd note with a
            # scale-tone passing note (one whole step above the chord tone).
            if j == 2 and s.cpu_per_max_min > 30.0:
                note_midi += 2  # up a whole step → passing tone
            sig = synth_tone(midi_to_hz(note_midi), EIGHTH_SEC * 0.9, sr, "tri",
                             dict(a=0.004, d=0.08, s_level=0.35, r=0.04))
            add_at(melody, sig, t, sr, gain=0.32)

        # --- Drums ---
        # Overall drum gain scaled gently by load average (cap at ~2.0).
        drum_gain = min(1.4, 0.7 + 0.35 * s.load_1m)

        # Kick on each of the 2 beats in this snapshot.
        for beat_i in range(2):
            add_at(drums, kick_buf, t0 + beat_i * BEAT_SEC, sr, gain=0.55 * drum_gain)

        # Snare on beat 2 of the snapshot (= beat 2 or 4 of the bar = backbeat).
        add_at(drums, snare_buf, t0 + BEAT_SEC, sr, gain=0.40 * drum_gain)

        # Hi-hats: eighths normally, sixteenths when network is busy.
        hat_step = SIXTEENTH_SEC if s.net_bps > 100_000 else EIGHTH_SEC
        n_hats = int(round(SNAPSHOT_INTERVAL / hat_step))
        for k in range(n_hats):
            t = t0 + k * hat_step
            # Each hat slightly different so they don't sound mechanical.
            hat = synth_hat(0.07, sr, seed=i * 100 + k)
            add_at(drums, hat, t, sr, gain=0.18 * drum_gain)

        # Tom hit on beat 1 if there's a disk-write spike (>1 MB/s).
        if s.disk_write_delta > 1_000_000:
            add_at(drums, tom_buf, t0, sr, gain=0.40 * drum_gain)

    # --- Final mix: balance, then soft-clip ---
    mix = melody + bass + drums
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.9
    mix = np.tanh(mix * 1.05)  # gentle saturation for warmth
    return mix.astype(np.float32)


# ---- IO ----------------------------------------------------------------------

def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write a mono float-array as a 16-bit PCM WAV using only stdlib."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-play", action="store_true",
                        help="Just write song.wav; don't play it back.")
    parser.add_argument("--out", default="song.wav", help="Output WAV path.")
    args = parser.parse_args()

    snaps = capture()
    print("\nComposing...")
    audio = compose(snaps)
    write_wav(args.out, audio, SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(f"Saved {args.out}  ({duration:.1f}s, {len(audio)} samples @ {SAMPLE_RATE} Hz)")

    if not args.no_play:
        print("Playing... (Ctrl+C to stop early)")
        try:
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("\nstopped.")


if __name__ == "__main__":
    main()
