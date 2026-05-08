"""compose_v2.py — captures system metrics and renders a more polished song.

Differences from compose.py (v1):

  Sound design (stylish):
    • Lead uses a 3-saw detuned stack + sub-sine instead of a bare triangle.
    • A sustained pad layer plays voiced 7th chords underneath everything.
    • Output is stereo with positional panning per voice.
    • A Schroeder reverb (4 comb filters + 2 allpass filters) gives space.

  Composition (melodious):
    • Chord progression uses 7th extensions: Cmaj7 – Am7 – Fmaj7 – G7.
    • Melody follows a phrase contour that arcs up and back over the song.
    • 8th notes are swung (~58/42) instead of dead-straight.
    • Velocity dynamics: downbeats are louder than off-beats.
    • Melody walks through chord tones AND scale passing tones, with a
      deliberate lead-in step from beat 4 of one bar into beat 1 of the next.

The capture phase is reused from compose.py.

Run:
    python compose_v2.py
    python compose_v2.py --no-play --out song_v2.wav
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np
import sounddevice as sd

# Reuse the capture loop and Snapshot dataclass from v1.
from compose import NUM_SNAPSHOTS, SNAPSHOT_INTERVAL, Snapshot, capture


# ---- Constants ---------------------------------------------------------------

SAMPLE_RATE = 44_100
BPM = 120
BEAT_SEC = 60.0 / BPM           # 0.5 s
SWING = 0.58                    # first 8th gets 58% of the beat (the "shuffle")
EIGHTH_DOWN_OFFSET = 0.0
EIGHTH_UP_OFFSET = BEAT_SEC * SWING

# 30 snapshots × 1 s = 30 s. At 120 BPM, each snapshot = 2 beats = half-bar.
# 30 snapshots × 2 beats = 60 beats = 15 bars of 4/4.
NUM_BARS = 15

# 7th-chord voicings: (root, 3rd, 5th, 7th) as MIDI numbers in the bass octave.
# Adding the 7th gives a richer, jazzier color than plain triads.
CHORDS = {
    "Cmaj7": (60, 64, 67, 71),
    "Am7":   (57, 60, 64, 67),
    "Fmaj7": (53, 57, 60, 64),
    "G7":    (55, 59, 62, 65),
}
PROG_MAJOR = ["Cmaj7", "Am7", "Fmaj7", "G7"]
PROG_MINOR = ["Am7", "Fmaj7", "Cmaj7", "G7"]

# C major / A natural minor share the same 7-note set, so one scale covers both.
SCALE_DEGREES = [0, 2, 4, 5, 7, 9, 11]   # semitones above C


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def step_in_scale(midi: int, direction: int) -> int:
    """Move one scale-degree (in C major) up or down from `midi`."""
    semitone = midi % 12
    octave = midi // 12
    if semitone in SCALE_DEGREES:
        idx = SCALE_DEGREES.index(semitone)
    else:
        # Snap to nearest scale tone, then step from there.
        idx = min(range(len(SCALE_DEGREES)),
                  key=lambda i: abs(SCALE_DEGREES[i] - semitone))
    new_idx = idx + direction
    new_oct = octave + (new_idx // len(SCALE_DEGREES))
    new_semi = SCALE_DEGREES[new_idx % len(SCALE_DEGREES)]
    return new_oct * 12 + new_semi


def closest(target: int, candidates: list[int]) -> int:
    return min(candidates, key=lambda n: abs(n - target))


# ---- Envelope & basic synth primitives ---------------------------------------

def adsr(n: int, sr: int, a: float, d: float, s_level: float, r: float) -> np.ndarray:
    n_a = max(1, int(a * sr))
    n_d = max(1, int(d * sr))
    n_r = max(1, int(r * sr))
    n_s = max(0, n - n_a - n_d - n_r)
    env = np.concatenate([
        np.linspace(0.0, 1.0, n_a, endpoint=False),
        np.linspace(1.0, s_level, n_d, endpoint=False),
        np.full(n_s, s_level),
        np.linspace(s_level, 0.0, n_r, endpoint=True),
    ])
    if len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    return env[:n]


def saw_stack(freq: float, dur: float, sr: int, detune_cents=(-9.0, 0.0, 9.0)) -> np.ndarray:
    """Sum of N saw waves at slightly different pitches.

    A small detune (cents) makes the stack beat against itself, which is what
    gives "supersaw"-style synths their thick, animated character. With one
    saw it's harsh; with three lightly detuned, it's lush.
    """
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for cents in detune_cents:
        f = freq * (2.0 ** (cents / 1200.0))
        ramp = f * t
        out += 2.0 * (ramp - np.floor(ramp + 0.5))   # saw ∈ [-1, 1]
    return out / len(detune_cents)


def lowpass_fir(x: np.ndarray, cutoff: float, sr: int, ntaps: int = 33) -> np.ndarray:
    """Vectorized low-pass via a windowed-sinc FIR filter.

    A sinc kernel windowed with Hamming makes a clean low-pass. We convolve
    rather than running an IIR loop because numpy.convolve is much faster
    than a per-sample Python recurrence.
    """
    if cutoff >= sr / 2:
        return x
    n = np.arange(ntaps) - (ntaps - 1) / 2.0
    h = np.sinc(2.0 * cutoff / sr * n) * np.hamming(ntaps)
    h = h / h.sum()
    return np.convolve(x, h, mode="same")


# ---- Voices ------------------------------------------------------------------

def synth_lead(freq: float, dur: float, sr: int, velocity: float = 1.0) -> np.ndarray:
    """Bright melody voice: detuned saws + sub-sine, low-passed, plucky envelope."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    saws = saw_stack(freq, dur, sr, detune_cents=(-9.0, 0.0, 9.0))
    sub = 0.35 * np.sin(2 * np.pi * (freq / 2) * t)         # body an octave below
    raw = saws + sub
    raw = lowpass_fir(raw, cutoff=2800.0, sr=sr)            # tame the saw fizz
    env = adsr(n, sr, a=0.005, d=0.07, s_level=0.45, r=0.06)
    return raw * env * velocity


def synth_pad(midi_notes, dur: float, sr: int, velocity: float = 1.0) -> np.ndarray:
    """Slow, sustained chord pad: a saw stack on each chord tone, low-passed warm."""
    n = int(dur * sr)
    out = np.zeros(n)
    for m in midi_notes:
        out += saw_stack(midi_to_hz(m), dur, sr, detune_cents=(-12.0, 0.0, 12.0))
    out /= max(1, len(midi_notes))
    out = lowpass_fir(out, cutoff=900.0, sr=sr)
    env = adsr(n, sr, a=0.45, d=0.5, s_level=0.7, r=0.55)
    return out * env * velocity


def synth_bass(freq: float, dur: float, sr: int, velocity: float = 1.0) -> np.ndarray:
    """Triangle + sub-sine bass: cuts through the mix without being boomy."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    ramp = freq * t
    tri = 2.0 * np.abs(2.0 * (ramp - np.floor(ramp + 0.5))) - 1.0
    sub = 0.5 * np.sin(2 * np.pi * (freq / 2) * t)
    raw = tri + sub
    env = adsr(n, sr, a=0.005, d=0.06, s_level=0.7, r=0.05)
    return raw * env * velocity


def synth_kick(dur: float, sr: int) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    freq = 45.0 + 80.0 * np.exp(-t * 30.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 11.0)


def synth_snare(dur: float, sr: int) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)
    return ((0.7 * rng.standard_normal(n) + 0.3 * np.sin(2 * np.pi * 200 * t))
            * np.exp(-t * 18.0))


def synth_hat(dur: float, sr: int, seed: int = 0, open_: bool = False) -> np.ndarray:
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    noise = np.diff(rng.standard_normal(n), prepend=0.0)    # crude high-pass
    decay = 22.0 if open_ else 60.0
    t = np.arange(n) / sr
    return noise * np.exp(-t * decay)


# ---- Schroeder reverb --------------------------------------------------------
# A classic small-room reverb: 4 parallel comb filters (each adds a series of
# decaying echoes) summed, then 2 allpass filters in series (which smear the
# echoes into a diffuse tail). All implemented with chunked vectorized adds —
# each chunk is `delay` samples wide, so a 30s buffer takes only ~1k iterations.

def comb_filter(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = x[n] + g·y[n-d].  Recursive; computed chunk-by-chunk for speed."""
    if d <= 0:
        return x.copy()
    y = x.copy()
    n = len(x)
    start = d
    while start < n:
        end = min(start + d, n)
        chunk = end - start
        y[start:end] += g * y[start - d:start - d + chunk]
        start = end
    return y


def allpass_filter(x: np.ndarray, d: int, g: float) -> np.ndarray:
    """y[n] = -g·x[n] + x[n-d] + g·y[n-d].  Diffuses the comb output."""
    if d <= 0:
        return x.copy()
    y = np.zeros_like(x)
    y[:d] = -g * x[:d]
    n = len(x)
    start = d
    while start < n:
        end = min(start + d, n)
        chunk = end - start
        y[start:end] = (-g * x[start:end]
                        + x[start - d:start - d + chunk]
                        + g * y[start - d:start - d + chunk])
        start = end
    return y


def schroeder_reverb(x: np.ndarray, sr: int, seed: int = 0, mix: float = 0.22) -> np.ndarray:
    """Mix a small room reverb into x. seed varies delay times slightly per channel."""
    rng = np.random.default_rng(seed)
    base_combs = [0.0297, 0.0371, 0.0411, 0.0437]
    comb_gain = 0.84
    wet = np.zeros_like(x)
    for d_sec in base_combs:
        # Tiny ±3% delay-time variation per call so L and R aren't identical.
        d_sec_jit = d_sec * (1.0 + rng.uniform(-0.03, 0.03))
        wet += comb_filter(x, int(d_sec_jit * sr), comb_gain)
    wet /= len(base_combs)

    for d_sec in [0.005, 0.0017]:
        wet = allpass_filter(wet, int(d_sec * sr), 0.5)

    return (1.0 - mix) * x + mix * wet


# ---- Composition helpers -----------------------------------------------------

def phrase_contour(num_snapshots: int) -> list[int]:
    """A song-shaped pitch-center per snapshot, in semitones from C4.

    Sections (across our 30 snapshots):
       0– 3   intro       — home (0)
       4– 9   build       — up a fourth (+5)
      10–19   chorus      — up an octave (+12)
      20–23   descent     — back to a fifth (+7)
      24–29   outro       — home (0)
    """
    contour = []
    for i in range(num_snapshots):
        if i < 4:
            contour.append(0)
        elif i < 10:
            contour.append(5)
        elif i < 20:
            contour.append(12)
        elif i < 24:
            contour.append(7)
        else:
            contour.append(0)
    return contour


def swing_offset(j: int) -> float:
    """Time (seconds) of the j-th 8th note within a 2-beat snapshot, with swing."""
    # Snapshot = 2 beats. Within each beat: down-eighth at 0, up-eighth at SWING.
    beat = j // 2
    is_up = (j % 2) == 1
    return beat * BEAT_SEC + (EIGHTH_UP_OFFSET if is_up else EIGHTH_DOWN_OFFSET)


def velocity_for(j: int, jitter_rng: np.random.Generator) -> float:
    """Beat-strength curve for the 4 8th-notes in a snapshot, plus a tiny jitter."""
    base = [0.95, 0.65, 0.85, 0.6][j]
    return float(np.clip(base + jitter_rng.uniform(-0.05, 0.05), 0.3, 1.0))


# ---- Composition (build event list, then render) -----------------------------

def compose(snaps: list[Snapshot]) -> np.ndarray:
    sr = SAMPLE_RATE
    total_samples = int(NUM_SNAPSHOTS * SNAPSHOT_INTERVAL * sr)

    # Stereo buses we mix at the end. Shape (samples, 2): col 0 = L, col 1 = R.
    melody_bus = np.zeros((total_samples, 2), dtype=np.float32)
    bass_bus = np.zeros((total_samples, 2), dtype=np.float32)
    pad_bus = np.zeros((total_samples, 2), dtype=np.float32)
    drum_bus = np.zeros((total_samples, 2), dtype=np.float32)

    avg_swap = sum(s.swap for s in snaps) / len(snaps)
    progression = PROG_MINOR if avg_swap > 5.0 else PROG_MAJOR
    print(f"\nMode: {'minor' if progression is PROG_MINOR else 'major'} "
          f"(avg swap = {avg_swap:.1f}%)")

    contour = phrase_contour(NUM_SNAPSHOTS)
    rng = np.random.default_rng(7)

    # Pre-render drum hits we can reuse.
    kick_buf = synth_kick(0.30, sr)
    snare_buf = synth_snare(0.28, sr)
    tom_buf = synth_kick(0.30, sr) * 0.0  # placeholder (no toms in v2 standard kit)

    last_lead_midi: int | None = None

    for i, s in enumerate(snaps):
        t0 = i * SNAPSHOT_INTERVAL
        chord_idx = (i // 2) % len(progression)
        chord = CHORDS[progression[chord_idx]]
        next_i = i + 1
        next_chord = (CHORDS[progression[(next_i // 2) % len(progression)]]
                      if next_i < NUM_SNAPSHOTS else None)

        shift = contour[i]
        target_center = 60 + shift                  # MIDI note around which melody hovers
        if s.mem > 65:                              # high mem: lift an extra octave
            target_center += 12

        # ---- Pad: one held chord per BAR (= every 2 snapshots, on even i) ----
        if i % 2 == 0:
            # Pad voicing: spread the chord across two octaves for fullness.
            pad_voicing = [chord[0], chord[1], chord[2], chord[3], chord[1] + 12]
            # Hold for 2 snapshots = 2 seconds (one bar).
            pad_dur = min(2.0 * SNAPSHOT_INTERVAL, NUM_SNAPSHOTS * SNAPSHOT_INTERVAL - t0)
            pad_sig = synth_pad(pad_voicing, pad_dur, sr, velocity=0.55)
            # Wide pan: pad goes to both channels but slightly different gains
            # for a stereo-spread feel (cheap chorus-like effect).
            pad_L = pad_sig * 0.85
            pad_R = pad_sig * 0.78
            i0 = int(t0 * sr)
            i1 = min(total_samples, i0 + len(pad_sig))
            pad_bus[i0:i1, 0] += pad_L[: i1 - i0]
            pad_bus[i0:i1, 1] += pad_R[: i1 - i0]

        # ---- Bass: 2 quarter notes (root, 5th) with velocity dynamics ----
        for beat_i, midi in enumerate([chord[0] - 12, chord[2] - 12]):
            t = t0 + beat_i * BEAT_SEC
            vel = 0.95 if beat_i == 0 else 0.75
            vel *= 1.0 + rng.uniform(-0.04, 0.04)
            sig = synth_bass(midi_to_hz(midi), BEAT_SEC * 0.95, sr, velocity=vel)
            i0 = int(t * sr)
            i1 = min(total_samples, i0 + len(sig))
            # Bass center, very slight L bias.
            bass_bus[i0:i1, 0] += sig[: i1 - i0] * 0.50
            bass_bus[i0:i1, 1] += sig[: i1 - i0] * 0.46

        # ---- Melody: 4 notes per snapshot, shape: strong–pass–chord–lead ----
        # Beat 1 (j=0): a chord ROOT or 5TH near the contour center → strong landing.
        candidates_strong = (
            [chord[0] + o for o in (-12, 0, 12)] +
            [chord[2] + o for o in (-12, 0, 12)]
        )
        n0 = closest(target_center, candidates_strong)

        # Beat 2 (j=1): step UP one scale degree from n0 → passing tone.
        n1 = step_in_scale(n0, +1)

        # Beat 3 (j=2): a chord 3RD or 7TH near contour → adds chord color.
        # If cores are diverging a lot, embellish further with one more step.
        candidates_color = (
            [chord[1] + o for o in (-12, 0, 12)] +
            [chord[3] + o for o in (-12, 0, 12)]
        )
        n2 = closest(target_center, candidates_color)
        if s.cpu_per_max_min > 30.0:
            # Small chromatic-ish lift — but we still snap to scale.
            n2 = step_in_scale(n2, +1)

        # Beat 4 (j=3): LEAD-IN — step toward the next bar's chord root.
        if next_chord is not None:
            target_next = next_chord[0] + (12 if n2 > 70 else 0)
            direction = 1 if target_next > n2 else -1
            n3 = step_in_scale(n2, direction)
        else:
            # Final bar: resolve down to tonic root in the melody octave.
            n3 = chord[0] + 12

        notes_midi = [n0, n1, n2, n3]

        # Smoothing: avoid wide leaps from the previous snapshot's last note.
        if last_lead_midi is not None:
            jump = notes_midi[0] - last_lead_midi
            if abs(jump) > 7:                # if leap > a fifth, drop n0 an octave
                notes_midi[0] -= 12 * (1 if jump > 0 else -1)

        for j, midi in enumerate(notes_midi):
            t = t0 + swing_offset(j)
            vel = velocity_for(j, rng)
            # Eighth-note duration with a hint of legato gap.
            dur = (swing_offset(j + 1) - swing_offset(j)) * 0.9 if j < 3 \
                else (2 * BEAT_SEC - swing_offset(3)) * 0.85
            sig = synth_lead(midi_to_hz(midi), dur, sr, velocity=vel)
            i0 = int(t * sr)
            i1 = min(total_samples, i0 + len(sig))
            # Pan melody slightly right for stereo separation from pad.
            melody_bus[i0:i1, 0] += sig[: i1 - i0] * 0.40
            melody_bus[i0:i1, 1] += sig[: i1 - i0] * 0.55
        last_lead_midi = notes_midi[-1]

        # ---- Drums ----
        drum_gain = float(np.clip(0.7 + 0.3 * s.load_1m, 0.7, 1.4))

        # Kick on each beat of the snapshot.
        for beat_i in range(2):
            t = t0 + beat_i * BEAT_SEC
            vel = 0.95 if beat_i == 0 else 0.85
            vel *= 1.0 + rng.uniform(-0.05, 0.05)
            i0 = int(t * sr)
            i1 = min(total_samples, i0 + len(kick_buf))
            sig = kick_buf[: i1 - i0] * (0.55 * drum_gain * vel)
            drum_bus[i0:i1, 0] += sig
            drum_bus[i0:i1, 1] += sig

        # Snare on beat 2 of the snapshot — backbeat.
        t = t0 + BEAT_SEC
        i0 = int(t * sr)
        i1 = min(total_samples, i0 + len(snare_buf))
        sig = snare_buf[: i1 - i0] * (0.40 * drum_gain
                                      * (1.0 + rng.uniform(-0.05, 0.05)))
        drum_bus[i0:i1, 0] += sig * 0.85
        drum_bus[i0:i1, 1] += sig * 0.95   # snare slightly right

        # Hats: 8ths normally, 16ths on heavy network bursts. Open hat
        # on the "and" of beat 2 every 2 bars for groove.
        hat_step = BEAT_SEC / 4 if s.net_bps > 100_000 else BEAT_SEC / 2
        n_hats = int(round(2 * BEAT_SEC / hat_step))
        for k in range(n_hats):
            t = t0 + k * hat_step
            is_open = (i % 4 == 2 and k == 3)
            hat_dur = 0.18 if is_open else 0.07
            hat_sig = synth_hat(hat_dur, sr, seed=i * 100 + k, open_=is_open)
            vel = 0.55 if (k % 2 == 0) else 0.35
            i0 = int(t * sr)
            i1 = min(total_samples, i0 + len(hat_sig))
            scaled = hat_sig[: i1 - i0] * (0.18 * drum_gain * vel)
            drum_bus[i0:i1, 0] += scaled * 0.85
            drum_bus[i0:i1, 1] += scaled * 1.0   # hats panned slightly right

    # ---- Master mix ----------------------------------------------------------
    print("Mixing...")
    # Pre-balance: pad needs to sit lower than melody to not muddy.
    mix = (melody_bus * 0.90
           + bass_bus * 0.95
           + pad_bus * 0.55
           + drum_bus * 0.85)

    # ---- Reverb (per-channel for natural width) ------------------------------
    print("Adding reverb...")
    L = schroeder_reverb(mix[:, 0].astype(np.float64), sr, seed=0, mix=0.20)
    R = schroeder_reverb(mix[:, 1].astype(np.float64), sr, seed=1, mix=0.20)
    mix = np.column_stack([L, R])

    # ---- Final limiting: normalize peak, then gentle saturation --------------
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.88
    mix = np.tanh(mix * 1.05)
    return mix.astype(np.float32)


# ---- IO ----------------------------------------------------------------------

def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write mono (1D) or stereo (N,2) float audio as 16-bit PCM WAV."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    # numpy's default flatten on (N,2) gives interleaved L0,R0,L1,R1,... which
    # is exactly what the WAV format expects.
    interleaved = pcm.flatten() if channels == 2 else pcm
    with wave.open(path, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(interleaved.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-play", action="store_true",
                        help="Just write the WAV file; don't play it back.")
    parser.add_argument("--out", default="song_v2.wav", help="Output WAV path.")
    args = parser.parse_args()

    snaps = capture()
    print("\nComposing v2 (this takes ~2s for the reverb pass)...")
    t_start = time.monotonic()
    audio = compose(snaps)
    print(f"Render took {time.monotonic() - t_start:.1f}s")

    write_wav(args.out, audio, SAMPLE_RATE)
    duration = audio.shape[0] / SAMPLE_RATE
    print(f"Saved {args.out}  ({duration:.1f}s, {audio.shape[0]} samples × "
          f"{audio.shape[1]} channels @ {SAMPLE_RATE} Hz)")

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
