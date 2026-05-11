"""compose_v3.py — section-based song with real arrangement.

Where v2 played one texture for 30s straight (which is why it sounded
monotonous), v3 treats the song as a sequence of named sections, each
with its OWN drum pattern, bassline, motif, and pad gain. That gives
the listener contrast — the basic ingredient of "exciting" music.

Song structure (15 bars at 120 BPM = 30 s):

    bars  0– 1   intro      no drums; pad fades in; sparse melody
    bars  2– 5   verse      half-time drums; syncopated bass; call-and-
                             response motif (A then B then A then B)
    bars  6– 8   build      16ths hats; climbing motif; tension
    bar      8     ↳ drum fill into chorus (replaces normal pattern)
    bars  9–12   chorus     full drums + crash, octave-doubled lead,
                             busiest bass, all stops out
    bar  13      break      everyone except pad drops; long held note
    bar  14      outro      G7 → Cmaj7 cadence with final crash

Within each section, the workload metrics still color things:

    cpu/100   → bias toward variant motif vs. primary motif on each bar
    cores Δ   → adds a chromatic embellishment on motif beat 3
    mem%>65   → lifts melody up an octave
    swap>5%   → key flips to minor
    disk Δ>1M → tom hit on the bar
    net KB/s  → if >100, hats stay at 16ths even in low-energy sections
    load avg  → drum-bus gain
    procs     → flips motif direction (mirror inversion)

Run:
    python compose_v3.py                # capture, compose, play
    python compose_v3.py --no-play
    python compose_v3.py --out song_v3.wav

Pair with workload.py for a guaranteed-exciting capture:
    python workload.py --render --out song_v3.wav --script compose_v3.py
"""

from __future__ import annotations

import argparse
import time
import wave

import numpy as np
import sounddevice as sd

# Reuse capture + a bunch of v2's synth primitives.
from compose import NUM_SNAPSHOTS, SNAPSHOT_INTERVAL, Snapshot, capture
from compose_v2 import (
    adsr,
    allpass_filter,
    comb_filter,
    lowpass_fir,
    saw_stack,
    schroeder_reverb,
    synth_bass,
    synth_hat,
    synth_kick,
    synth_lead,
    synth_pad,
    synth_snare,
    write_wav,
)


# ---- Constants ---------------------------------------------------------------

SAMPLE_RATE = 44_100
BPM = 120
BEAT_SEC = 60.0 / BPM        # 0.5 s
BAR_SEC = 4 * BEAT_SEC       # 2 s — one 4/4 bar
NUM_BARS = 15                # 15 × 2 s = 30 s

SWING = 0.58                 # for the 8ths within each beat
KEY_ROOT = 60                # MIDI C4 (the tonic of the major key)
SCALE_DEGREES = [0, 2, 4, 5, 7, 9, 11]   # C major / A natural minor


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


# ---- Section layout ---------------------------------------------------------

# (bar_start_inclusive, bar_end_inclusive, section_name)
# Each bar runs for 2 s. Bar indices are 0-based.
SECTIONS = [
    (0,  1,  "intro"),
    (2,  5,  "verse"),
    (6,  8,  "build"),
    (9,  12, "chorus"),
    (13, 13, "break"),
    (14, 14, "outro"),
]


def section_for_bar(bar: int) -> str:
    for b0, b1, name in SECTIONS:
        if b0 <= bar <= b1:
            return name
    return "outro"


# ---- Harmony ----------------------------------------------------------------

# Chord = (root, 3rd, 5th, 7th) in MIDI; voiced in the bass octave (around C4).
CHORDS = {
    "Cmaj7": (60, 64, 67, 71),
    "Am7":   (57, 60, 64, 67),
    "Fmaj7": (53, 57, 60, 64),
    "G7":    (55, 59, 62, 65),
    "Em7":   (52, 55, 59, 62),
    "Dm7":   (50, 53, 57, 60),
}

# One chord per bar. Length must equal NUM_BARS.
# The chorus uses I-V-vi-IV (the "pop chord" progression — strong on purpose).
PROGRESSION_MAJOR = [
    # intro          verse                      build              chorus                       break    outro
    "Cmaj7", "Fmaj7",  "Cmaj7", "Am7", "Fmaj7", "G7",  "Am7", "Fmaj7", "G7",  "Cmaj7", "G7", "Am7", "Fmaj7",  "Fmaj7", "Cmaj7",
]

# A minor variant — swap I↔vi so the song hangs on Am. Selected if swap is used.
PROGRESSION_MINOR = [
    "Am7", "Fmaj7",  "Am7", "Em7", "Fmaj7", "G7",  "Em7", "Fmaj7", "G7",  "Am7", "G7", "Fmaj7", "Em7",  "Fmaj7", "Am7",
]

assert len(PROGRESSION_MAJOR) == NUM_BARS
assert len(PROGRESSION_MINOR) == NUM_BARS


# ---- Scale-step helpers -----------------------------------------------------

def scale_step_from_root(root_midi: int, step: int) -> int:
    """Return the MIDI note `step` scale-degrees from `root_midi`.

    `step` is interpreted in the C-major / A-minor diatonic scale (same
    7-note set). Negative steps go below; large positive steps span octaves.
    Chord-tone steps from any diatonic chord root are 0, 2, 4, 6 — always
    landing on a chord tone (root / 3rd / 5th / 7th). Steps 1, 3, 5 are
    passing tones; 7 is the octave; etc.
    """
    semitone = (root_midi - KEY_ROOT) % 12
    octave_base = root_midi - semitone
    # Find the scale-index of root_midi.
    if semitone in SCALE_DEGREES:
        idx = SCALE_DEGREES.index(semitone)
    else:
        idx = min(range(len(SCALE_DEGREES)),
                  key=lambda i: abs(SCALE_DEGREES[i] - semitone))
    new_idx = idx + step
    octave_offset = (new_idx // len(SCALE_DEGREES)) * 12
    return octave_base + octave_offset + SCALE_DEGREES[new_idx % len(SCALE_DEGREES)]


# ---- Melodic motifs ---------------------------------------------------------
# Each motif is a list of notes within ONE bar (4 beats). Tuple format:
#   (beat_offset_in_bar, scale_step_from_chord_root, duration_beats, velocity)
#
# The same motif gets played over different chords (motif A in bar 2 over
# Cmaj7 sounds different than motif A in bar 3 over Am7) — that's what
# makes a 4-bar phrase feel coherent.

MOTIFS = {
    "intro_A": [
        (0.0, 4, 1.5, 0.55),     # held 5th — sparse, expectant
        (2.5, 2, 1.5, 0.50),
    ],
    "intro_B": [
        (0.0, 2, 1.5, 0.55),
        (2.0, 0, 2.0, 0.60),     # land on root, fade
    ],

    "verse_A": [   # the "hook" — call shape (outlines chord 5–3–5–7–5)
        (0.0, 4, 0.5, 0.85),
        (0.5, 2, 0.5, 0.65),
        (1.0, 4, 1.0, 0.85),
        (2.0, 6, 0.5, 0.80),
        (2.5, 4, 0.5, 0.65),
        (3.0, 2, 1.0, 0.70),
    ],
    "verse_B": [   # response — answer phrase, ends low
        (0.0, 2, 0.5, 0.80),
        (0.5, 4, 0.5, 0.70),
        (1.0, 6, 0.5, 0.85),
        (1.5, 4, 0.5, 0.70),
        (2.0, 2, 0.5, 0.75),
        (2.5, 0, 0.5, 0.70),
        (3.0, 4, 1.0, 0.80),     # lead-in to next bar (5th, held)
    ],

    "build_A": [   # ascending tension
        (0.0, 4, 0.5, 0.90),
        (0.5, 4, 0.5, 0.80),
        (1.0, 6, 0.5, 0.85),
        (1.5, 6, 0.5, 0.75),
        (2.0, 7, 0.5, 0.90),
        (2.5, 7, 0.5, 0.85),
        (3.0, 9, 1.0, 0.90),
    ],
    "build_B": [
        (0.0, 7, 0.5, 0.90),
        (0.5, 4, 0.5, 0.75),
        (1.0, 6, 0.5, 0.85),
        (1.5, 7, 0.5, 0.80),
        (2.0, 9, 1.0, 0.92),
        (3.0, 7, 1.0, 0.85),
    ],

    "chorus_A": [  # busy, octave above the verse
        (0.0, 7, 0.5, 1.00),
        (0.5, 6, 0.25, 0.70),
        (0.75, 4, 0.25, 0.65),
        (1.0, 7, 0.5, 0.95),
        (1.5, 4, 0.5, 0.80),
        (2.0, 6, 0.5, 0.85),
        (2.5, 9, 0.5, 0.90),     # +9 = 2nd above octave (a "9th")
        (3.0, 7, 0.5, 0.85),
        (3.5, 4, 0.5, 0.75),
    ],
    "chorus_B": [
        (0.0, 9, 0.5, 1.00),
        (0.5, 7, 0.5, 0.85),
        (1.0, 6, 0.5, 0.90),
        (1.5, 4, 0.5, 0.80),
        (2.0, 7, 0.25, 0.85),
        (2.25, 6, 0.25, 0.75),
        (2.5, 4, 0.5, 0.70),
        (3.0, 2, 0.5, 0.65),
        (3.5, 0, 0.5, 0.85),     # land on root
    ],

    "break": [    # one held note — silence around it is the drama
        (0.0, 4, 4.0, 0.65),
    ],
    "outro": [    # final cadence to tonic
        (0.0, 2, 1.0, 0.75),
        (1.0, 4, 1.0, 0.80),
        (2.0, 0, 2.0, 0.85),     # land on tonic root, held
    ],
}


def motif_for_bar(bar: int, section: str) -> list:
    """Pick A or B variant — alternate within each section for call-and-response."""
    if section == "break":
        return MOTIFS["break"]
    if section == "outro":
        return MOTIFS["outro"]
    # Within a section, alternate A/B/A/B based on bar parity within the section.
    section_start = next(b0 for (b0, _, name) in SECTIONS if name == section)
    rel = bar - section_start
    variant = "A" if rel % 2 == 0 else "B"
    return MOTIFS[f"{section}_{variant}"]


# ---- Bass patterns ----------------------------------------------------------
# Same tuple shape as motifs but played an octave or two lower.
BASS_PATTERNS = {
    "intro": [(0.0, 0, 4.0, 0.65)],     # one long root, sustained the whole bar

    "verse": [                          # syncopated — leaves space
        (0.0, 0, 0.75, 0.90),           # root, downbeat
        (1.5, 0, 0.5, 0.70),            # syncopated root on the "and" of 2
        (2.0, 4, 1.0, 0.85),            # fifth on 3
        (3.0, 0, 0.5, 0.65),
        (3.5, 2, 0.5, 0.75),            # 3rd as pickup into next bar
    ],

    "build": [                          # 8th-note driving
        (0.0, 0, 0.5, 0.90),
        (0.5, 4, 0.5, 0.80),
        (1.0, 0, 0.5, 0.85),
        (1.5, 4, 0.5, 0.80),
        (2.0, 0, 0.5, 0.90),
        (2.5, 4, 0.5, 0.85),
        (3.0, 7, 0.5, 0.85),            # octave climb
        (3.5, 4, 0.5, 0.75),
    ],

    "chorus": [                         # busiest, with octave jumps
        (0.0, 0, 0.5, 1.00),
        (0.5, 0, 0.25, 0.60),
        (0.75, 0, 0.25, 0.65),
        (1.0, 4, 0.5, 0.85),
        (1.5, 7, 0.5, 0.90),            # octave up — punchy
        (2.0, 0, 0.5, 0.95),
        (2.5, 0, 0.25, 0.60),
        (2.75, 0, 0.25, 0.65),
        (3.0, 4, 0.5, 0.85),
        (3.5, 2, 0.5, 0.70),
    ],

    "break": [(0.0, 0, 4.0, 0.55)],     # very soft sustained root
    "outro": [(0.0, 0, 4.0, 0.80)],     # final root, held
}


# ---- Drum patterns ----------------------------------------------------------
# Each pattern is a dict of (beat_position, velocity) lists per drum.

DRUM_PATTERNS = {
    "intro": {                          # no drums in the intro
        "kick": [], "snare": [], "hat": [], "crash": [],
    },
    "verse": {
        "kick":  [(0.0, 0.95), (2.0, 0.90)],
        "snare": [(1.0, 0.85), (3.0, 0.85)],
        "hat":   [(i * 0.5, 0.45 + (0.15 if i % 2 == 0 else 0)) for i in range(8)],
        "crash": [],
    },
    "build": {
        "kick":  [(0.0, 0.95), (1.5, 0.80), (2.0, 0.90), (3.0, 0.70)],
        "snare": [(1.0, 0.85), (3.0, 0.90)],
        "hat":   [(i * 0.25, 0.40 + (0.20 if i % 2 == 0 else 0)) for i in range(16)],
        "crash": [],
    },
    "chorus": {
        "kick":  [(0.0, 1.00), (1.5, 0.70), (2.0, 0.95), (2.5, 0.65), (3.5, 0.75)],
        "snare": [(1.0, 0.95), (3.0, 1.00)],
        "hat":   [(i * 0.25, 0.45 + (0.20 if i % 2 == 0 else 0)) for i in range(16)],
        "crash": [(0.0, 0.70)],         # crash on every chorus bar's downbeat
    },
    "break": {                          # near-silent — only soft kick on beat 1
        "kick":  [(0.0, 0.55)],
        "snare": [], "hat": [], "crash": [],
    },
    "outro": {                          # final hit
        "kick":  [(0.0, 0.85)],
        "snare": [(0.0, 0.40)],
        "hat":   [],
        "crash": [(0.0, 0.90)],         # closing crash
    },
}


# Pad gain per section. 0 = no pad in this section.
PAD_GAIN = {
    "intro":  0.30,                     # fades in
    "verse":  0.50,
    "build":  0.65,
    "chorus": 0.85,                     # full
    "break":  0.70,                     # sustains while drums drop
    "outro":  0.55,
}


# ---- Extra synth voices -----------------------------------------------------

def synth_crash(dur: float, sr: int, seed: int = 7) -> np.ndarray:
    """Bright noise burst with slow decay. Cymbal-ish."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    noise = np.diff(rng.standard_normal(n), prepend=0.0)     # high-pass-ish
    t = np.arange(n) / sr
    # Slight tremolo gives the noise some shimmer.
    mod = 1.0 + 0.2 * np.sin(2 * np.pi * 11.0 * t)
    return noise * np.exp(-t * 1.8) * mod


def synth_tom(freq: float, dur: float, sr: int) -> np.ndarray:
    """Pitched drum hit — used for fills."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = freq + 30 * np.exp(-t * 25)     # tiny pitch sweep
    phase = 2 * np.pi * np.cumsum(f) / sr
    return np.sin(phase) * np.exp(-t * 9)


# ---- Drum fill (replaces bar 8's normal drums) ------------------------------

def render_drum_fill(out: np.ndarray, t0: float, sr: int, gain: float) -> None:
    """Tom roll ascending across one bar, ending with a snare flam.

    Builds tension going into the chorus on bar 9.
    """
    # Eight tom hits, climbing in pitch.
    for i in range(8):
        t = t0 + i * (BAR_SEC / 8)
        freq = 90 + i * 18                    # roughly G2 climbing
        sig = synth_tom(freq, 0.18, sr)
        i0 = int(t * sr)
        i1 = min(out.shape[0], i0 + len(sig))
        sig = sig[: i1 - i0] * (0.55 + i * 0.04) * gain
        out[i0:i1, 0] += sig
        out[i0:i1, 1] += sig
    # A snare flam on the very last 16th of the bar — sets up the chorus.
    flam_t = t0 + BAR_SEC - 0.06
    snare = synth_snare(0.10, sr)
    i0 = int(flam_t * sr)
    i1 = min(out.shape[0], i0 + len(snare))
    out[i0:i1, 0] += snare[: i1 - i0] * 0.85 * gain
    out[i0:i1, 1] += snare[: i1 - i0] * 0.95 * gain


# ---- Echo / delay effect ----------------------------------------------------

def apply_echo(track: np.ndarray, sr: int, delay_sec: float,
               feedback: float = 0.45, mix: float = 0.32, taps: int = 4) -> np.ndarray:
    """Multi-tap echo on a mono track (sums delayed-and-attenuated copies)."""
    out = track.copy()
    n = len(track)
    for tap in range(1, taps + 1):
        d = int(delay_sec * tap * sr)
        if d >= n:
            break
        gain = (feedback ** tap) * mix
        out[d:] += track[: n - d] * gain
    return out


# ---- Bar-level rendering ----------------------------------------------------

def render_bar(
    out_lead: np.ndarray,
    out_bass: np.ndarray,
    out_pad: np.ndarray,
    out_drum: np.ndarray,
    bar: int,
    section: str,
    chord: tuple[int, int, int, int],
    next_chord_root: int | None,
    snap: Snapshot,
    sr: int,
    rng: np.random.Generator,
    is_drum_fill: bool,
) -> None:
    t_bar = bar * BAR_SEC

    # Workload metric modulations (per-bar):
    mem_octave_lift = 12 if snap.mem > 65 else 0
    procs_invert = (snap.pids % 2 == 1)
    cpu_bias = snap.cpu / 100.0
    spread_chromatic = snap.cpu_per_max_min > 30.0
    busy_hats = snap.net_bps > 100_000
    tom_hit = snap.disk_write_delta > 1_000_000
    drum_gain = float(np.clip(0.7 + 0.3 * snap.load_1m, 0.7, 1.5))

    # ---- Melody (motif notes) ----
    motif = motif_for_bar(bar, section)
    chord_root = chord[0]
    melody_octave_base = 12 + mem_octave_lift           # melody lives an octave above bass
    melody_notes_played: list[tuple[float, int, float, float]] = []

    # Optional second motif variant blending: high CPU pushes toward the "harder" variant.
    # Simple version — we already alternate A/B by bar; nothing extra needed here.

    for k, (beat_off, step, dur_beats, vel) in enumerate(motif):
        # Optionally mirror the motif (flip step around chord root) when procs is odd.
        s = -step if procs_invert and step != 0 else step
        # Optional chromatic embellishment on a middle note when cores diverge.
        chromatic = 1 if (spread_chromatic and k == 2) else 0

        midi = scale_step_from_root(chord_root, s) + melody_octave_base + chromatic
        t = t_bar + beat_off * BEAT_SEC
        dur = dur_beats * BEAT_SEC * 0.92               # tiny gap = legato but distinct
        vel_jit = vel * (1.0 + rng.uniform(-0.04, 0.04))
        sig = synth_lead(midi_to_hz(midi), dur, sr, velocity=vel_jit)
        i0 = int(t * sr)
        i1 = min(out_lead.shape[0], i0 + len(sig))
        # Pan melody slightly right.
        out_lead[i0:i1, 0] += sig[: i1 - i0] * 0.45
        out_lead[i0:i1, 1] += sig[: i1 - i0] * 0.60
        melody_notes_played.append((t, midi, dur, vel_jit))

    # ---- Counter-melody during chorus: octave-doubled lead, quieter ----
    if section == "chorus":
        for (t, midi, dur, vel) in melody_notes_played:
            sig = synth_lead(midi_to_hz(midi + 12), dur, sr, velocity=vel * 0.55)
            i0 = int(t * sr)
            i1 = min(out_lead.shape[0], i0 + len(sig))
            # Counter-line pans slightly left for stereo width.
            out_lead[i0:i1, 0] += sig[: i1 - i0] * 0.42
            out_lead[i0:i1, 1] += sig[: i1 - i0] * 0.30

    # ---- Bass pattern ----
    bass_pattern = BASS_PATTERNS[section]
    for beat_off, step, dur_beats, vel in bass_pattern:
        midi = scale_step_from_root(chord_root, step) - 12      # one octave down
        t = t_bar + beat_off * BEAT_SEC
        dur = dur_beats * BEAT_SEC * 0.92
        vel_jit = vel * (1.0 + rng.uniform(-0.03, 0.03))
        sig = synth_bass(midi_to_hz(midi), dur, sr, velocity=vel_jit)
        i0 = int(t * sr)
        i1 = min(out_bass.shape[0], i0 + len(sig))
        out_bass[i0:i1, 0] += sig[: i1 - i0] * 0.50
        out_bass[i0:i1, 1] += sig[: i1 - i0] * 0.46

    # ---- Pad: one held voicing per bar, gain scaled by section ----
    pad_gain = PAD_GAIN[section]
    if pad_gain > 0.05:
        # Voiced pad: root, 3rd, 5th, 7th, plus 9th up top for color.
        voicing = list(chord) + [scale_step_from_root(chord_root, 8) + 12]
        sig = synth_pad(voicing, BAR_SEC, sr, velocity=pad_gain)
        i0 = int(t_bar * sr)
        i1 = min(out_pad.shape[0], i0 + len(sig))
        # Slight L/R imbalance for stereo spread.
        out_pad[i0:i1, 0] += sig[: i1 - i0] * 0.88
        out_pad[i0:i1, 1] += sig[: i1 - i0] * 0.80

    # ---- Drums ----
    if is_drum_fill:
        # Drum fill replaces the normal pattern for this bar.
        render_drum_fill(out_drum, t_bar, sr, gain=drum_gain)
    else:
        pattern = DRUM_PATTERNS[section]

        # Kicks.
        kick_buf = synth_kick(0.30, sr)
        for beat_off, vel in pattern["kick"]:
            t = t_bar + beat_off * BEAT_SEC
            v = vel * (1.0 + rng.uniform(-0.04, 0.04))
            i0 = int(t * sr)
            i1 = min(out_drum.shape[0], i0 + len(kick_buf))
            sig = kick_buf[: i1 - i0] * (0.55 * drum_gain * v)
            out_drum[i0:i1, 0] += sig
            out_drum[i0:i1, 1] += sig

        # Snares.
        snare_buf = synth_snare(0.28, sr)
        for beat_off, vel in pattern["snare"]:
            t = t_bar + beat_off * BEAT_SEC
            v = vel * (1.0 + rng.uniform(-0.04, 0.04))
            i0 = int(t * sr)
            i1 = min(out_drum.shape[0], i0 + len(snare_buf))
            sig = snare_buf[: i1 - i0] * (0.42 * drum_gain * v)
            out_drum[i0:i1, 0] += sig * 0.85
            out_drum[i0:i1, 1] += sig * 0.98

        # Hi-hats. If the workload metric says network is busy, double 8th hats to 16ths.
        hat_events = pattern["hat"]
        if busy_hats and section in ("verse",):                 # only upgrade where we'd notice
            hat_events = [(i * 0.25, 0.4 + (0.2 if i % 2 == 0 else 0)) for i in range(16)]
        for k, (beat_off, vel) in enumerate(hat_events):
            t = t_bar + beat_off * BEAT_SEC
            sig = synth_hat(0.07, sr, seed=bar * 100 + k)
            v = vel * (1.0 + rng.uniform(-0.05, 0.05))
            i0 = int(t * sr)
            i1 = min(out_drum.shape[0], i0 + len(sig))
            scaled = sig[: i1 - i0] * (0.20 * drum_gain * v)
            out_drum[i0:i1, 0] += scaled * 0.85
            out_drum[i0:i1, 1] += scaled * 1.00                 # hats slightly right

        # Crash cymbals (chorus downbeats + outro hit).
        for beat_off, vel in pattern["crash"]:
            t = t_bar + beat_off * BEAT_SEC
            sig = synth_crash(1.4, sr, seed=bar)
            v = vel * (1.0 + rng.uniform(-0.05, 0.05))
            i0 = int(t * sr)
            i1 = min(out_drum.shape[0], i0 + len(sig))
            scaled = sig[: i1 - i0] * (0.22 * drum_gain * v)
            out_drum[i0:i1, 0] += scaled * 0.90
            out_drum[i0:i1, 1] += scaled * 1.00

        # Tom hit if the workload had a disk-write spike this bar.
        if tom_hit:
            sig = synth_tom(110, 0.30, sr)
            i0 = int(t_bar * sr)
            i1 = min(out_drum.shape[0], i0 + len(sig))
            scaled = sig[: i1 - i0] * 0.45 * drum_gain
            out_drum[i0:i1, 0] += scaled
            out_drum[i0:i1, 1] += scaled


# ---- Compose ----------------------------------------------------------------

def compose(snaps: list[Snapshot]) -> np.ndarray:
    sr = SAMPLE_RATE
    total_samples = int(NUM_BARS * BAR_SEC * sr)

    out_lead = np.zeros((total_samples, 2), dtype=np.float32)
    out_bass = np.zeros((total_samples, 2), dtype=np.float32)
    out_pad = np.zeros((total_samples, 2), dtype=np.float32)
    out_drum = np.zeros((total_samples, 2), dtype=np.float32)

    avg_swap = sum(s.swap for s in snaps) / len(snaps)
    progression = PROGRESSION_MINOR if avg_swap > 5.0 else PROGRESSION_MAJOR
    print(f"\nMode: {'minor' if progression is PROGRESSION_MINOR else 'major'} "
          f"(avg swap = {avg_swap:.1f}%)")

    rng = np.random.default_rng(13)

    # Map each bar to one snapshot. With 15 bars and 30 snapshots, take every
    # other snapshot (bar i ← snap[2*i]) so each bar corresponds to a real
    # 1-second slice of recorded metrics.
    print("\nbar  section    chord      cpu  net    disk  notes")
    print("-" * 65)

    for bar in range(NUM_BARS):
        section = section_for_bar(bar)
        chord_name = progression[bar]
        chord = CHORDS[chord_name]
        snap = snaps[min(2 * bar, len(snaps) - 1)]
        next_root = CHORDS[progression[bar + 1]][0] if bar + 1 < NUM_BARS else None

        # Drum fill on the last bar of the build (bar 8) — lifts into chorus.
        is_fill = (section == "build" and bar == 8)

        render_bar(out_lead, out_bass, out_pad, out_drum,
                   bar=bar, section=section, chord=chord,
                   next_chord_root=next_root, snap=snap, sr=sr,
                   rng=rng, is_drum_fill=is_fill)

        notes_for_log = ("FILL" if is_fill else "")
        print(f"{bar:2d}   {section:10s} {chord_name:8s}  "
              f"{snap.cpu:5.1f}% {snap.net_bps / 1024:6.1f} "
              f"{snap.disk_write_delta / 1024:7.1f}  {notes_for_log}")

    # ---- Master effects ------------------------------------------------------
    print("\nApplying echo to lead bus...")
    # Quarter-note echo on lead (L and R get slightly different times for
    # ping-pong-ish movement).
    out_lead[:, 0] = apply_echo(out_lead[:, 0], sr, delay_sec=BEAT_SEC,
                                feedback=0.40, mix=0.30, taps=4)
    out_lead[:, 1] = apply_echo(out_lead[:, 1], sr, delay_sec=BEAT_SEC * 0.75,
                                feedback=0.42, mix=0.30, taps=4)

    print("Mixing buses...")
    mix = (out_lead * 0.85
           + out_bass * 0.95
           + out_pad * 0.55
           + out_drum * 0.85)

    print("Applying stereo reverb...")
    L = schroeder_reverb(mix[:, 0].astype(np.float64), sr, seed=0, mix=0.22)
    R = schroeder_reverb(mix[:, 1].astype(np.float64), sr, seed=1, mix=0.22)
    mix = np.column_stack([L, R])

    # Per-section gain riding for dynamics (intro/break softer than chorus).
    # We taper the master gain by bar so the chorus actually sounds louder.
    print("Riding section gains...")
    gain_curve = np.ones(total_samples, dtype=np.float64)
    for bar in range(NUM_BARS):
        section = section_for_bar(bar)
        bar_gain = {"intro": 0.55, "verse": 0.85, "build": 0.95,
                    "chorus": 1.00, "break": 0.65, "outro": 0.80}[section]
        i0 = int(bar * BAR_SEC * sr)
        i1 = int((bar + 1) * BAR_SEC * sr)
        # Smooth ramp toward bar_gain across the bar so transitions don't click.
        gain_curve[i0:i1] = np.linspace(gain_curve[max(i0 - 1, 0)], bar_gain,
                                        i1 - i0, endpoint=True)
    mix[:, 0] *= gain_curve
    mix[:, 1] *= gain_curve

    # Final limit: normalize, soft-clip.
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.90
    mix = np.tanh(mix * 1.05)
    return mix.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-play", action="store_true",
                        help="Just write the WAV file; don't play it back.")
    parser.add_argument("--out", default="song_v3.wav", help="Output WAV path.")
    args = parser.parse_args()

    snaps = capture()
    print("\nComposing v3...")
    t0 = time.monotonic()
    audio = compose(snaps)
    print(f"Render took {time.monotonic() - t0:.1f}s")

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
