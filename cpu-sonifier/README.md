# cpu-sonifier

Three small toys that turn live system metrics into sound.

| Script | What it does | Output |
|---|---|---|
| `sonify.py` | Streams continuous ambient tones driven by **live** CPU/mem/net. | Plays in real time until Ctrl+C |
| `compose.py` | Captures 30s of metric snapshots, then renders a **30s mono song**. | `song.wav` |
| `compose_v2.py` | Same idea, but with detuned-saw lead, chord pad, swing, reverb, and stereo. | `song_v2.wav` |

## Setup

```bash
cd cpu-sonifier
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested on macOS (Python 3.11+). On first run, the OS may ask the terminal for
permission to use audio output.

## Usage

### Live ambient mode (`sonify.py`)

Continuous tone, changes as you use your machine. Ctrl+C to quit.

```bash
.venv/bin/python sonify.py
```

Mappings:
- **CPU%** → melody pitch (low CPU = low note, C-major pentatonic)
- **Memory%** → low drone volume
- **Net bytes/sec** → noise "tick" on bursts

### Render a song (`compose.py`)

Captures one snapshot per second for 30 seconds, then composes an offline
song at 120 BPM, 4/4, in C major (or A minor if swap is in use).

```bash
.venv/bin/python compose.py             # capture + auto-play
.venv/bin/python compose.py --no-play   # write song.wav, skip playback
.venv/bin/python compose.py --out my.wav
```

Mappings:
- **CPU%** → which arpeggio pattern over the chord
- **Per-core spread** → adds a passing tone on beat 3
- **Memory%** → melody octave (>65% lifts it up an octave)
- **Swap > 0** → switches to minor progression
- **Net bytes/sec** → 16th-note hi-hats on bursts
- **Disk write delta** → tom hit on busy snapshots
- **Load avg** → drum kit gain
- **Process count** → arpeggio direction (even = up, odd = down)

### Render a polished song (`compose_v2.py`)

Same capture phase, but the synthesis and composition are richer:

```bash
.venv/bin/python compose_v2.py             # capture + auto-play stereo
.venv/bin/python compose_v2.py --no-play
```

What v2 adds on top of v1:

- **Sound design**: 3-saw detuned lead + sub-sine, sustained chord *pad* on
  every bar, stereo output, per-channel Schroeder reverb (4 comb + 2 allpass
  filters), gentle saturation on the master.
- **Composition**: 7th-chord progression (Cmaj7 → Am7 → Fmaj7 → G7),
  phrase contour that arcs from intro → chorus → outro across the 30s,
  swung 8ths (~58/42 shuffle), velocity dynamics, and a melody that walks
  through scale passing tones with a deliberate lead-in step into each new
  chord (rather than restarting an arpeggio every bar).

## Playback after the fact

```bash
afplay song.wav         # macOS
afplay song_v2.wav
```

## Tips for varying the song

While `compose.py` / `compose_v2.py` is in its 30s capture phase, run
something CPU-heavy or network-heavy in another terminal to color the
result:

```bash
yes > /dev/null            # pin a CPU core
curl -s -o /dev/null https://speed.cloudflare.com/__down?bytes=100000000   # net spike
```

## Files

```
sonify.py          live ambient toy
compose.py         v1: 30s mono song
compose_v2.py      v2: 30s stereo song with reverb + melody
requirements.txt   sounddevice, numpy, psutil
song.wav           last v1 render
song_v2.wav        last v2 render
```
