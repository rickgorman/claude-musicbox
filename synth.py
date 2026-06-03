#!/usr/bin/env python3
"""Dynamic sox synthesis engine (Strudel/superdough + trance/rave flavored).

Each voice is synthesized at render time with a patch whose timbre is modulated
by affect (cutoff/drive/reverb/attack/delay scale with intensity & valence).

Melodic/pad patches:  supersaw, acid, bell, glass, pluck
Trance/rave patches:  hoover (rave stab), bass (rolling), kick (4-on-floor),
                      hat (offbeat), riser (the rush)

Per-patch tails keep arps/drums tight and pads/bells lush. Voices are cached to
disk keyed by (patch, midi, duration, intensity, valence) buckets so a warm
session is mostly cache hits. Velocity is applied at mix time.
"""

import array
import hashlib
import os
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "voices")
SAMPLE_RATE = 44100

os.makedirs(CACHE_DIR, exist_ok=True)

# Ring time after the note-off, per patch. Short = tight (arps/drums).
PATCH_TAIL_MS = {
    "supersaw": 900, "acid": 320, "bell": 900, "glass": 1100, "pluck": 240,
    "hoover": 420, "bass": 130, "kick": 150, "hat": 60, "riser": 250,
    "snare": 180, "clap": 220, "sub": 150, "keys": 700, "blip": 160,
    "pad_warm": 500, "pad_dark": 450, "pad_glass": 600,
    "pad_hollow": 420, "pad_soft": 500,
    "chirp": 100, "fall": 100, "coo": 220, "whoop": 200, "hoot": 280,
    "growl": 260, "purr": 320, "croak": 140, "snarl": 200,
    "rumble": 300, "whimper": 140,
}
DEFAULT_TAIL_MS = 500
PATCHES = tuple(PATCH_TAIL_MS.keys())
UNPITCHED = {"kick", "hat", "snare", "clap", "purr"}


def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _params(intensity, valence, length, tail_ms):
    pos = max(0.0, valence)
    return {
        "cutoff":  int(_clamp(1300 + 4200 * intensity + 800 * pos, 500, 7200)),
        "drive":   round(8 + 22 * intensity, 1),
        "reverb":  int(_clamp(70 - 42 * intensity, 18, 75)),
        "attack":  round(_clamp(min(0.30 - 0.28 * intensity, length * 0.22),
                                0.004, 0.32), 3),
        "delay":   int(_clamp(170 - 80 * intensity, 70, 180)),
        "bcut":    int(_clamp(460 + 520 * intensity, 400, 1100)),
        "release": round(max(0.04, min(length * 0.7, tail_ms / 1000.0)), 3),
    }


def _chain(patch, freq, length, p):
    f = f"{freq:.2f}"
    L = f"{length:.3f}"
    if patch == "supersaw":
        return ["synth", L, "saw", f,
                "chorus", "0.6", "0.9", "50", "0.4", "0.25", "2", "-t",
                "chorus", "0.5", "0.8", "60", "0.3", "0.4", "2.3", "-s",
                "lowpass", str(p["cutoff"]), "reverb", str(p["reverb"]),
                "fade", "t", str(p["attack"]), L, str(p["release"]), "gain", "-6"]
    if patch == "acid":
        return ["synth", L, "saw", f,
                "overdrive", str(p["drive"]), "20",
                "phaser", "0.7", "0.7", "3", "0.5", "1", "-t",
                "lowpass", str(p["cutoff"]),
                "echos", "0.8", "0.7", str(p["delay"]), "0.55",
                "fade", "t", "0.005", L, str(p["release"]), "gain", "-8"]
    if patch == "bell":
        return ["synth", L, "pluck", f,
                "reverb", str(p["reverb"]),
                "echos", "0.8", "0.7", str(p["delay"]), "0.4",
                str(int(p["delay"] * 1.6)), "0.25",
                "lowpass", str(min(p["cutoff"] + 800, 8000)),
                "fade", "t", "0.004", L, str(p["release"]), "gain", "-5"]
    if patch == "glass":
        return ["synth", L, "triangle", f,
                "phaser", "0.6", "0.66", "3", "0.6", "0.5", "-t",
                "lowpass", str(p["cutoff"]), "reverb", str(p["reverb"]),
                "fade", "t", str(p["attack"]), L, str(p["release"]), "gain", "-5"]
    if patch == "pluck":
        return ["synth", L, "saw", f,
                "lowpass", str(p["cutoff"]),
                "echos", "0.7", "0.6", str(p["delay"]), "0.5",
                "fade", "t", "0.002", L, str(p["release"]), "gain", "-7"]
    if patch == "hoover":
        return ["synth", L, "saw", f,
                "bend", f"0.05,-60,{length * 0.7:.3f}",
                "overdrive", str(p["drive"]), "10",
                "chorus", "0.6", "0.9", "55", "0.4", "0.3", "2", "-t",
                "lowpass", str(p["cutoff"]), "reverb", "22",
                "fade", "t", "0.005", L, str(p["release"]), "gain", "-8"]
    if patch == "bass":
        return ["synth", L, "saw", f,
                "lowpass", str(p["bcut"]), "overdrive", "5",
                "fade", "t", "0.002", L, str(p["release"]), "gain", "-3"]
    if patch == "kick":
        return ["synth", L, "sine", "162:50",
                "fade", "t", "0", L, f"{length * 0.9:.3f}",
                "overdrive", "6", "gain", "-2"]
    if patch == "hat":
        return ["synth", L, "whitenoise", "highpass", "7000",
                "fade", "t", "0", L, f"{length * 0.9:.3f}", "gain", "-11"]
    if patch == "riser":
        return ["synth", L, "saw", f"{freq:.2f}:{freq * 3.5:.2f}",
                "lowpass", "4500", "reverb", "60",
                "fade", "t", f"{length * 0.8:.3f}", L, "0.05", "gain", "-9"]
    if patch == "snare":
        return ["synth", L, "whitenoise", "bandpass", "1700", "700",
                "fade", "t", "0", L, f"{length * 0.9:.3f}", "gain", "-7"]
    if patch == "clap":
        return ["synth", L, "whitenoise", "bandpass", "1200", "800",
                "echos", "0.8", "0.7", "10", "0.6", "18", "0.4",
                "fade", "t", "0", L, f"{length * 0.9:.3f}", "gain", "-8"]
    if patch == "sub":
        return ["synth", L, "sine", f, "lowpass", "230",
                "fade", "t", "0.005", L, str(p["release"]), "gain", "-2"]
    if patch == "keys":
        return ["synth", L, "pluck", f,
                "chorus", "0.5", "0.9", "50", "0.4", "0.25", "2", "-t",
                "tremolo", "5", "40", "lowpass", "2600", "reverb", "30",
                "fade", "t", "0.01", L, str(p["release"]), "gain", "-6"]
    if patch == "blip":
        return ["synth", L, "square", f, "lowpass", str(p["cutoff"]),
                "fade", "t", "0.002", L, str(p["release"]), "gain", "-9"]
    if patch == "pad_warm":
        return ["synth", L, "saw", f,
                "chorus", "0.6", "0.9", "50", "0.4", "0.25", "2", "-t",
                "chorus", "0.5", "0.8", "60", "0.3", "0.4", "2.3", "-s",
                "lowpass", str(p["cutoff"]), "reverb", str(p["reverb"]),
                "fade", "t", str(max(p["attack"], 0.05)), L, str(p["release"]),
                "gain", "-6"]
    if patch == "pad_dark":
        return ["synth", L, "saw", f,
                "overdrive", "8", "12",
                "chorus", "0.5", "0.9", "60", "0.35", "0.2", "2.5", "-t",
                "lowpass", str(max(700, p["cutoff"] - 2600)),
                "reverb", str(min(85, p["reverb"] + 12)),
                "fade", "t", str(max(p["attack"], 0.06)), L, str(p["release"]),
                "gain", "-7"]
    if patch == "pad_glass":
        return ["synth", L, "triangle", f,
                "phaser", "0.6", "0.66", "3", "0.6", "0.5", "-t",
                "lowpass", str(min(8000, p["cutoff"] + 1200)),
                "reverb", "75",
                "fade", "t", str(max(p["attack"], 0.08)), L, str(p["release"]),
                "gain", "-5"]
    if patch == "pad_hollow":
        return ["synth", L, "square", f,
                "chorus", "0.6", "0.9", "55", "0.4", "0.3", "2", "-t",
                "lowpass", str(max(900, p["cutoff"] - 1200)),
                "reverb", str(p["reverb"]),
                "fade", "t", str(max(p["attack"], 0.04)), L, str(p["release"]),
                "gain", "-9"]
    if patch == "pad_soft":
        return ["synth", L, "triangle", f,
                "chorus", "0.4", "0.8", "45", "0.3", "0.2", "1.8", "-t",
                "lowpass", str(max(800, p["cutoff"] - 1500)),
                "reverb", str(min(80, p["reverb"] + 8)),
                "fade", "t", str(max(p["attack"], 0.10)), L, str(p["release"]),
                "gain", "-6"]
    # ---- creature patches: glides + breath noise + AM roughness ----
    # `brownnoise remix 1,2vX` layers breath under the tone; fade h = organic
    if patch == "chirp":
        return ["synth", L, "sine", f"{freq * 0.72:.1f}:{freq * 1.28:.1f}",
                "brownnoise", "remix", "1,2v0.10", "lowpass", "6000",
                "fade", "h", "0.005", L, str(p["release"]), "gain", "-5"]
    if patch == "fall":
        return ["synth", L, "sine", f"{freq * 1.25:.1f}:{freq * 0.78:.1f}",
                "brownnoise", "remix", "1,2v0.10", "lowpass", "6000",
                "fade", "h", "0.005", L, str(p["release"]), "gain", "-5"]
    if patch == "coo":
        return ["synth", L, "triangle", f"{freq:.1f}:{freq * 0.90:.1f}",
                "brownnoise", "remix", "1,2v0.08",
                "tremolo", "6", "30", "tremolo", "4.3", "12", "lowpass", "1400",
                "fade", "h", "0.02", L, str(p["release"]), "gain", "-4"]
    if patch == "whoop":
        return ["synth", L, "sine", f"{freq * 0.60:.1f}:{freq * 1.75:.1f}",
                "brownnoise", "remix", "1,2v0.12", "lowpass", "5000",
                "fade", "h", "0.012", L, str(p["release"]), "gain", "-4"]
    if patch == "hoot":
        return ["synth", L, "sine", f"{freq:.1f}:{freq * 0.94:.1f}",
                "brownnoise", "remix", "1,2v0.10",
                "tremolo", "5", "12", "tremolo", "3.7", "10", "lowpass", "1100",
                "fade", "h", "0.035", L, str(p["release"]), "gain", "-4"]
    if patch == "growl":
        return ["synth", L, "saw", f,
                "brownnoise", "remix", "1,2v0.30",
                "tremolo", "28", "75", "tremolo", "5.1", "18",
                "overdrive", "18", "15", "lowpass", "650",
                "fade", "h", "0.015", L, str(p["release"]), "gain", "-6"]
    if patch == "purr":
        return ["synth", L, "brownnoise",
                "tremolo", "23", "85", "tremolo", "4.6", "15", "lowpass", "420",
                "fade", "h", "0.05", L, str(p["release"]), "gain", "-8"]
    if patch == "croak":
        return ["synth", L, "square", f,
                "brownnoise", "remix", "1,2v0.18",
                "tremolo", "16", "65", "bandpass", "600", "500",
                "overdrive", "8", "12",
                "fade", "h", "0.004", L, str(p["release"]), "gain", "-7"]
    if patch == "snarl":
        return ["synth", L, "saw", f,
                "brownnoise", "remix", "1,2v0.30",
                "overdrive", "28", "18", "tremolo", "33", "85",
                "bandpass", "800", "800",
                "fade", "h", "0.006", L, str(p["release"]), "gain", "-7"]
    if patch == "rumble":
        return ["synth", L, "saw", f,
                "brownnoise", "remix", "1,2v0.15",
                "tremolo", "17", "40", "tremolo", "4.2", "14", "lowpass", "320",
                "fade", "h", "0.03", L, str(p["release"]), "gain", "-7"]
    if patch == "whimper":
        return ["synth", L, "sine", f"{freq * 1.06:.1f}:{freq * 0.88:.1f}",
                "brownnoise", "remix", "1,2v0.10",
                "tremolo", "9", "35", "lowpass", "2500",
                "fade", "h", "0.01", L, str(p["release"]), "gain", "-7"]
    return _chain("supersaw", freq, length, p)


def _bucket(intensity, valence, dur_ms, detune):
    return (round(intensity * 10), round((valence + 1) * 10), round(dur_ms / 50),
            int(detune / 5))


def _cache_path(patch, midi, intensity, valence, dur_ms, detune=0):
    key = (patch, midi) + _bucket(intensity, valence, dur_ms, detune)
    digest = hashlib.md5(repr(key).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{patch}_{midi}_{digest}.wav")


def _read_wav(path):
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
    samples = array.array("h")
    samples.frombytes(frames)
    return samples


def _synthesize(patch, midi, intensity, valence, dur_ms, path, detune=0):
    tail = min(PATCH_TAIL_MS.get(patch, DEFAULT_TAIL_MS), max(80, int(dur_ms * 1.5)))
    length = (dur_ms + tail) / 1000.0
    freq = midi_to_freq(midi if midi else 60) * (2.0 ** (detune / 1200.0))
    p = _params(intensity, valence, length, tail)
    cmd = ["sox", "-n", "-r", str(SAMPLE_RATE), "-b", "16", "-c", "1",
           "-e", "signed-integer", path] + _chain(patch, freq, length, p)
    subprocess.run(cmd, capture_output=True)


_MEM = {}


def voice(patch, midi, intensity, valence, dur_ms, detune=0):
    if patch not in PATCHES:
        patch = "supersaw"
    if patch in UNPITCHED:
        midi = 0
    path = _cache_path(patch, midi, intensity, valence, dur_ms, detune)
    if path in _MEM:
        return _MEM[path]
    if not os.path.exists(path):
        _synthesize(patch, midi, intensity, valence, dur_ms, path, detune)
    samples = _read_wav(path)
    _MEM[path] = samples
    return samples


def prewarm(events, intensity, valence):
    """Synthesize any uncached voices for these events in parallel."""
    missing = []
    seen = set()
    for ev in events:
        patch = ev["inst"] if ev["inst"] in PATCHES else "supersaw"
        midi = 0 if patch in UNPITCHED else ev["midi"]
        detune = ev.get("detune", 0)
        path = _cache_path(patch, midi, intensity, valence, ev["dur_ms"], detune)
        if path in seen or path in _MEM or os.path.exists(path):
            continue
        seen.add(path)
        missing.append((patch, midi, ev["dur_ms"], detune, path))
    if not missing:
        return
    with ThreadPoolExecutor(max_workers=12) as pool:
        for patch, midi, dur_ms, detune, path in missing:
            pool.submit(_synthesize, patch, midi, intensity, valence, dur_ms,
                        path, detune)
