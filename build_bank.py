#!/usr/bin/env python3
"""Pre-render a note bank with sox. One-time install step.

Each note is rendered long (it can ring); the runtime renderer slices and
envelopes it to the event duration. Realism comes from sox effects baked in
here once, so the hot path stays a pure-stdlib sample mix.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BANK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes")

SAMPLE_RATE = 44100
RENDER_SECONDS = 1.9
MIDI_LOW = 45
MIDI_HIGH = 84  # inclusive

# Each instrument is a sox effect chain template applied after `synth`.
# {f} = fundamental frequency in Hz, {d} = render seconds.
INSTRUMENTS = {
    "string": "synth {d} pluck {f} fade h 0 {d} {rel} lowpass 6000 reverb 25 vol 0.55",
    "bell":   "synth {d} sine {f} fade t 0.008 {d} {d} reverb 55 lowpass 9000 vol 0.42",
    "pad":    "synth {d} triangle {f} fade t 0.10 {d} {pad_rel} lowpass 3000 reverb 60 vol 0.38",
}


def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def render_one(instrument, chain, midi):
    freq = midi_to_freq(midi)
    out = os.path.join(BANK_DIR, instrument, f"{midi}.wav")
    effects = chain.format(
        f=f"{freq:.3f}",
        d=RENDER_SECONDS,
        rel=RENDER_SECONDS * 0.7,
        pad_rel=RENDER_SECONDS * 0.85,
    ).split()
    cmd = [
        "sox", "-n",
        "-r", str(SAMPLE_RATE), "-b", "16", "-c", "1",
        "-e", "signed-integer",
        out, *effects,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return f"FAIL {instrument} {midi}: {result.stderr.strip()}"
    return None


def main():
    jobs = []
    for instrument, chain in INSTRUMENTS.items():
        os.makedirs(os.path.join(BANK_DIR, instrument), exist_ok=True)
        for midi in range(MIDI_LOW, MIDI_HIGH + 1):
            jobs.append((instrument, chain, midi))

    print(f"Rendering {len(jobs)} notes -> {BANK_DIR}")
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for err in pool.map(lambda j: render_one(*j), jobs):
            if err:
                errors.append(err)

    if errors:
        print(f"{len(errors)} failures:", file=sys.stderr)
        for e in errors[:10]:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print("Bank built.")


if __name__ == "__main__":
    main()
