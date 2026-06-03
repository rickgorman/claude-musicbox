#!/usr/bin/env python3
"""Session-music black box.

Pipeline:
  text (last ~20 lines)  -->  compose()  -->  phrase (note events)
  phrase                 -->  render()   -->  out.wav  -->  afplay

compose() is deterministic: same text -> same music (session identity).
A coarse *class* (error / success / question / code / status) picks the
semantic frame. Two continuous affect dimensions then modulate everything
*within* that class:

  valence   in [-1, 1]  negative..positive  -> scale brightness, instrument,
                                               cadence resolution, chord color
  intensity in [0, 1]   calm..intense       -> duration (inverse), density,
                                               register, articulation, dynamics

Both come from a bag-of-words pass over the buffer (v1). render() mixes
pre-rendered bank notes in pure stdlib so the hot path stays fast.
"""

import argparse
import array
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_DIR = os.path.join(HERE, "notes")
SAMPLE_RATE = 44100

sys.path.insert(0, HERE)
import embedder  # noqa: E402
import synth  # noqa: E402

ENGINE = os.environ.get("MUSICBOX_ENGINE", "synth")
MODE = os.environ.get("MUSICBOX_MODE", "vocab")
PATCH_TO_BANK = {"supersaw": "pad", "glass": "pad", "acid": "string",
                 "pluck": "string", "bell": "bell"}

# ---------------------------------------------------------------------------
# Music theory tables
# ---------------------------------------------------------------------------

SCALES = {
    "phrygian":        [0, 1, 3, 5, 7, 8, 10],
    "harmonic_minor":  [0, 2, 3, 5, 7, 8, 11],
    "natural_minor":   [0, 2, 3, 5, 7, 8, 10],
    "dorian":          [0, 2, 3, 5, 7, 9, 10],
    "major":           [0, 2, 4, 5, 7, 9, 11],
    "lydian":          [0, 2, 4, 6, 7, 9, 11],
    "whole_tone":      [0, 2, 4, 6, 8, 10],
}

# Dark -> bright. valence indexes into a class-restricted slice of this ladder.
LADDER = ["phrygian", "harmonic_minor", "natural_minor", "dorian", "major", "lydian"]

# Progressions are scale-degree roots (0-indexed into the chosen scale).
PROGRESSIONS = {
    "happy":    [0, 4, 5, 3],
    "resolve":  [1, 4, 0],
    "dark":     [0, 6, 5],
    "tension":  [0, 1],
    "open":     [0, 3],
    "ascend":   [0, 2, 4],
    "question": [0, 1],
}

DURATION_CALM_MS = 1200
DURATION_INTENSE_MS = 200
MAX_PHRASE_TAIL_MS = 400

STYLE_MIN_INTENSITY = 0.30
KICK_MIDI = 36
SNARE_MIDI = 38
HAT_MIDI = 60

PUMP_DEPTH = 0.30          # how far non-kick voices duck on each beat
PUMP_RECOVERY = 0.55       # exponent: lower = snappier recovery
HAAS_SECONDS = 0.012       # inter-channel delay for stereo width
WIDE_PATCHES = {"supersaw", "glass", "bell", "hoover", "riser",
                "pad_warm", "pad_dark", "pad_glass", "pad_hollow", "pad_soft"}
NO_DUCK_PATCHES = {"kick"}

PAD_CALM_MS = 750
PAD_INTENSE_MS = 300
PAD_BY_CLASS = {"success": "pad_warm", "error": "pad_dark",
                "question": "pad_glass", "code": "pad_hollow",
                "status": "pad_soft"}
# urgency shortens, openness lingers
PAD_DUR_FACTOR = {"error": 0.8, "success": 1.0, "code": 0.85,
                  "status": 0.9, "question": 1.25}
PULSE_MIN_INTENSITY = 0.6
PULSE_CLASSES = {"error", "success"}

# class -> framing. valence/intensity choose within these. inst_* are synth patches.
CLASS_CONFIG = {
    "error":    {"ladder": (0, 3), "base": 50, "inst_bright": "acid",     "inst_dark": "acid",
                 "prog_bright": "ascend",   "prog_neutral": "dark",  "prog_dark": "tension", "arp_bias": 0.2},
    "success":  {"ladder": (3, 5), "base": 60, "inst_bright": "bell",     "inst_dark": "supersaw",
                 "prog_bright": "resolve",  "prog_neutral": "happy", "prog_dark": "open",    "arp_bias": 0.5},
    "question": {"ladder": (3, 5), "base": 64, "inst_bright": "glass",    "inst_dark": "glass",
                 "prog_bright": "question", "prog_neutral": "question", "prog_dark": "question", "arp_bias": 0.6},
    "code":     {"ladder": (2, 5), "base": 55, "inst_bright": "pluck",    "inst_dark": "pluck",
                 "prog_bright": "ascend",   "prog_neutral": "open",  "prog_dark": "ascend",  "arp_bias": 0.9},
    "status":   {"ladder": (1, 5), "base": 58, "inst_bright": "supersaw", "inst_dark": "glass",
                 "prog_bright": "happy",    "prog_neutral": "open",  "prog_dark": "dark",    "arp_bias": 0.5},
}

# ---------------------------------------------------------------------------
# Bag-of-words affect lexicons
# ---------------------------------------------------------------------------

POSITIVE = (
    "success", "passed", "passing", "pass", "done", "complete", "completed",
    "fixed", "works", "working", "great", "good", "nice", "ready", "merged",
    "ok", "resolved", "clean", "improved", "optimized", "win", "yes", "perfect",
    "excellent", "smooth", "confirmed", "valid", "approved", "green",
)
NEGATIVE = (
    "error", "fail", "failed", "failure", "exception", "bug", "broken", "crash",
    "denied", "fatal", "cannot", "undefined", "panic", "abort", "rejected",
    "invalid", "missing", "wrong", "blocked", "stuck", "timeout", "conflict",
    "no", "never", "bad", "slow", "leak", "corrupt", "deprecated", "refused",
    "wrestling", "fighting", "struggling", "battling", "nasty", "brutal",
    "beaten", "defeated", "lost", "hopeless",
)
HIGH_AROUSAL = (
    "urgent", "critical", "immediately", "now", "crash", "fatal", "panic",
    "emergency", "blocked", "broken", "fail", "error", "exception", "must",
    "always", "never", "asap", "severe", "alert", "kill", "force", "dangerous",
    "race", "deadlock", "breaking", "destroy", "halt",
)
LOW_AROUSAL = (
    "note", "fyi", "maybe", "perhaps", "consider", "idle", "waiting", "minor",
    "small", "slightly", "gentle", "calm", "eventually", "later", "optional",
    "info", "fine", "quietly", "background", "trivial",
)

ERROR_WORDS = ("error", "fail", "failed", "failure", "exception", "traceback",
               "denied", "fatal", "panic", "abort", "rejected", "✗")
SUCCESS_WORDS = ("success", "passed", "passing", "done", "complete", "fixed",
                 "merged", "works", "ready", "✓")
CODE_TOKENS = ("`", "{", "}", "()", "=>", "::", "def ", "function", "</", "/>", ";")
TOOL_WORDS = ("bash", "edit", "grep", "read", "write", "search", "git ", "npm ", "rspec")

WORD_RE = re.compile(r"[a-z']+")


# ---------------------------------------------------------------------------
# Feature extraction + affect
# ---------------------------------------------------------------------------

def extract_features(text):
    lower = text.lower()
    words = WORD_RE.findall(lower)
    bag = {}
    for w in words:
        bag[w] = bag.get(w, 0) + 1

    def count(lexicon):
        return sum(bag.get(w, 0) for w in lexicon if " " not in w)

    alpha = [c for c in text if c.isalpha()]
    caps_ratio = (sum(1 for c in alpha if c.isupper()) / len(alpha)) if len(alpha) > 20 else 0.0

    return {
        "pos":         count(POSITIVE),
        "neg":         count(NEGATIVE),
        "high":        count(HIGH_AROUSAL),
        "low":         count(LOW_AROUSAL),
        "errors":      sum(lower.count(w) for w in ERROR_WORDS),
        "success":     sum(lower.count(w) for w in SUCCESS_WORDS),
        "code":        sum(lower.count(t) for t in CODE_TOKENS),
        "tools":       sum(lower.count(w) for w in TOOL_WORDS),
        "exclaims":    text.count("!"),
        "caps_ratio":  caps_ratio,
        "is_question": text.strip().endswith("?"),
        "length":      len(text.strip()),
    }


def valence_score(f):
    raw = (f["pos"] - f["neg"]) / (f["pos"] + f["neg"] + 2.0)
    return max(-1.0, min(1.0, raw * 1.4))


def intensity_score(f):
    score = 0.32
    score += 0.12 * min(f["exclaims"], 4)
    score += 0.60 * f["caps_ratio"]
    score += 0.10 * min(f["high"], 4)
    score -= 0.10 * min(f["low"], 4)
    score += 0.04 * min(f["neg"], 5)
    score -= 0.12 if f["is_question"] else 0.0
    score -= 0.10 if f["length"] > 400 else 0.0
    return max(0.0, min(1.0, score))


def decide_class(f, valence):
    if f["errors"] >= 1 and valence < 0.1:
        return "error"
    if f["success"] >= 1 and valence > 0.0:
        return "success"
    if f["is_question"]:
        return "question"
    if f["code"] >= 3 or f["tools"] >= 2:
        return "code"
    return "status"


# ---------------------------------------------------------------------------
# Embedding analysis (semantic retrieval + affect projection), bow fallback
# ---------------------------------------------------------------------------

_ANCHOR_TABLE = None


def anchor_table():
    global _ANCHOR_TABLE
    if _ANCHOR_TABLE is not None:
        return _ANCHOR_TABLE
    try:
        with open(os.path.join(HERE, "anchors.embedded.json")) as fh:
            _ANCHOR_TABLE = json.load(fh)
    except Exception:
        _ANCHOR_TABLE = {}
    return _ANCHOR_TABLE


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cosine(a, b):
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return -1.0
    return _dot(a, b) / (na * nb)


def _rescale(raw, lo, hi, out_lo, out_hi):
    if hi == lo:
        return (out_lo + out_hi) / 2.0
    t = max(0.0, min(1.0, (raw - lo) / (hi - lo)))
    return out_lo + (out_hi - out_lo) * t


def analyze(text):
    """Return (class, valence, intensity, source). Embedding first, bow fallback."""
    table = anchor_table()
    f = extract_features(text)
    if table.get("centroids"):
        vec = embedder.embed(text)
        if vec and len(vec) == table.get("dim"):
            return _embed_analyze(vec, table, f)

    valence = valence_score(f)
    intensity = intensity_score(f)
    return decide_class(f, valence), valence, intensity, "bow"


def _embed_analyze(vec, table, f):
    best = max(table["centroids"], key=lambda c: _cosine(vec, c["centroid"]))
    valence = _rescale(_dot(vec, table["valence_axis"]),
                       table["valence_scale"]["lo"], table["valence_scale"]["hi"],
                       -1.0, 1.0)
    semantic = _rescale(_dot(vec, table["intensity_axis"]),
                        table["intensity_scale"]["lo"], table["intensity_scale"]["hi"],
                        0.0, 1.0)
    # the embedding misses structural shouting (CAPS, !!!) and calm hedges
    # ("minor", "later") — blend the lexical signal in
    intensity = max(0.0, min(1.0, 0.6 * semantic + 0.4 * intensity_score(f)))
    return best["class"], valence, intensity, "embed:" + best["name"]


# ---------------------------------------------------------------------------
# Rich analysis: soft blend, extra axes, texture geometry, confidence
# ---------------------------------------------------------------------------

def _unit(vec):
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _axis_value(vec, table, name, default=0.5):
    axis = table.get("axes", {}).get(name)
    if not axis:
        return default
    return _rescale(_dot(vec, axis["vec"]), axis["lo"], axis["hi"], 0.0, 1.0)


def analyze_rich(text):
    """Everything the vocabulary needs from one embedding pass. No anchor
    classification — affect axes + raw geometry only. Familiarity is filled in
    later from the session's own trajectory."""
    table = anchor_table()
    f = extract_features(text)
    vec = embedder.embed(text) if table.get("axes") else None

    if not vec or len(vec) != table.get("dim"):
        valence = valence_score(f)
        arousal = intensity_score(f)
        return {"source": "bow", "vec": None, "valence": valence,
                "arousal": arousal, "certainty": 0.6, "progress": 0.5,
                "texture": [0.5] * 8, "f": f}

    semantic_valence = _rescale(_dot(vec, table["valence_axis"]),
                                table["valence_scale"]["lo"],
                                table["valence_scale"]["hi"], -1.0, 1.0)
    valence = max(-1.0, min(1.0, 0.65 * semantic_valence + 0.35 * valence_score(f)))
    semantic = _rescale(_dot(vec, table["intensity_axis"]),
                        table["intensity_scale"]["lo"], table["intensity_scale"]["hi"],
                        0.0, 1.0)
    arousal = max(0.0, min(1.0, 0.6 * semantic + 0.4 * intensity_score(f)))

    unit = _unit(vec)
    texture = [(math.tanh(2.5 * _dot(unit, d)) + 1.0) / 2.0
               for d in table.get("texture_bank", [])] or [0.5] * 8

    return {
        "source": "embed", "vec": vec, "valence": valence, "arousal": arousal,
        "certainty": _axis_value(vec, table, "certainty"),
        "progress": _axis_value(vec, table, "progress"),
        "texture": texture, "f": f,
    }


# ---------------------------------------------------------------------------
# Trajectory: tiny per-session state of recent embeddings
# ---------------------------------------------------------------------------

STATE_PATH = os.path.join(HERE, "state.json")
TRAJECTORY_KEEP = 8


def update_trajectory(session, vec, valence):
    """Returns (movement 0..1, valence_trend, looping, history_sim).
    history_sim is the max cosine to this session's recent states (None when
    there is no history) — familiarity relative to the session's own work
    rather than to any curated list. Appends current state."""
    if vec is None:
        return 0.0, 0.0, False, None
    try:
        with open(STATE_PATH) as fh:
            state = json.load(fh)
    except Exception:
        state = {}
    entries = state.get(session, [])

    movement, trend, looping, history_sim = 0.0, 0.0, False, None
    if entries:
        sims = [_cosine(e["v"], vec) for e in entries[-6:]]
        history_sim = max(sims)
        movement = max(0.0, min(1.0, (1.0 - _cosine(entries[-1]["v"], vec)) * 4.0))
        trend = valence - entries[-1]["val"]
        looping = any(s > 0.88 for s in sims)

    entries.append({"v": vec, "val": valence})
    state[session] = entries[-TRAJECTORY_KEEP:]
    try:
        with open(STATE_PATH, "w") as fh:
            json.dump(state, fh)
    except Exception:
        pass
    return movement, trend, looping, history_sim


# ---------------------------------------------------------------------------
# Vocabulary: need words (motifs) + embedding-driven inflection
# ---------------------------------------------------------------------------

NEEDS = ("halted", "alert", "triumph", "question", "done", "departure", "status")

# (scale_degree, start_units, dur_units, gain)
# Three elaboration tiers per need: deliberation (how much thought the
# utterance asks of the listener) picks the tier. 1-3 notes = glance,
# 3-5 notes = sit down.
MOTIFS_SHORT = {
    "done":      [(2, 0.0, 0.9, 0.9), (0, 1.0, 2.0, 1.0)],
    "halted":    [(0, 0.0, 0.7, 1.0), (3, 0.9, 0.7, 1.0), (6, 2.2, 1.8, 0.5)],
    "question":  [(0, 0.0, 1.0, 0.9), (3, 1.2, 2.0, 1.0)],
    "status":    [(0, 0.0, 1.0, 0.8)],
    "alert":     [(0, 0.0, 1.2, 1.0), (1, 0.0, 1.2, 0.9)],
    "triumph":   [(0, 0.0, 0.8, 0.9), (4, 0.9, 0.8, 1.0), (7, 1.8, 2.2, 1.05)],
    "departure": [(0, 0.0, 1.0, 1.0), (4, 1.1, 2.2, 0.6)],
}
MOTIFS = {
    "done":      [(4, 0.0, 1.4, 1.0), (2, 1.5, 0.8, 0.9), (0, 2.5, 2.6, 1.0)],
    "halted":    [(0, 0.0, 0.7, 1.0), (3, 0.9, 0.7, 1.0),
                  (0, 2.3, 0.7, 0.95), (3, 3.2, 0.7, 0.95), (6, 4.8, 2.2, 0.55)],
    "question":  [(0, 0.0, 1.0, 0.9), (2, 1.2, 1.0, 0.95), (3, 2.4, 2.4, 1.0)],
    "status":    [(0, 0.0, 1.3, 0.8)],
    "alert":     [(0, 0.0, 1.3, 1.0), (1, 0.0, 1.3, 0.9)],
    "triumph":   [(0, 0.0, 0.8, 0.9), (2, 0.9, 0.8, 0.95),
                  (4, 1.8, 0.8, 1.0), (7, 2.7, 2.8, 1.1)],
    "departure": [(0, 0.0, 1.0, 1.0), (2, 1.1, 1.0, 0.8), (4, 2.2, 2.8, 0.6)],
}
MOTIFS_LONG = {
    "done":      [(4, 0.0, 1.4, 1.0), (2, 1.5, 0.8, 0.9), (0, 2.5, 1.8, 1.0),
                  (-3, 4.4, 2.4, 0.6)],
    "halted":    [(0, 0.0, 0.7, 1.0), (3, 0.9, 0.7, 1.0),
                  (0, 2.3, 0.7, 0.95), (3, 3.2, 0.7, 0.95), (6, 4.8, 2.6, 0.55)],
    "question":  [(0, 0.0, 1.0, 0.85), (2, 1.2, 1.0, 0.9), (4, 2.4, 1.0, 0.95),
                  (3, 3.6, 1.0, 0.9), (8, 4.8, 2.8, 1.0)],
    "status":    [(0, 0.0, 0.9, 0.8), (0, 1.2, 0.9, 0.7)],
    "alert":     [(0, 0.0, 1.3, 1.0), (1, 0.0, 1.3, 0.9), (0, 1.6, 2.0, 0.6)],
    "triumph":   [(0, 0.0, 0.8, 0.9), (2, 0.9, 0.8, 0.95), (4, 1.8, 0.8, 1.0),
                  (7, 2.7, 1.6, 1.1), (9, 4.4, 2.2, 0.7)],
    "departure": [(0, 0.0, 1.0, 1.0), (2, 1.1, 1.0, 0.85), (4, 2.2, 1.0, 0.7),
                  (7, 3.3, 2.6, 0.5)],
}

DELIBERATION_BASE = {"status": 0.10, "done": 0.25, "departure": 0.30,
                     "triumph": 0.35, "alert": 0.50, "question": 0.45,
                     "halted": 0.60}
DELIBERATION_SHORT = 0.35
DELIBERATION_LONG = 0.62
TERSE_CHARS = 60


def deliberation_score(text, need, wobble):
    """How much thought this utterance asks of the user."""
    score = DELIBERATION_BASE[need]
    score += 0.20 * wobble
    lower = text.lower()
    score += min(0.2, 0.1 * lower.count(" or "))
    if text.count("?") >= 2:
        score += 0.1
    if len(text) > 600:
        score += 0.1
    if len(text.strip()) < TERSE_CHARS:
        score -= 0.12
    return max(0.0, min(1.0, score))


def pick_motif(need, deliberation):
    if deliberation < DELIBERATION_SHORT:
        return MOTIFS_SHORT[need]
    if deliberation < DELIBERATION_LONG:
        return MOTIFS[need]
    return MOTIFS_LONG[need]

NEED_PATCH = {"alert": "pad_dark", "done": "pad_warm", "triumph": "pad_warm",
              "question": "pad_glass", "halted": "pad_hollow",
              "status": "pad_soft", "departure": "pad_glass"}
NEED_REGISTER = {"alert": 48, "done": 60, "triumph": 62, "question": 66,
                 "halted": 64, "status": 62, "departure": 64}
NEED_LADDER = {"alert": (0, 2), "triumph": (4, 5), "done": (1, 5),
               "question": (3, 5), "halted": (2, 4), "status": (1, 5),
               "departure": (2, 5)}
NEED_BASS_SCALE = {"status": 0.0, "question": 0.35, "departure": 0.5}

UNIT_CALM_MS = 220
UNIT_INTENSE_MS = 90
WOBBLE_THRESHOLD = 0.45
LOW_FAMILIARITY = 0.18
ANALYSIS_TAIL_CHARS = 1000
GRACE_TREND = 0.08
BIG_MOVEMENT = 0.5

HALTED_HINTS = ("permission", "approve", "blocked on you", "waiting for you",
                "need your", "your call", "confirm before")
DEPARTURE_HINTS = ("kicking off", "running in the background", "will take a while",
                   "starting the long", "back in a bit", "going to work on")


def infer_need(text, rich, trend=0.0):
    """Structure first (hints, question mark), then affect thresholds.
    No semantic classification — valence/arousal/trend carry it."""
    lower = text.lower()
    if any(h in lower for h in HALTED_HINTS):
        return "halted"
    if any(h in lower for h in DEPARTURE_HINTS):
        return "departure"
    if text.strip().endswith("?"):
        return "question"

    valence, arousal = rich["valence"], rich["arousal"]
    if valence < -0.15 or (trend < -0.35 and arousal > 0.5):
        return "alert"
    if valence > 0.3 and arousal >= 0.6:
        return "triumph"
    if valence > 0.25:
        return "done"
    if arousal < 0.45:
        return "status"
    return "done"


def degree_to_midi(scale_steps, root_midi, degree):
    n = len(scale_steps)
    return root_midi + 12 * (degree // n) + scale_steps[degree % n]


def build_vocab(text, need, session):
    text = text[-ANALYSIS_TAIL_CHARS:]
    rich = analyze_rich(text)
    rng = seeded_rng(text)
    movement, trend, looping, history_sim = update_trajectory(
        session, rich["vec"], rich["valence"])
    if need is None:
        need = infer_need(text, rich, trend)
    familiarity = (0.6 if history_sim is None
                   else _rescale(history_sim, 0.35, 0.80, 0.0, 1.0))
    t = rich["texture"]
    valence, arousal = rich["valence"], rich["arousal"]

    lo, hi = NEED_LADDER[need]
    scale_name = LADDER[round(lo + (hi - lo) * (valence + 1.0) / 2.0)]
    if familiarity < LOW_FAMILIARITY:
        scale_name = "whole_tone" if "whole_tone" in SCALES else scale_name
    scale_steps = SCALES.get(scale_name, SCALES["major"])

    root_pc = int(hashlib.md5(session.encode()).hexdigest()[:2], 16) % 12
    register_nudge = round((t[6] - 0.5) * 4)
    root_midi = NEED_REGISTER[need] + root_pc - 6 + register_nudge

    patch = NEED_PATCH[need]
    base_gain = 0.6 + 0.3 * arousal
    width = (0.4, 0.7, 0.95)[min(2, int(t[3] * 3))]
    jitter = 4 if t[5] < 0.5 else 14
    wobble = 1.0 - rich["certainty"]
    deliberation = deliberation_score(text, need, wobble)

    # arousal compresses the pulse; deliberation unhurries it
    unit = (UNIT_CALM_MS + (UNIT_INTENSE_MS - UNIT_CALM_MS) * arousal) \
        * (0.9 + 0.3 * deliberation)

    notes = list(pick_motif(need, deliberation))
    if need == "halted" and arousal > 0.7:
        notes = MOTIFS["halted"][:4] + [(0, 4.6, 0.7, 0.9), (3, 5.5, 0.7, 0.9),
                                        (6, 7.1, 2.2, 0.5)]
    if need == "alert" and arousal >= 0.6:
        notes = []
        for k in range(4):
            d = 1.0 if k < 3 else 1.8
            notes += [(0, k * 1.2, d, 1.0 if k in (0, 3) else 0.85),
                      (1, k * 1.2, d, 0.85)]
    if need == "status" and looping:
        notes = [(0, 0.0, 0.9, 0.8), (0, 1.2, 0.9, 0.7)]

    offset = 0.0
    if abs(trend) > GRACE_TREND and need not in ("alert",):
        grace_degree = -1 if trend > 0 else 1
        notes = [(grace_degree, 0.0, 0.35, 0.5)] + \
                [(d, s + 0.5, du, g) for d, s, du, g in notes]
        offset = 0.5
    if movement > BIG_MOVEMENT:
        last = notes[-1]
        notes.append((last[0] + 7, last[1], last[2], 0.5))

    events = []
    span_ms = int(max(s + d for _, s, d, _ in notes) * unit)

    bass_amount = 0.15 + 0.5 * arousal + 0.35 * max(0.0, -valence)
    bass_amount *= NEED_BASS_SCALE.get(need, 1.0)
    bass_amount = max(0.0, min(1.0, bass_amount * (0.85 + 0.3 * t[0])))
    if bass_amount >= 0.2:
        events.append(_event(root_midi - 12, 0, span_ms, patch,
                             base_gain * (0.3 + 0.7 * bass_amount), rng,
                             jitter=jitter))
    if bass_amount >= 0.65:
        events.append(_event(root_midi - 24, 0, span_ms, "sub",
                             base_gain * bass_amount * 0.9, rng, jitter=jitter))

    for i, (degree, start_u, dur_u, gain) in enumerate(notes):
        midi = degree_to_midi(scale_steps, root_midi, degree)
        start = int(start_u * unit)
        dur = int(dur_u * unit)
        pan = _spread_pan(i % 3, 3, width * 0.6)
        if wobble > WOBBLE_THRESHOLD:
            cents = int(10 + 30 * wobble)
            events.append(_event(midi, start, dur, patch, base_gain * gain * 0.62,
                                 rng, pan=pan, jitter=jitter, detune=cents))
            events.append(_event(midi, start, dur, patch, base_gain * gain * 0.62,
                                 rng, pan=-pan, jitter=jitter, detune=-cents))
        else:
            events.append(_event(midi, start, dur, patch, base_gain * gain, rng,
                                 pan=pan, jitter=jitter))

    if need in ("done", "triumph", "question"):
        final_degree, final_start, final_dur, _ = notes[-1]
        chord = diatonic_chord(scale_steps, root_midi, 0, size=3)
        if t[1] > 0.66:
            chord.append(degree_to_midi(scale_steps, root_midi, 8))
        elif t[1] > 0.33:
            chord.append(degree_to_midi(scale_steps, root_midi, 6))
        inversion = int(t[0] * 3.99)
        chord = chord[inversion:] + [m + 12 for m in chord[:inversion]]
        strum = list(enumerate(chord))
        if t[2] < 0.5:
            strum = list(reversed(strum))
        for k, (j, midi) in enumerate(strum):
            events.append(_event(midi, int(final_start * unit) + k * 22,
                                 int(final_dur * unit), patch,
                                 base_gain * 0.45, rng,
                                 pan=_spread_pan(j, len(chord), width),
                                 jitter=jitter))

    total_ms = span_ms
    return {
        "text": text[:200], "source": rich["source"], "need": need,
        "valence": round(valence, 3),
        "intensity": round(arousal, 3),
        "certainty": round(rich["certainty"], 3),
        "progress": round(rich["progress"], 3),
        "familiarity": round(familiarity, 3),
        "movement": round(movement, 3), "trend": round(trend, 3),
        "looping": looping, "wobble": round(wobble, 3),
        "deliberation": round(deliberation, 3), "notes": len(notes),
        "scale": scale_name, "progression": "-", "articulation": "vocab",
        "instrument": patch, "root_midi": root_midi,
        "texture": [round(x, 2) for x in t],
        "total_ms": total_ms, "beat_ms": None, "pump": False,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def seeded_rng(text):
    digest = hashlib.sha256(text.encode("utf-8", "replace")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def diatonic_chord(scale_steps, root_midi, degree, size=3):
    n = len(scale_steps)
    notes = []
    for i in range(size):
        idx = degree + 2 * i
        notes.append(root_midi + 12 * (idx // n) + scale_steps[idx % n])
    return notes


def pick_scale(cfg, valence):
    lo, hi = cfg["ladder"]
    t = (valence + 1.0) / 2.0
    return LADDER[round(lo + (hi - lo) * t)]


def pick_progression(cfg, valence):
    if valence > 0.15:
        return cfg["prog_bright"]
    if valence < -0.15:
        return cfg["prog_dark"]
    return cfg["prog_neutral"]


def pick_articulation(cfg, intensity):
    if intensity > 0.78:
        return "stab"
    if cfg["arp_bias"] >= 0.85 or (cfg["arp_bias"] > 0.4 and intensity > 0.5):
        return "arp"
    if intensity < 0.35:
        return "shimmer"
    return "block_prog"


# Genre arrangements on a 16th-note grid (one bar). Each spec is data; the
# interpreter render_pattern() turns it into events. Adding a genre = a spec.
STYLES = {
    "trance": {
        "bpm": (138, 150), "pump": True, "swing": 0.0,
        "kick": [0, 4, 8, 12], "snare": [], "clap": [], "hat": [2, 6, 10, 14],
        "bass": {"steps": [2, 6, 10, 14], "patch": "bass", "oct": -12},
        "harmony": {"mode": "arp16", "patch": "pluck"},
        "stab": {"steps": [0, 8], "patch": "supersaw"}, "riser": True,
    },
    "french_house": {
        "bpm": (118, 126), "pump": True, "swing": 0.06,
        "kick": [0, 4, 8, 12], "snare": [], "clap": [4, 12], "hat": [2, 6, 10, 14],
        "bass": {"steps": [2, 6, 10, 14], "patch": "bass", "oct": -12},
        "harmony": {"mode": "stab", "patch": "supersaw", "steps": [2, 6, 10, 14]},
        "riser": False,
    },
    "house": {
        "bpm": (120, 126), "pump": True, "swing": 0.04,
        "kick": [0, 4, 8, 12], "snare": [], "clap": [4, 12], "hat": [2, 6, 10, 14],
        "bass": {"steps": [2, 6, 10, 14], "patch": "bass", "oct": -12},
        "harmony": {"mode": "stab", "patch": "keys", "steps": [2, 6, 10, 14]},
        "riser": False,
    },
    "techno": {
        "bpm": (128, 134), "pump": True, "swing": 0.0,
        "kick": [0, 4, 8, 12], "snare": [], "clap": [], "hat": [2, 6, 10, 14],
        "bass": {"steps": [0, 4, 8, 12], "patch": "bass", "oct": -12},
        "harmony": {"mode": "stab", "patch": "hoover", "steps": [0, 8]},
        "riser": False,
    },
    "dnb": {
        "bpm": (168, 174), "pump": False, "swing": 0.0,
        "kick": [0, 6, 10], "snare": [4, 12], "clap": [], "hat": [2, 6, 10, 14],
        "bass": {"steps": [0, 8], "patch": "sub", "oct": -12},
        "harmony": {"mode": "stab", "patch": "hoover", "steps": [0, 8]},
        "riser": False,
    },
    "boom_bap": {
        "bpm": (86, 94), "pump": False, "swing": 0.18,
        "kick": [0, 3, 8], "snare": [4, 12], "clap": [], "hat": [0, 2, 4, 6, 8, 10, 12, 14],
        "bass": {"steps": [0, 8], "patch": "sub", "oct": -12},
        "harmony": {"mode": "chords", "patch": "keys", "steps": [0, 8]},
        "riser": False,
    },
    "synthwave": {
        "bpm": (100, 112), "pump": True, "swing": 0.0,
        "kick": [0, 4, 8, 12], "snare": [4, 12], "clap": [], "hat": [2, 6, 10, 14],
        "bass": {"steps": [0, 3, 8, 11], "patch": "bass", "oct": -12},
        "harmony": {"mode": "arp16", "patch": "blip"},
        "stab": {"steps": [0, 8], "patch": "supersaw"}, "riser": False,
    },
    "lofi": {
        "bpm": (72, 82), "pump": False, "swing": 0.20,
        "kick": [0, 8], "snare": [4, 12], "clap": [], "hat": [2, 6, 10, 14],
        "bass": {"steps": [0, 8], "patch": "sub", "oct": -12},
        "harmony": {"mode": "chords", "patch": "keys", "steps": [0, 8]},
        "riser": False,
    },
}

STYLE_CANDIDATES = {
    ("error", "hi"):   ["techno", "dnb"],
    ("error", "mid"):  ["boom_bap", "techno"],
    ("success", "hi"): ["trance", "french_house", "house"],
    ("success", "mid"): ["synthwave", "french_house", "house"],
    ("code", "hi"):    ["dnb", "techno", "house"],
    ("code", "mid"):   ["boom_bap", "house", "synthwave"],
    ("status", "hi"):  ["house", "techno", "synthwave"],
    ("status", "mid"): ["lofi", "boom_bap", "house", "synthwave"],
}


def pick_style(cls, intensity, rng):
    tier = "hi" if intensity >= 0.6 else "mid"
    candidates = (STYLE_CANDIDATES.get((cls, tier))
                  or STYLE_CANDIDATES.get((cls, "mid")) or ["house"])
    return candidates[rng.randrange(len(candidates))]


def render_pattern(spec, scale_steps, root_midi, degrees, cls, intensity, valence, base_gain, rng):
    """Interpret a genre spec into events. Returns (events, total_ms, beat_ms, pump)."""
    lo, hi = spec["bpm"]
    bpm = lo + (hi - lo) * intensity
    beat_ms = 60000.0 / bpm
    six = beat_ms / 4.0
    swing = spec["swing"] * six
    step_count = next((n for n in (16, 12, 8) if six * n <= 1500), 8)
    total_ms = int(six * step_count)

    chord = diatonic_chord(scale_steps, root_midi, degrees[0], size=4 if valence > 0 else 3)
    pool = chord + [chord[0] + 12, chord[1] + 12]

    def at(step):
        return int(step * six + (swing if step % 2 else 0))

    def grid(lst):
        return [s for s in lst if s < step_count]

    events = []

    for s in grid(spec.get("kick", [])):
        events.append(_event(KICK_MIDI, at(s), int(six * 1.8), "kick", 1.0, rng))
    for s in grid(spec.get("snare", [])):
        events.append(_event(SNARE_MIDI, at(s), int(six * 1.5), "snare", 0.8, rng,
                             pan=rng.uniform(-0.12, 0.12)))
    for s in grid(spec.get("clap", [])):
        events.append(_event(SNARE_MIDI, at(s), int(six * 1.5), "clap", 0.7, rng))
    for k, s in enumerate(grid(spec.get("hat", []))):
        events.append(_event(HAT_MIDI, at(s), 70, "hat", 0.42, rng,
                             pan=(-0.4 if k % 2 else 0.4)))

    bass = spec["bass"]
    for s in grid(bass["steps"]):
        events.append(_event(root_midi + bass["oct"], at(s), int(six * 1.6),
                             bass["patch"], 0.9, rng))

    harmony = spec["harmony"]
    if harmony["mode"] == "arp16":
        for s in range(step_count):
            midi = pool[s % len(pool)]
            accent = 1.0 if s % 4 == 0 else 0.72
            events.append(_event(midi, at(s), int(six * 0.9), harmony["patch"],
                                 base_gain * accent, rng, pan=(-0.55 if s % 2 else 0.55)))
    elif harmony["mode"] == "stab":
        for s in grid(harmony["steps"]):
            for j, midi in enumerate(chord):
                events.append(_event(midi, at(s), int(six * 3.2), harmony["patch"],
                                     base_gain * 0.6, rng, pan=_spread_pan(j, len(chord), 0.8)))
    elif harmony["mode"] == "chords":
        for s in grid(harmony["steps"]):
            for j, midi in enumerate(chord):
                events.append(_event(midi, at(s), int(six * 6), harmony["patch"],
                                     base_gain * 0.7, rng, pan=_spread_pan(j, len(chord), 0.5)))

    stab = spec.get("stab")
    if stab and intensity > 0.55:
        for s in grid(stab["steps"]):
            for j, midi in enumerate(chord):
                events.append(_event(midi, at(s), int(six * 3.5), stab["patch"],
                                     base_gain * 0.5, rng, pan=_spread_pan(j, len(chord), 0.85)))

    if spec.get("riser") and cls == "success" and intensity > 0.6:
        events.append(_event(root_midi, 0, int(total_ms * 0.95), "riser", base_gain * 0.8, rng))

    # resolve on "the one": final downbeat kick + bass root + tonic chord,
    # left to decay naturally
    end = int(step_count * six)
    if spec.get("kick"):
        events.append(_event(KICK_MIDI, end, int(six * 2), "kick", 1.0, rng))
    events.append(_event(root_midi + bass["oct"], end, int(six * 2.5),
                         bass["patch"], 0.95, rng))
    tonic = diatonic_chord(scale_steps, root_midi, 0, size=3)
    end_patch = "pluck" if harmony["patch"] == "blip" else harmony["patch"]
    for j, midi in enumerate(tonic):
        events.append(_event(midi, end, int(six * 3.0), end_patch,
                             base_gain * 0.8, rng, pan=_spread_pan(j, len(tonic), 0.6)))
    total_ms = int(six * (step_count + 3))

    return events, total_ms, beat_ms, spec["pump"]


def build_pad(cls, scale_steps, root_midi, intensity, valence, base_gain, rng):
    """Pad-only mode: one synth pad, tone keyed to class, affect shapes it.
    success/error get a two-chord resolve; question stays suspended. High
    intensity breaks the chord into 4 short pulses (urgency as rhythm)."""
    patch = PAD_BY_CLASS[cls]
    total_ms = int((PAD_CALM_MS + (PAD_INTENSE_MS - PAD_CALM_MS) * intensity)
                   * PAD_DUR_FACTOR[cls])
    size = 4 if valence > 0.2 else 3
    pulsed = intensity >= PULSE_MIN_INTENSITY and cls in PULSE_CLASSES
    if pulsed:
        total_ms = int(total_ms * 1.5)

    tonic = diatonic_chord(scale_steps, root_midi, 0, size=size)

    bass_amount = 0.15 + 0.55 * intensity + 0.35 * max(0.0, -valence)
    if cls == "question":
        bass_amount *= 0.35
    bass_amount = max(0.0, min(1.0, bass_amount * rng.uniform(0.85, 1.15)))

    events = []
    if bass_amount >= 0.2:
        events.append(_event(tonic[0] - 12, 0, total_ms, patch,
                             base_gain * (0.3 + 0.7 * bass_amount), rng))
    if bass_amount >= 0.65:
        events.append(_event(tonic[0] - 24, 0, total_ms, "sub",
                             base_gain * bass_amount * 0.9, rng))

    def chord_at(chord, start, dur, gain):
        for j, midi in enumerate(chord):
            events.append(_event(midi, int(start) + j * 18, int(dur), patch,
                                 gain * (0.85 - 0.04 * j), rng,
                                 pan=_spread_pan(j, len(chord), 0.75)))

    if pulsed and cls == "error":
        step = total_ms / 4.0
        for k in range(4):
            dur = step * 0.62 if k < 3 else step * 1.5
            chord_at(tonic, k * step, dur, base_gain * (1.0 if k in (0, 3) else 0.85))
    elif pulsed and cls == "success":
        pre = diatonic_chord(scale_steps, root_midi, 4, size=size)
        pulse_span = total_ms * 0.55
        step = pulse_span / 4.0
        for k in range(4):
            chord_at(pre, k * step, step * 0.62, base_gain * 0.85)
        chord_at(tonic, pulse_span, total_ms - pulse_span, base_gain)
    elif cls == "question":
        sus = [root_midi + scale_steps[0], root_midi + scale_steps[3],
               root_midi + 7, root_midi + 12 + scale_steps[1]]
        chord_at(sus, 0, total_ms, base_gain)
    elif cls in ("success", "error") and abs(valence) > 0.35:
        pre_degree = 4 if cls == "success" else 1
        pre = diatonic_chord(scale_steps, root_midi, pre_degree, size=size)
        split = int(total_ms * 0.42)
        chord_at(pre, 0, split, base_gain)
        chord_at(tonic, split, total_ms - split, base_gain)
    else:
        chord_at(tonic, 0, total_ms, base_gain)

    return events, total_ms, patch


# ---------------------------------------------------------------------------
# Creature mode: same vocabulary semantics, organic voice. Pitch glides not
# scale degrees; roughness for negative valence; body size = stakes.
# ---------------------------------------------------------------------------

CREATURE_BASE_MIDI = {"status": 84, "done": 72, "question": 76, "triumph": 70,
                      "halted": 60, "alert": 40, "departure": 62}

# (patch, semitone_offset, start_units, dur_units, gain)
CALLS_SHORT = {
    "status":    [("chirp", 0, 0.0, 0.6, 0.7)],
    "done":      [("coo", 3, 0.0, 1.2, 0.9), ("coo", 0, 1.5, 1.8, 1.0)],
    "question":  [("chirp", 0, 0.0, 0.8, 0.9), ("chirp", 5, 1.1, 1.4, 1.0)],
    "halted":    [("croak", 0, 0.0, 0.5, 1.0), ("croak", 1, 0.7, 0.5, 1.0),
                  ("chirp", 6, 1.8, 1.2, 0.6)],
    "alert":     [("growl", 0, 0.0, 2.6, 1.0)],
    "triumph":   [("whoop", 0, 0.0, 1.4, 1.0), ("chirp", 9, 1.6, 0.8, 0.9)],
    "departure": [("hoot", 0, 0.0, 1.2, 1.0), ("hoot", -3, 1.6, 1.8, 0.7)],
}
CALLS_BASE = {
    "status":    [("chirp", 0, 0.0, 0.6, 0.75), ("chirp", -1, 0.8, 0.7, 0.6)],
    "done":      [("coo", 5, 0.0, 1.1, 0.85), ("coo", 2, 1.3, 1.0, 0.9),
                  ("coo", 0, 2.5, 2.0, 1.0)],
    "question":  [("chirp", 0, 0.0, 0.8, 0.85), ("chirp", 3, 1.0, 0.8, 0.9),
                  ("chirp", 7, 2.0, 1.6, 1.0)],
    "halted":    [("croak", 0, 0.0, 0.5, 1.0), ("croak", 1, 0.7, 0.5, 1.0),
                  ("croak", 0, 1.9, 0.5, 0.95), ("croak", 1, 2.6, 0.5, 0.95),
                  ("chirp", 8, 3.8, 1.4, 0.55)],
    "alert":     [("growl", 0, 0.0, 3.2, 1.0)],
    "triumph":   [("whoop", 0, 0.0, 1.5, 1.0), ("chirp", 7, 1.7, 0.6, 0.9),
                  ("chirp", 12, 2.5, 0.9, 0.95)],
    "departure": [("hoot", 0, 0.0, 1.2, 1.0), ("hoot", -3, 1.6, 1.4, 0.8),
                  ("hoot", -7, 3.2, 2.0, 0.55)],
}
CALLS_LONG = {
    "status":    CALLS_BASE["status"],
    "done":      [("coo", 5, 0.0, 1.1, 0.85), ("coo", 2, 1.3, 1.0, 0.9),
                  ("coo", 0, 2.5, 1.6, 1.0), ("coo", -5, 4.3, 2.2, 0.7)],
    "question":  [("chirp", 0, 0.0, 0.8, 0.8), ("chirp", 3, 1.0, 0.8, 0.85),
                  ("chirp", 5, 2.0, 0.8, 0.9), ("chirp", 7, 3.0, 0.8, 0.9),
                  ("chirp", 12, 4.0, 1.8, 1.0)],
    "halted":    CALLS_BASE["halted"],
    "alert":     [("growl", 0, 0.0, 2.8, 1.0), ("snarl", 2, 3.0, 1.6, 0.8)],
    "triumph":   [("whoop", 0, 0.0, 1.5, 1.0), ("chirp", 7, 1.7, 0.6, 0.9),
                  ("chirp", 10, 2.4, 0.6, 0.92), ("chirp", 12, 3.1, 0.6, 0.95),
                  ("whoop", 12, 3.9, 1.6, 0.85)],
    "departure": [("hoot", 0, 0.0, 1.2, 1.0), ("hoot", -3, 1.6, 1.2, 0.85),
                  ("hoot", -5, 3.0, 1.2, 0.7), ("hoot", -9, 4.4, 2.2, 0.5)],
}


def pick_call(need, deliberation):
    if deliberation < DELIBERATION_SHORT:
        return list(CALLS_SHORT[need])
    if deliberation < DELIBERATION_LONG:
        return list(CALLS_BASE[need])
    return list(CALLS_LONG[need])


def creature_call(need, deliberation, arousal, valence, looping):
    """The full single-creature call: tier + arousal mods + valence droop.
    Shared by solo creature mode and both voices of the duet."""
    calls = pick_call(need, deliberation)
    if need == "alert" and arousal >= 0.6:
        calls = [("growl", 0, 0.0, 2.2, 1.0)] + \
                [("snarl", 2, 2.3 + k * 1.1, 0.8, 0.85) for k in range(3)]
    if need == "halted" and arousal > 0.7:
        calls = CALLS_BASE["halted"][:4] + \
                [("croak", 0, 3.8, 0.5, 0.9), ("croak", 1, 4.5, 0.5, 0.9),
                 ("chirp", 8, 5.7, 1.4, 0.5)]
    if need == "status" and looping:
        calls = [("purr", 0, 0.0, 3.0, 0.7)]
    if valence < -0.2 and need in ("status", "done", "question", "departure"):
        patch_f, semi_f, start_f, dur_f, gain_f = calls[-1]
        calls[-1] = ("fall", semi_f, start_f, dur_f, gain_f)
    return calls


def _shift_call(calls, dt):
    return [(p, s, st + dt, d, g) for p, s, st, d, g in calls]


def _call_end(calls):
    return max(st + d for _, _, st, d, _ in calls)


def build_creature(text, need, session):
    text = text[-ANALYSIS_TAIL_CHARS:]
    rich = analyze_rich(text)
    rng = seeded_rng(text)
    movement, trend, looping, history_sim = update_trajectory(
        session, rich["vec"], rich["valence"])
    if need is None:
        need = infer_need(text, rich, trend)
    t = rich["texture"]
    valence, arousal = rich["valence"], rich["arousal"]
    wobble = 1.0 - rich["certainty"]
    deliberation = deliberation_score(text, need, wobble)

    unit = (UNIT_CALM_MS + (UNIT_INTENSE_MS - UNIT_CALM_MS) * arousal) \
        * (0.9 + 0.3 * deliberation)
    base_gain = 0.6 + 0.3 * arousal
    width = (0.4, 0.7, 0.95)[min(2, int(t[3] * 3))]
    jitter = 6 if t[5] < 0.5 else 16

    size = 0.15 + 0.5 * arousal + 0.35 * max(0.0, -valence)
    root = CREATURE_BASE_MIDI[need] + round((t[6] - 0.5) * 4) - round(size * 10)

    calls = creature_call(need, deliberation, arousal, valence, looping)
    if trend > GRACE_TREND:
        end_u = max(s + d for _, _, s, d, _ in calls)
        calls.append(("chirp", 12, end_u + 0.2, 0.4, 0.5))
    elif trend < -GRACE_TREND:
        end_u = max(s + d for _, _, s, d, _ in calls)
        calls.append(("fall", -5, end_u + 0.2, 0.7, 0.5))

    spread = 1.0 + 0.4 * movement
    events = []
    for i, (patch, semis, start_u, dur_u, gain) in enumerate(calls):
        midi = root + round(semis * spread)
        start = int(start_u * unit)
        dur = int(dur_u * unit * rng.uniform(0.92, 1.08))
        pan = _spread_pan(i % 3, 3, width * 0.5)
        scatter = rng.randint(-14, 14)
        if wobble > WOBBLE_THRESHOLD and patch != "purr":
            cents = int(10 + 30 * wobble)
            events.append(_event(midi, start, dur, patch, base_gain * gain * 0.62,
                                 rng, pan=pan, jitter=jitter,
                                 detune=cents + scatter))
            events.append(_event(midi, start, dur, patch, base_gain * gain * 0.62,
                                 rng, pan=-pan, jitter=jitter,
                                 detune=-cents + scatter))
        else:
            events.append(_event(midi, start, dur, patch, base_gain * gain, rng,
                                 pan=pan, jitter=jitter, detune=scatter))

    total_ms = int(max(s + d for _, _, s, d, _ in calls) * unit)
    return {
        "text": text[:200], "source": rich["source"], "need": need,
        "valence": round(valence, 3), "intensity": round(arousal, 3),
        "certainty": round(rich["certainty"], 3),
        "progress": round(rich["progress"], 3),
        "familiarity": round(0.6 if history_sim is None else
                             _rescale(history_sim, 0.35, 0.80, 0.0, 1.0), 3),
        "movement": round(movement, 3), "trend": round(trend, 3),
        "looping": looping, "wobble": round(wobble, 3),
        "deliberation": round(deliberation, 3), "notes": len(calls),
        "scale": "-", "progression": "-", "articulation": "creature",
        "instrument": calls[0][0], "root_midi": root,
        "texture": [round(x, 2) for x in t],
        "total_ms": total_ms, "beat_ms": None, "pump": False,
        "events": events,
    }


# ---------------------------------------------------------------------------
# Duet mode: two creatures, dominance <-> submission. A = the session,
# B = the other (the problem, or the silent user being petitioned).
# Morton's rules: dominant = low/harsh/loud/last word; submissive =
# high/tonal/quiet/rising. Overlap = conflict; turn-taking = peace.
# ---------------------------------------------------------------------------

PAN_A = -0.38
PAN_B = 0.38

def duet_layout(interaction, need, deliberation, arousal, valence, looping):
    """Compose the exchange from full single-creature calls. A speaks its
    proven menagerie call; B answers with its own; the interaction shapes
    timing (gap = peace, overlap = conflict), B's choice, and who closes."""
    def A(calls):
        return [("a",) + c for c in calls]

    def B(calls):
        return [("b",) + c for c in calls]

    a_call = creature_call(need, deliberation, arousal, valence, looping)
    end_a = _call_end(a_call)

    if interaction == "victory":
        yield_call = _shift_call([("whimper", 6, 0.0, 0.9, 0.55),
                                  ("whimper", 2, 1.1, 1.3, 0.45)], end_a + 0.3)
        coda = _shift_call([("chirp", 14, 0.0, 0.8, 0.85)],
                           _call_end(yield_call) + 0.3)
        return A(a_call) + B(yield_call) + A(coda)

    if interaction == "confrontation":
        b_call = _shift_call(
            creature_call("alert", deliberation, arousal, valence, False),
            end_a * 0.55)
        a2 = _shift_call([("snarl", 2, 0.0, 0.8, 0.95)], _call_end(b_call) + 0.1)
        b2 = _shift_call([("snarl", 0, 0.0, 1.1, 1.0)], _call_end(a2) - 0.4)
        return A(a_call) + B(b_call) + A(a2) + B(b2)

    if interaction == "retreat":
        b_call = creature_call("alert", deliberation, max(arousal, 0.5),
                               valence, False)
        appease = _shift_call([("whimper", 8, 0.0, 0.9, 0.6),
                               ("whimper", 12, 1.1, 0.8, 0.5)],
                              _call_end(b_call) * 0.6)
        fading = _shift_call([("whimper", 15, 0.0, 0.8, 0.35)],
                             _call_end(b_call) + 0.6)
        close = _shift_call([("rumble", -2, 0.0, 2.0, 0.8)],
                            _call_end(fading) + 0.3)
        return B(b_call) + A(appease) + A(fading) + B(close)

    if interaction == "petition":
        presence = _shift_call([("rumble", 0, 0.0, 1.6, 0.35)], end_a + 0.6)
        return A(a_call) + B(presence)

    if interaction == "acceptance":
        echo = a_call[-2:] if len(a_call) >= 2 else a_call
        base = min(st for _, _, st, _, _ in echo)
        echo = [("coo" if p in ("chirp", "fall") else p, s, st - base, d, g * 0.65)
                for p, s, st, d, g in echo]
        echo = _shift_call(echo, end_a + 0.4)
        return A(a_call) + B(echo)

    if interaction == "mutual_calm":
        answer = _shift_call([("coo", -2, 0.0, 1.0, 0.7),
                              ("coo", -4, 1.2, 1.4, 0.6)], end_a + 0.5)
        if deliberation < DELIBERATION_SHORT:
            return A(a_call) + B(answer)
        reply = _shift_call([("chirp", 5, 0.0, 0.6, 0.75)],
                            _call_end(answer) + 0.4)
        settle = _shift_call([("coo", 0, 0.0, 1.6, 0.6)], _call_end(reply) + 0.3)
        return A(a_call) + B(answer) + A(reply) + B(settle)

    # standoff: same short growl circled, B holds the field
    growl = [("growl", 0, 0.0, 1.5, 0.9)]
    return (A(growl) + B(_shift_call(growl, 1.8)) + A(_shift_call(growl, 3.6))
            + B(_shift_call([("rumble", -1, 0.0, 1.8, 0.7)], 5.4)))


def dominance_score(rich):
    """Session's dominance over the other: winning + progressing. Certainty
    only emboldens when winning — confidently asserting defeat is not power."""
    dom = 0.5 + 0.35 * rich["valence"] \
        + 0.15 * (rich["progress"] - 0.5) * 2.0
    if rich["valence"] > 0:
        dom += 0.2 * (rich["certainty"] - 0.5) * 2.0
    return max(0.0, min(1.0, dom))


def pick_interaction(need, dom, valence, looping):
    if looping and need == "alert":
        return "standoff"
    if need == "triumph":
        return "victory"
    if need == "alert":
        return "retreat" if dom < 0.4 else "confrontation"
    if need in ("question", "halted"):
        return "petition"
    if need in ("done", "status", "departure"):
        return "mutual_calm" if valence > 0.2 else "acceptance"
    return "acceptance"


def build_duet(text, need, session):
    text = text[-ANALYSIS_TAIL_CHARS:]
    rich = analyze_rich(text)
    rng = seeded_rng(text)
    movement, trend, looping, history_sim = update_trajectory(
        session, rich["vec"], rich["valence"])
    if need is None:
        need = infer_need(text, rich, trend)
    t = rich["texture"]
    valence, arousal = rich["valence"], rich["arousal"]
    wobble = 1.0 - rich["certainty"]
    deliberation = deliberation_score(text, need, wobble)
    dom = dominance_score(rich)
    interaction = pick_interaction(need, dom, valence, looping)

    unit = (UNIT_CALM_MS + (UNIT_INTENSE_MS - UNIT_CALM_MS) * arousal) \
        * (0.9 + 0.3 * deliberation)
    base_gain = 0.6 + 0.3 * arousal
    jitter = 6 if t[5] < 0.5 else 16

    size = 0.15 + 0.5 * arousal + 0.35 * max(0.0, -valence)
    root_a = CREATURE_BASE_MIDI[need] + round((t[6] - 0.5) * 4) - round(size * 8)
    # dominant party sounds bigger: B sits below A when it holds power
    root_b = root_a + round((dom - 0.5) * 20) - 5
    gain_b = 0.75 + 0.5 * (1.0 - dom)

    calls = duet_layout(interaction, need, deliberation, arousal, valence, looping)

    a_index = 0
    b_index = 0
    events = []
    for voice, patch, semis, start_u, dur_u, gain in calls:
        if voice == "a":
            root, vgain, pan = root_a, base_gain * gain, PAN_A
            if interaction == "retreat":
                pan = PAN_A - 0.22 * a_index
                vgain *= max(0.4, 1.0 - 0.18 * a_index)
            elif interaction == "confrontation":
                pan = PAN_A + 0.13 * a_index
            a_index += 1
        else:
            root, vgain, pan = root_b, base_gain * gain * gain_b, PAN_B
            if interaction == "confrontation":
                pan = PAN_B - 0.13 * b_index
            b_index += 1

        midi = root + semis
        start = int(start_u * unit)
        dur = int(dur_u * unit * rng.uniform(0.92, 1.08))
        scatter = rng.randint(-14, 14)
        if wobble > WOBBLE_THRESHOLD and voice == "a" and patch != "purr":
            cents = int(10 + 30 * wobble)
            events.append(_event(midi, start, dur, patch, vgain * 0.62, rng,
                                 pan=pan, jitter=jitter, detune=cents + scatter))
            events.append(_event(midi, start, dur, patch, vgain * 0.62, rng,
                                 pan=pan * 0.7, jitter=jitter,
                                 detune=-cents + scatter))
        else:
            events.append(_event(midi, start, dur, patch, vgain, rng,
                                 pan=pan, jitter=jitter, detune=scatter))

    total_ms = int(max(s + d for _, _, _, s, d, _ in calls) * unit)
    return {
        "text": text[:200], "source": rich["source"], "need": need,
        "interaction": interaction, "dominance": round(dom, 3),
        "valence": round(valence, 3), "intensity": round(arousal, 3),
        "certainty": round(rich["certainty"], 3),
        "progress": round(rich["progress"], 3),
        "familiarity": round(0.6 if history_sim is None else
                             _rescale(history_sim, 0.35, 0.80, 0.0, 1.0), 3),
        "movement": round(movement, 3), "trend": round(trend, 3),
        "looping": looping, "wobble": round(wobble, 3),
        "deliberation": round(deliberation, 3), "notes": len(calls),
        "scale": "-", "progression": "-", "articulation": "duet",
        "instrument": interaction, "root_midi": root_a,
        "texture": [round(x, 2) for x in t],
        "total_ms": total_ms, "beat_ms": None, "pump": False,
        "events": events,
    }


def compose(text, need=None, session="default", mode=None):
    mode = mode or MODE
    if mode == "duet":
        return build_duet(text, need, session)
    if mode == "creature":
        return build_creature(text, need, session)
    if mode == "vocab":
        return build_vocab(text, need, session)

    rng = seeded_rng(text)
    cls, valence, intensity, source = analyze(text)
    cfg = CLASS_CONFIG[cls]

    scale_name = pick_scale(cfg, valence)
    scale_steps = SCALES[scale_name]
    prog_name = pick_progression(cfg, valence)
    articulation = pick_articulation(cfg, intensity)
    instrument = cfg["inst_bright"] if valence >= 0 else cfg["inst_dark"]

    root_pc = hashlib.md5(text.encode("utf-8", "replace")).digest()[0] % 12
    shift = round(valence * 3)
    if intensity > 0.82 and valence >= -0.1:
        shift += 12
    if valence < -0.5:
        shift -= 7
    root_midi = cfg["base"] + root_pc + shift

    chord_size = 4 if valence > 0.2 else 3
    base_gain = 0.6 + 0.3 * intensity
    total_ms = int(DURATION_CALM_MS + (DURATION_INTENSE_MS - DURATION_CALM_MS) * intensity)

    degrees = PROGRESSIONS[prog_name]
    events = []
    beat_ms = None
    pump = False

    if mode == "pad":
        events, total_ms, instrument = build_pad(cls, scale_steps, root_midi,
                                                 intensity, valence, base_gain, rng)
        articulation = "pad"
    elif cls != "question" and intensity >= STYLE_MIN_INTENSITY:
        style = pick_style(cls, intensity, rng)
        events, total_ms, beat_ms, pump = render_pattern(
            STYLES[style], scale_steps, root_midi, degrees,
            cls, intensity, valence, base_gain, rng)
        articulation = style
    elif articulation == "stab":
        chord = diatonic_chord(scale_steps, root_midi, degrees[0], size=chord_size + 1)
        for j, midi in enumerate(chord):
            events.append(_event(midi, 0, total_ms, instrument, base_gain, rng,
                                 pan=_spread_pan(j, len(chord), 0.7)))
        events.append(_event(chord[0] - 12, 0, total_ms, instrument, base_gain * 0.8, rng))
    else:
        slot = total_ms / max(1, len(degrees))
        for i, degree in enumerate(degrees):
            chord = diatonic_chord(scale_steps, root_midi, degree, size=chord_size)
            start = int(i * slot)
            dur = int(slot * (1.7 if articulation == "shimmer" else 1.0))
            if articulation == "arp":
                spread = slot / (len(chord) + 1)
                for j, midi in enumerate(chord):
                    events.append(_event(midi, int(start + j * spread),
                                         int(dur * 0.9), instrument,
                                         base_gain - 0.05 * j, rng,
                                         pan=_spread_pan(j, len(chord), 0.6)))
            else:
                gain = base_gain if articulation == "block_prog" else base_gain * 0.75
                for j, midi in enumerate(chord):
                    events.append(_event(midi, start, dur, instrument,
                                         gain - 0.04 * j, rng,
                                         pan=_spread_pan(j, len(chord), 0.6)))

    return {
        "text": text[:200],
        "source": source,
        "class": cls,
        "valence": round(valence, 3),
        "intensity": round(intensity, 3),
        "scale": scale_name,
        "progression": prog_name,
        "articulation": articulation,
        "instrument": instrument,
        "root_midi": root_midi,
        "total_ms": total_ms,
        "beat_ms": beat_ms,
        "pump": pump,
        "events": events,
    }


def _spread_pan(index, count, width):
    if count <= 1:
        return 0.0
    return ((index - (count - 1) / 2.0) / ((count - 1) / 2.0)) * width


def _event(midi, start_ms, dur_ms, instrument, gain, rng, pan=0.0, jitter=8,
           detune=0):
    midi = max(24, min(100, midi))
    ev = {
        "midi": midi,
        "start_ms": max(0, start_ms + rng.randint(-jitter, jitter)),
        "dur_ms": max(90, dur_ms),
        "inst": instrument,
        "gain": max(0.1, min(1.0, gain * rng.uniform(0.92, 1.0))),
        "pan": max(-1.0, min(1.0, pan)),
    }
    if detune:
        ev["detune"] = detune
    return ev


# ---------------------------------------------------------------------------
# Rendering (pure stdlib sample mix)
# ---------------------------------------------------------------------------

_BANK_CACHE = {}


def load_note(instrument, midi):
    midi = max(45, min(84, midi))
    key = (instrument, midi)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    path = os.path.join(BANK_DIR, instrument, f"{midi}.wav")
    try:
        with wave.open(path, "rb") as w:
            frames = w.readframes(w.getnframes())
        samples = array.array("h")
        samples.frombytes(frames)
    except (FileNotFoundError, wave.Error):
        samples = array.array("h")
    _BANK_CACHE[key] = samples
    return samples


def _samples_for(ev, phrase, engine):
    if engine == "synth":
        return synth.voice(ev["inst"], ev["midi"],
                           phrase["intensity"], phrase["valence"], ev["dur_ms"],
                           ev.get("detune", 0))
    return load_note(PATCH_TO_BANK.get(ev["inst"], "pad"), ev["midi"])


def _pan_gains(pan):
    angle = (pan + 1.0) * (math.pi / 4.0)
    return math.cos(angle), math.sin(angle)


def _pump_cycle(beat_ms):
    clen = max(1, int(round(SAMPLE_RATE * beat_ms / 1000.0)))
    return clen, [PUMP_DEPTH + (1.0 - PUMP_DEPTH) * ((i / clen) ** PUMP_RECOVERY)
                  for i in range(clen)]


def render(phrase, engine=None):
    engine = engine or ENGINE
    if engine == "synth":
        synth.prewarm(phrase["events"], phrase["intensity"], phrase["valence"])

    total_ms = phrase["total_ms"] + MAX_PHRASE_TAIL_MS
    total_samples = int(SAMPLE_RATE * total_ms / 1000) + SAMPLE_RATE
    left = array.array("i", bytes(4 * total_samples))
    right = array.array("i", bytes(4 * total_samples))

    pump = bool(phrase.get("pump") and phrase.get("beat_ms"))
    clen, cycle = _pump_cycle(phrase["beat_ms"]) if pump else (1, [1.0])
    haas = int(SAMPLE_RATE * HAAS_SECONDS)
    reach = 0

    for ev in phrase["events"]:
        note = _samples_for(ev, phrase, engine)
        start = int(SAMPLE_RATE * ev["start_ms"] / 1000)
        gain = ev["gain"]
        lg, rg = _pan_gains(ev.get("pan", 0.0))
        wide = ev["inst"] in WIDE_PATCHES
        duck = pump and ev["inst"] not in NO_DUCK_PATCHES

        if engine == "synth":
            sustain = release = None
            n = len(note)
        else:
            sustain = int(SAMPLE_RATE * ev["dur_ms"] / 1000)
            release = min(int(SAMPLE_RATE * 0.12), sustain // 2 + 1)
            n = min(len(note), sustain + release)

        for i in range(n):
            pos = start + i
            if pos >= total_samples:
                break
            s = note[i] * gain
            if sustain is not None and i > sustain:
                s *= max(0.0, 1.0 - (i - sustain) / release)
            if duck:
                s *= cycle[pos % clen]
            left[pos] += int(s * lg)
            right[pos] += int(s * rg)
            if wide:
                hp = pos + haas
                if hp < total_samples:
                    left[hp] += int(s * rg * 0.5)
                    right[hp] += int(s * lg * 0.5)
        end = start + n + (haas if wide else 0)
        if end > reach:
            reach = end

    n_frames = min(total_samples, reach)

    peak = 1
    for v in left[:n_frames]:
        a = -v if v < 0 else v
        if a > peak:
            peak = a
    for v in right[:n_frames]:
        a = -v if v < 0 else v
        if a > peak:
            peak = a
    norm = min(1.0, 30000.0 / peak)

    threshold = max(2, int(peak * 0.004))
    floor_idx = SAMPLE_RATE // 10
    while n_frames > floor_idx:
        l = left[n_frames - 1]
        r = right[n_frames - 1]
        if (l if l >= 0 else -l) >= threshold or (r if r >= 0 else -r) >= threshold:
            break
        n_frames -= 1

    out = array.array("h", bytes(4 * n_frames))
    for i in range(n_frames):
        out[2 * i] = max(-32768, min(32767, int(left[i] * norm)))
        out[2 * i + 1] = max(-32768, min(32767, int(right[i] * norm)))
    return out


def write_wav(samples, path, channels=2):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())


PLAY_VOLUME = os.environ.get("MUSICBOX_VOLUME", "0.35")


def play(samples):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    write_wav(samples, tmp.name)
    subprocess.Popen(["afplay", "-v", PLAY_VOLUME, tmp.name])
    return tmp.name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_text(args):
    if args.text is not None:
        return args.text
    return sys.stdin.read()


def main():
    parser = argparse.ArgumentParser(description="Session music black box")
    parser.add_argument("command", choices=["play", "render", "phrase"])
    parser.add_argument("--text", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--engine", default=None, choices=["synth", "bank"])
    parser.add_argument("--need", default=None, choices=list(NEEDS))
    parser.add_argument("--session", default="default")
    parser.add_argument("--mode", default=None,
                        choices=["vocab", "creature", "duet", "pad", "full"])
    args = parser.parse_args()

    text = read_text(args)
    phrase = compose(text, need=args.need, session=args.session, mode=args.mode)

    if args.command == "phrase":
        print(json.dumps(phrase, indent=2))
        return

    samples = render(phrase, engine=args.engine)
    if args.command == "render":
        out = args.out or "out.wav"
        write_wav(samples, out)
        print(out)
        return

    play(samples)


if __name__ == "__main__":
    main()
