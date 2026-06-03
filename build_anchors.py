#!/usr/bin/env python3
"""Embed anchor exemplars once and bake the retrieval/projection tables.

For each anchor: embed every exemplar, mean-pool -> centroid (the archetype
position in embedding space).

Affect axes are built from the anchor groups:
  valence_axis   = mean(centroids tagged val=pos) - mean(val=neg)
  intensity_axis = mean(centroids tagged aro=hi)  - mean(aro=lo)

At runtime we project a text embedding onto each axis to get continuous
valence/intensity. We also record the p5/p95 of exemplar projections so the
runtime can linearly rescale raw dot products into [-1,1] / [0,1].

Output: anchors.embedded.json (backend + dim + centroids + axes + scaling).
Rebuild whenever anchors.json or the embedder backend changes.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import embedder  # noqa: E402

ANCHORS_IN = os.path.join(HERE, "anchors.json")
ANCHORS_OUT = os.path.join(HERE, "anchors.embedded.json")


def mean(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    return [x / n for x in out]


def subtract(a, b):
    return [x - y for x, y in zip(a, b)]


def normalize(v):
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0:
        return v
    return [x / norm for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    backend = embedder.backend_name()
    spec = json.load(open(ANCHORS_IN))
    anchors = spec["anchors"]

    print(f"Embedding {sum(len(a['exemplars']) for a in anchors)} exemplars via '{backend}'...")

    centroids = []
    all_exemplar_vecs = []
    for a in anchors:
        vecs = []
        for text in a["exemplars"]:
            v = embedder.embed(text)
            if v is None:
                print(f"FAIL embed ({backend}): {text!r}", file=sys.stderr)
                sys.exit(1)
            vecs.append(v)
            all_exemplar_vecs.append((a, v))
        centroid = normalize(mean(vecs))
        centroids.append({"name": a["name"], "class": a["class"], "centroid": centroid})

    dim = len(centroids[0]["centroid"])

    def group_centroids(tag_key, tag_val):
        members = [c["centroid"] for c, a in zip(centroids, anchors) if a[tag_key] == tag_val]
        return members

    valence_axis = normalize(subtract(mean(group_centroids("val", "pos")),
                                      mean(group_centroids("val", "neg"))))
    intensity_axis = normalize(subtract(mean(group_centroids("aro", "hi")),
                                        mean(group_centroids("aro", "lo"))))

    v_projections = [dot(v, valence_axis) for _, v in all_exemplar_vecs]
    i_projections = [dot(v, intensity_axis) for _, v in all_exemplar_vecs]

    extra_axes = {}
    for name, poles in spec.get("axes", {}).items():
        pos_vecs = [embedder.embed(t) for t in poles["pos"]]
        neg_vecs = [embedder.embed(t) for t in poles["neg"]]
        if any(v is None for v in pos_vecs + neg_vecs):
            print(f"FAIL embedding axis {name}", file=sys.stderr)
            sys.exit(1)
        axis = normalize(subtract(mean(pos_vecs), mean(neg_vecs)))
        projections = [dot(v, axis) for v in pos_vecs + neg_vecs]
        extra_axes[name] = {"vec": axis,
                            "lo": percentile(projections, 0.05),
                            "hi": percentile(projections, 0.95)}

    # fixed pseudo-random directions: project an embedding onto these to get
    # texture knobs with locality (near texts -> near sounds)
    import random
    texture_rng = random.Random(1234)
    texture_bank = []
    for _ in range(8):
        direction = normalize([texture_rng.gauss(0, 1) for _ in range(dim)])
        texture_bank.append(direction)

    out = {
        "backend": backend,
        "dim": dim,
        "centroids": centroids,
        "valence_axis": valence_axis,
        "intensity_axis": intensity_axis,
        "valence_scale": {"lo": percentile(v_projections, 0.05),
                          "hi": percentile(v_projections, 0.95)},
        "intensity_scale": {"lo": percentile(i_projections, 0.05),
                            "hi": percentile(i_projections, 0.95)},
        "axes": extra_axes,
        "texture_bank": texture_bank,
    }
    json.dump(out, open(ANCHORS_OUT, "w"))
    print(f"Wrote {ANCHORS_OUT}: backend={backend} dim={dim} "
          f"anchors={len(centroids)} axes={list(extra_axes)} texture=8")


if __name__ == "__main__":
    main()
