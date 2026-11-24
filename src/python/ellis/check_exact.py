"""Exactness test: the C++ masked frame vs Stim on the explicit lossy circuit C_s.

Noiseless Clifford outcome distributions are uniform over an affine subspace A = r + V of
GF(2)^m, so two samplers agree exactly iff their samples span the same V and one sample of each
differs by an element of V.  With N >= dim(V) + 64 samples the span is complete w.p. 1 - 2^-64.
"""
import glob
import json
import os
import sys

import numpy as np
import stim

from ellis import frame as mf
from ellis._core import MaskFrameSampler
from ellis.gf2 import Span, to_ints


def same_distribution(a, b):
    """Two samples (shots, measurements) of noiseless Clifford outcomes come from the same distribution: equal
    spans V and offsets that differ by an element of V (outcomes are uniform on an affine subspace r + V)."""
    A, B = to_ints(a), to_ints(b)
    va, vb, vj = Span(), Span(), Span()
    for r in A[1:]:
        va.add(r ^ A[0]); vj.add(r ^ A[0])
    for r in B[1:]:
        vb.add(r ^ B[0]); vj.add(r ^ B[0])
    return va.dim == vb.dim == vj.dim and vj.reduce(A[0] ^ B[0]) == 0


def mark_losses(circuit, losses):
    """`circuit` flattened, with `I_ERROR[loss](1)` markers that lose losses[k] right after instruction k (k = -1:
    from the start): the loss pattern of explicit_lossy_circuit(circuit, losses), in every shot."""
    out = stim.Circuit()
    if -1 in losses:
        out.append("I_ERROR", losses[-1], [1.0], tag="loss")
    for k, inst in enumerate(circuit.flattened()):
        out.append(inst)
        if k in losses:
            out.append("I_ERROR", losses[k], [1.0], tag="loss")
    return out


def compare(circ, losses, seed):
    m = circ.num_measurements
    N = m + 64
    a = MaskFrameSampler(mark_losses(circ, losses), seed=seed).sample(N)[0]
    b = mf.explicit_lossy_circuit(circ, losses).compile_sampler(seed=seed).sample(N)
    A, B = to_ints(a), to_ints(b)
    va, vb, vj = Span(), Span(), Span()
    for r in A[1:]:
        va.add(r ^ A[0]); vj.add(r ^ A[0])
    for r in B[1:]:
        vb.add(r ^ B[0]); vj.add(r ^ B[0])
    offset_ok = vj.reduce(A[0] ^ B[0]) == 0
    return va.dim == vb.dim == vj.dim and offset_ok, (va.dim, vb.dim, vj.dim, offset_ok)


def random_losses(circ, rng, k, anywhere):
    flat = list(circ.flattened())
    if anywhere:   # after any gate (incl. between single-qubit gates)
        cand = [i for i, s in enumerate(flat) if s.name not in mf.ANNOTATIONS and s.targets_copy()]
    else:          # after two-qubit gates (the physical placement)
        cand = [i for i, s in enumerate(flat) if s.name in ("CX", "CZ", "SWAP")]
    losses = {}
    for i in rng.choice(cand, size=min(k, len(cand)), replace=False):
        qs = [t.value for t in flat[i].targets_copy()]
        losses.setdefault(int(i), []).append(int(rng.choice(qs)))
    return losses


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m ellis.check_exact 'CIRCUITS/*.stim' [TRIALS]")
    files = sorted(glob.glob(sys.argv[1]))
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = np.random.default_rng(0)
    for f in files:
        circ = stim.Circuit(open(f).read()).flattened()
        fails, first = 0, None
        for t in range(trials):
            losses = random_losses(circ, rng, int(rng.integers(1, 5)), anywhere=(t % 2 == 1))
            ok, info = compare(circ, losses, seed=t)
            if not ok:
                fails += 1; first = first or (losses, info)
        print(json.dumps(dict(circuit=os.path.basename(f)[:-5], trials=trials, fails=fails,
                              example=str(first)[:200] if first else None)), flush=True)
