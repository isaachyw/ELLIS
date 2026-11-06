"""Certify one reference record for the masked Pauli frame under atom loss (ellis.certify).

usage: python tools/classify_lightstim.py [CIRCUIT.stim ...] [--benchmark] [--only SUBSTR ...]
           [--placement 2q|any] [--reload] [--noiseless-loss] [--pairs] [--random N] [--max-losses K]
           [--seed S] [--workers N] [--save DIR]

Prints one JSON line per circuit: verdict ("proven" / "contextual" / "consistent_on_family"), method, the
reference (number of ones), the gauge size, a minimal witness when contextual, and timing.  Proofs (frame-CSS,
closed diagonal blocks) hold for every loss pattern of every model; a witness holds for the chosen --placement
and --reload ("anywhere" = at any time, including at `noiseless` operations).
--benchmark takes one circuit per LightStim protocol from tests/fixtures/lightstim (paper App. C).
--save DIR writes DIR/<name>.cert.npz with the reference (and the witness or gauge as JSON text).
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import stim
from ellis.certify import certify

LIGHTSTIM_CIRCUITS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "lightstim"


def benchmark_files():
    seen = {}
    for f in sorted(LIGHTSTIM_CIRCUITS.glob("*.stim")):
        seen.setdefault(re.sub(r"_+[0-9a-f]{10}\.stim$", "", f.name), f)
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="*", type=Path)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="keep circuits whose name contains one of these")
    ap.add_argument("--placement", choices=("2q", "any", "anywhere"), default="2q")
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--noiseless-loss", action="store_true", help="also lose atoms at `noiseless`-tagged gates")
    ap.add_argument("--pairs", action="store_true", help="search every pair of single losses")
    ap.add_argument("--random", type=int, default=1000, help="random multi-loss patterns in the search")
    ap.add_argument("--max-losses", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="processes computing the search's constraints")
    ap.add_argument("--save", type=Path, default=None)
    a = ap.parse_args()
    files = a.circuits + (benchmark_files() if a.benchmark else [])
    if a.only:
        files = [f for f in files if any(s in f.name for s in a.only)]
    for f in files:
        t0 = time.perf_counter()
        cert = certify(stim.Circuit(f.read_text()), placement=a.placement, reload=a.reload,
                       include_noiseless=a.noiseless_loss, pairs=a.pairs, n_random=a.random,
                       max_losses=a.max_losses, seed=a.seed, workers=a.workers)
        print(json.dumps(dict(circuit=f.stem, **cert.summary(), secs=round(time.perf_counter() - t0, 2))), flush=True)
        if a.save:
            a.save.mkdir(parents=True, exist_ok=True)
            extra = cert.witness if cert.witness is not None else cert.gauge
            np.savez_compressed(a.save / f"{f.stem}.cert.npz", reference=cert.reference if cert.reference is not None
                                else np.zeros(0, bool), verdict=cert.verdict, method=cert.method,
                                extra=json.dumps(extra, default=str))


if __name__ == "__main__":
    main()
