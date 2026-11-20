"""frame_css against the independent oracles at scale: one JSON line per disagreement, then one summary line.

Each task takes one loss model and one seed and checks the corpus of tests/python/test_frame_css_oracle.py
(random Clifford circuits, constructive frame-CSS circuits, and their single-edit near misses):
  P1  an accepted kappa is a possible record of every lossy circuit tried;
  P2  a constructive circuit is accepted;
  P3  the verdict equals the brute-force frame search (unless frame_css says outside-assumptions).

    python tools/frame_css_sweep.py --seeds 200 --workers 16 --out sweep.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
import frame_oracle as fo  # noqa: E402
import test_frame_css_oracle as T  # noqa: E402
from ellis.certify import frame_css  # noqa: E402

EXACT_O2 = {"zero"}


def check(name: str, seed: int, n_random: int, per_set: int) -> tuple[Counter, list]:
    model, rng = T.MODELS[name], T.seed("sweep", name, seed)
    counts, bad = Counter(), []
    for src, c in T.corpus(rng, model, n_random=n_random, per_set=per_set):
        res = frame_css(c, model)
        counts[res.verdict] += 1
        if res.verdict == "outside-assumptions":
            continue
        want = T.oracle(c, model)
        counts[f"oracle {want}"] += 1
        if res.verdict != want:
            bad.append(dict(prop="P3", model=name, seed=seed, src=src, ellis=res.verdict, oracle=want, circuit=str(c)))
        if "/" in src and src.count("/") == 1 and res.verdict != "frame-css":
            bad.append(dict(prop="P2", model=name, seed=seed, src=src, ellis=res.verdict, circuit=str(c)))
        if res.verdict == "frame-css":
            if model.reload_state in EXACT_O2 and model.reload != "anywhere":
                fails = fo.common_reference_failures(c, model, res.reference, fo.loss_patterns(c, model, cap=1 << 8, n_random=30, n_pairs=40, seed=seed))
            else:
                fails = fo.simulated_failures(c, model, res.reference, np.random.default_rng(seed), n_patterns=60)
            counts["kappa checked"] += 1
            if fails:
                bad.append(dict(prop="P1", model=name, seed=seed, src=src, patterns=str(fails[:3]), circuit=str(c)))
    return counts, bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--per-set", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    total, n_bad = Counter(), 0
    with args.out.open("w") as out, ProcessPoolExecutor(args.workers) as pool:
        tasks = [pool.submit(check, name, s, args.n_random, args.per_set) for name in T.MODELS for s in range(args.seeds)]
        for task in as_completed(tasks):
            counts, bad = task.result()
            total.update(counts)
            for row in bad:
                out.write(json.dumps(row) + "\n")
            n_bad += len(bad)
            out.flush()
        out.write(json.dumps(dict(summary=dict(total), disagreements=n_bad)) + "\n")
    print(json.dumps(dict(summary=dict(total), disagreements=n_bad)))


if __name__ == "__main__":
    main()
