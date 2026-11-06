"""Speed of the C++ masked-frame sampler (ellis._core) against Stim, on the gross code and large surface codes.

usage: python tools/bench/bench_frame.py run OUT.jsonl [--budget S] [--circuits bb144,surface_d25] [--p 0,0.001] [--tools mask,stim]
       python tools/bench/bench_frame.py threads OUT.jsonl [--budget S]       # thread scaling, bb144 and surface_d25 at p_loss 1e-3
       python tools/bench/bench_frame.py summary OUT.jsonl [THREADS.jsonl]    # markdown tables
Circuits: bb72, bb144 (gross code [[144,12,12]], 12 rounds, from fixtures/; ellis.noise.noisy) and
surface_dD (rotated memory-Z, D rounds; ellis.noise.surface_memory).  One core unless `threads`.
Tools (all but stim_bool / mask_bool / numpy write bit-packed output):
  stim       pip stim: compile_detector_sampler().sample(bit_packed=True), Stim's Python API (ignores loss)
  stim_bool  the same with bool output, Stim's default
  stim_cpp   stim::FrameSimulator compiled into ellis._core, run in the same batched harness (ignores loss)
  mask       MaskFrameSampler.sample_detectors: detection events and observables
  mask_her   the same plus per-measurement heralds
  mask_meas  MaskFrameSampler.sample: measurements and heralds
  mask_bool  MaskFrameSampler.sample_detectors with bool output (compare with stim_bool)
  numpy      ellis.frame.sample, the numpy reference implementation (bool measurements and heralds)
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import stim
from ellis._core import MaskFrameSampler, StimSampler
from ellis.noise import P, noisy, surface_memory

from ellis import frame as mf

BB_CIRCUITS = Path(__file__).resolve().parent / "fixtures"

CIRCUITS = ["bb72", "bb144", "surface_d11", "surface_d15", "surface_d21", "surface_d25", "surface_d31"]
P_LOSS = [0.0, 1e-3, 5e-3]
TOOLS = ["stim", "stim_bool", "stim_cpp", "mask", "mask_her", "mask_meas", "mask_bool", "numpy"]
MAX_OUTPUT_BYTES = 1.5e9


def circuit(name, p_loss):
    if name.startswith("bb"):
        base = stim.Circuit((BB_CIRCUITS / f"{name}.stim").read_text()).flattened()
        return stim.Circuit(noisy(base, p_loss=p_loss))
    d = int(name.removeprefix("surface_d"))
    return surface_memory(d, d, P, p_loss)


def sampler(tool, c, threads=1):
    """(f, output bytes per shot): f(n) samples n shots."""
    D, O, M = c.num_detectors, c.num_observables, c.num_measurements
    packed = (D + 7) // 8 + (O + 7) // 8
    if tool in ("stim", "stim_bool"):
        pip = c.compile_detector_sampler(seed=1)
        bit_packed = tool == "stim"
        return (lambda n: pip.sample(n, bit_packed=bit_packed, separate_observables=True)), (packed if bit_packed else D + O)
    if tool == "stim_cpp":
        harness = StimSampler(c, seed=1)
        return (lambda n: harness.sample_detectors(n, bit_packed=True, threads=threads)), packed
    if tool == "numpy":
        return (lambda n: mf.sample(c, n, seed=1)), 2 * M
    s = MaskFrameSampler(c, seed=1)
    if tool == "mask_meas":
        return (lambda n: s.sample(n, bit_packed=True, threads=threads)), 2 * ((M + 7) // 8)
    if tool == "mask_bool":
        return (lambda n: s.sample_detectors(n, threads=threads)), D + O
    heralds = tool == "mask_her"
    return (lambda n: s.sample_detectors(n, bit_packed=True, heralds=heralds, threads=threads)), packed + heralds * ((M + 7) // 8)


def timed(f, n0, budget, cap):
    """(shots, seconds): a pilot of n0 shots sizes calls of about `budget` seconds, at most `cap` shots each;
    calls repeat until half the budget has passed (several only when the output cap binds, e.g. many threads)."""
    f(min(n0, 256))
    t = time.perf_counter(); f(n0); dt = time.perf_counter() - t
    if dt >= budget / 3:
        return n0, dt
    n = int(min(cap, max(n0, n0 * budget / max(dt, 1e-9))))
    shots, start = 0, time.perf_counter()
    while True:
        f(n)
        shots += n
        if time.perf_counter() - start >= budget / 2:
            return shots, time.perf_counter() - start


def measure(name, p_loss, tool, budget, threads=1):
    c = circuit(name, p_loss)
    f, bytes_per_shot = sampler(tool, c, threads)
    n0 = max(64, int(3e5 / c.num_measurements)) if tool == "numpy" else max(4096, int(3e7 / c.num_measurements) * threads)
    cap = MAX_OUTPUT_BYTES / bytes_per_shot
    n, dt = timed(f, int(min(n0, cap)), budget, cap)
    row = {"circuit": name, "p_loss": p_loss, "tool": tool, "threads": threads, "shots": n, "secs": round(dt, 3),
           "shots_per_s": round(n / dt, 1), "qubits": c.num_qubits, "measurements": c.num_measurements,
           "detectors": c.num_detectors, "host": platform.node()}
    if tool.startswith("mask"):
        s = MaskFrameSampler(c, seed=2)
        row["batch"] = s.batch_size(detectors=tool != "mask_meas", heralds=tool != "mask")
        row["simd_width"] = s.simd_width
        row["herald_rate"] = round(float(s.sample(20000)[1].any(1).mean()), 4)
    return row


def ratio(rate, a, b):
    return f"{rate[a]['shots_per_s'] / rate[b]['shots_per_s']:.2f}" if a in rate and b in rate else ""


def summary(rows, thread_rows):
    out = [("Single core, shots/s (bit-packed output unless noted).  ratio = mask / stim_cpp (same harness) and "
            "mask / stim (pip stim's Python API)."), ""]
    tools = [t for t in TOOLS if any(r["tool"] == t for r in rows)]
    out.append("| circuit | p_loss | " + " | ".join(tools) + " | mask/stim_cpp | mask/stim | shots with a loss |")
    out.append("|---" * (len(tools) + 5) + "|")
    for name in dict.fromkeys(r["circuit"] for r in rows):
        for p in dict.fromkeys(r["p_loss"] for r in rows if r["circuit"] == name):
            rate = {r["tool"]: r for r in rows if r["circuit"] == name and r["p_loss"] == p and r["threads"] == 1}
            cells = [f"{rate[t]['shots_per_s']:,.0f}" if t in rate else "" for t in tools]
            her = next((f"{rate[t]['herald_rate']:.1%}" for t in ("mask", "mask_her", "mask_meas") if t in rate), "")
            out.append(f"| {name} | {p:g} | " + " | ".join(cells) +
                       f" | {ratio(rate, 'mask', 'stim_cpp')} | {ratio(rate, 'mask', 'stim')} | {her} |")
    if thread_rows:
        out += ["", "Threads (detection events, bit-packed), shots/s:", ""]
        ts = sorted({r["threads"] for r in thread_rows})
        out.append("| circuit | p_loss | tool | " + " | ".join(map(str, ts)) + " |")
        out.append("|---" * (len(ts) + 3) + "|")
        for key in dict.fromkeys((r["circuit"], r["p_loss"], r["tool"]) for r in thread_rows):
            rate = {r["threads"]: r["shots_per_s"] for r in thread_rows if (r["circuit"], r["p_loss"], r["tool"]) == key}
            out.append(f"| {key[0]} | {key[1]:g} | {key[2]} | " + " | ".join(f"{rate[t]:,.0f}" if t in rate else "" for t in ts) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "threads", "summary"])
    ap.add_argument("out")
    ap.add_argument("more", nargs="?")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--circuits", default=",".join(CIRCUITS))
    ap.add_argument("--p", default=",".join(map(str, P_LOSS)))
    ap.add_argument("--tools", default=",".join(TOOLS))
    ap.add_argument("--threads", default="1,2,4,8,16,32,64,128,256")
    a = ap.parse_args()
    if a.mode == "summary":
        read = lambda path: [json.loads(line) for line in Path(path).read_text().splitlines()] if path else []
        print(summary(read(a.out), read(a.more)))
        return
    with open(a.out, "a") as f:
        if a.mode == "run":
            jobs = [(c, float(p), t, 1) for c in a.circuits.split(",") for p in a.p.split(",") for t in a.tools.split(",")]
        else:
            jobs = [(c, 1e-3, t, int(T)) for c in ("bb144", "surface_d25") for t in ("mask", "stim_cpp")
                    for T in a.threads.split(",") if int(T) <= (os.cpu_count() or 1)]
        for name, p, tool, threads in jobs:
            if tool == "numpy" and (name == "surface_d31" or p == 0):
                continue                                   # minutes per point; the ratio is clear from the others
            row = measure(name, p, tool, a.budget, threads)
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
