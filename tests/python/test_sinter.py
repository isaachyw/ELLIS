"""ellis.sinter: the masked frame inside sinter matches sinter's own Stim sampling when nothing is lost."""
import pytest

sinter = pytest.importorskip("sinter")
pytest.importorskip("ellis._core")
from ellis.noise import surface_memory
from ellis.sampler import route
from ellis.sinter import _certificate, masked


def test_masked_sampler_matches_stim_without_loss_and_counts_losses():
    c = surface_memory(3, 3, 0.01, 0.0)
    lossy = surface_memory(3, 3, 0.01, 0.01)
    tasks = [sinter.Task(circuit=c, json_metadata={"loss": 0}), sinter.Task(circuit=lossy, json_metadata={"loss": 1})]
    from sinter._decoding._decoding_pymatching import PyMatchingDecoder
    kw = dict(num_workers=2, max_shots=40_000, max_errors=10**9)
    mask = {s.json_metadata["loss"]: s for s in sinter.collect(tasks=tasks, decoders=["m"], custom_decoders={"m": masked(PyMatchingDecoder())}, **kw)}
    ref = sinter.collect(tasks=tasks[:1], decoders=["pymatching"], **kw)[0]
    a, b = mask[0].errors / mask[0].shots, ref.errors / ref.shots
    assert mask[0].shots == ref.shots == 40_000 and abs(a - b) < 5 * ((a * (1 - a) + b * (1 - b)) / 40_000) ** 0.5 + 1e-3
    assert mask[0].custom_counts["shots_with_loss"] == 0 and mask[1].custom_counts["shots_with_loss"] > 1000
    assert mask[1].errors > mask[0].errors


def test_certifies_only_the_marked_losses():
    # contextual under PESSIMISTIC, but sampling never loses an atom without a marker
    unmarked = "RX 2\nCX 2 0\nCZ 0 2\nCX 2 0\nMX 1\nCX 1 2\nMX 1\n"
    assert route(_certificate(unmarked))[0] == "frame"
