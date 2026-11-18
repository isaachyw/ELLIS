"""Does one reference record make the masked Pauli frame exact for every atom-loss pattern?

The masked frame gets every shot's randomness and every Pauli flip right for any loss pattern s; the
only shared ingredient is the reference record r, which must be a possible noiseless record of every lossy
circuit C_s (gate-drop semantics; lost readouts are coins).  Equivalently r must satisfy y.r = sigma_s(y)
for every deterministic parity y of every C_s.  `certify(circuit, model)` decides this in up to two steps:

  1. a static PROOF: frame-CSS under the most pessimistic model (frame_css.PESSIMISTIC: loss before the first
     operation and after every operation, |0> reload anywhere), i.e. CSS in one local frame per state segment
     (frame_css.py).  It gives the reference kappa, common to every pattern of every loss model, `model` included.
     This is sufficient, not necessary: a circuit that fails it may still have a common reference under `model`.
     A circuit outside the assumptions under which frame_css decides frame-CSS exactly stops here, verdict
     "outside-assumptions"; any other circuit must use only operations the masked frame simulates exactly
     (frame.validate);
  2. otherwise an exact search under `model` (constraints.py): the GF(2) system of the exact sign constraints of a family of loss
     patterns (no loss, every single loss up to ghost equivalence, optionally all pairs, random multi-loss
     patterns).  The constraints come from Stim's flow generators of each lossy circuit, with no sampling.
     A contradiction is a PROOF that no single reference exists, returned as a witness; no contradiction
     is evidence only ("consistent_on_family"), together with a reference that fits the whole family.
     With reload="anywhere" the family has no reloads (a subset of the model's patterns): a contradiction
     is still a proof, consistency is weaker evidence.

Ghost equivalence (Appendix A2 of arXiv:2603.04156, generalized): a loss only acts through the lost atom's
later two-qubit gates (dropped) and later readouts (coins); single-qubit gates on a lost atom are
invisible.  So two loss times of one atom with no two-qubit gate, readout, or (with reload) reset of that
atom in between give the same lossy circuit; families keep one slot per class.
"""

from dataclasses import dataclass, field

import numpy as np
import stim

from ellis import frame as mf
from ellis.certify.constraints import (
    PLACEMENTS,
    _Search,
    check_reference,
    constraints,
    loss_family,
    loss_slots,
    representative_slots,
    search,
    verify_witness,
)
from ellis.certify.frame_css import PESSIMISTIC, FrameCSS, LossModel, frame_css, loss_locations

__all__ = ["PESSIMISTIC", "PLACEMENTS", "Certificate", "FrameCSS", "LossModel", "certify", "check_reference",
           "constraints", "frame_css", "loss_family", "loss_locations", "loss_slots", "representative_slots",
           "search", "verify_witness"]


@dataclass
class Certificate:
    verdict: str  # "proven", "contextual", "consistent_on_family" or "outside-assumptions"
    method: str  # "frame-css", "exact-search" or "scope"
    reference: np.ndarray | None = None
    witness: list | None = None  # [(loss pattern, [measurement indices of y], value)], sums to 0 = 1
    family_size: int = 0
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "verdict": self.verdict,
            "method": self.method,
            "reference_ones": None if self.reference is None else int(self.reference.sum()),
            "witness_patterns": None if self.witness is None else [w[0] for w in self.witness],
            "family_size": self.family_size,
            "notes": self.notes,
        }



# ------------------------------------------------------------------ entry point
def certify(
    circuit: stim.Circuit,
    model: LossModel = LossModel(),
    pairs: bool = False,
    n_random: int = 1000,
    max_losses: int = 4,
    seed: int = 0,
    spot_check: int = 0,
    workers: int = 1,
) -> Certificate:
    """Prove that one reference works for every loss and reload pattern of every loss model (frame-CSS under
    PESSIMISTIC), prove that none does under `model`, or report the evidence.  pairs / n_random / max_losses /
    seed choose the search's initial family; workers > 1 computes the search's constraints in that many processes.
    spot_check optionally checks up to that many loss patterns of `model` after a static proof; zero skips
    these additional checks."""
    circuit = circuit.flattened()
    reload = model.reload_at_reset

    def family(**kw):
        return loss_family(circuit, reload=reload, slots=loss_locations(circuit, model), **kw)

    fc = frame_css(circuit)
    if fc.verdict == "outside-assumptions":
        return Certificate("outside-assumptions", "scope", notes=list(fc.reasons))
    mf.validate(circuit)
    if fc.proven:
        cert = Certificate("proven", "frame-css", fc.reference, notes=list(fc.reasons))
        if spot_check:
            fam = family(n_random=0)[:spot_check]
            bad = check_reference(circuit, fc.reference, fam, reload)
            cert.family_size = len(fam)
            if bad:
                raise AssertionError(f"frame-CSS reference fails on {bad[:3]}: a bug, please report")
        return cert
    srch = _Search(circuit, reload, workers)
    witness = srch.extend(family(pairs=pairs, n_random=n_random, max_losses=max_losses, seed=seed))
    notes = [f"not frame-CSS under the pessimistic model: {'; '.join(fc.reasons[:3])}"]
    if model.reload == "anywhere":
        notes.append("the search family has no reloads")
    if witness is not None:
        verify_witness(circuit, witness, reload)
        return Certificate(
            "contextual",
            "exact-search",
            witness=witness,
            family_size=len(srch.family),
            notes=notes + ["witness re-checked by Stim sampling"],
        )
    return Certificate("consistent_on_family", "exact-search", reference=srch.reference(), family_size=len(srch.family), notes=notes)
