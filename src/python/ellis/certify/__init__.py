"""Does one reference record make the masked Pauli frame exact for every atom-loss pattern?

The masked frame gets every shot's randomness and every Pauli flip right for any loss pattern s; the
only shared ingredient is the reference record r, which must be a possible noiseless record of every lossy
circuit C_s (gate-drop semantics; lost readouts are coins).  Equivalently r must satisfy y.r = sigma_s(y)
for every deterministic parity y of every C_s.  `certify(circuit)` decides this in up to four steps:

  1. validation: only operations the masked frame simulates exactly (frame.validate);
  2. a static PROOF for every loss pattern (losses after any operation, any number, with or without
     reload): frame-CSS, i.e. CSS after relabelling each atom life (frame_css.py).  It gives the reference
     kappa, plus a masked gauge when the labels only exist up to Pauli signs;
  3. otherwise a static proof from closed S/CZ blocks (blocks.py);
  4. otherwise an exact search (constraints.py): the GF(2) system of the exact sign constraints of a family of loss
     patterns (no loss, every single loss up to ghost equivalence, optionally all pairs, random multi-loss
     patterns).  The constraints come from Stim's flow generators of each lossy circuit, with no sampling.
     A contradiction is a PROOF that no single reference exists, returned as a witness; no contradiction
     is evidence only ("consistent_on_family"), together with a reference that fits the whole family.

Ghost equivalence (Appendix A2 of arXiv:2603.04156, generalized): a loss only acts through the lost atom's
later two-qubit gates (dropped) and later readouts (coins); single-qubit gates on a lost atom are
invisible.  So two loss times of one atom with no two-qubit gate, readout, or (with reload) reset of that
atom in between give the same lossy circuit; families keep one slot per class.
"""

from dataclasses import dataclass, field

import numpy as np
import stim

from ellis import frame as mf
from ellis.certify.blocks import DiagonalBlocks, diagonal_blocks
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
from ellis.certify.frame_css import Relabeling, relabel

__all__ = ["PLACEMENTS", "Certificate", "DiagonalBlocks", "Relabeling", "certify", "check_reference", "constraints",
           "diagonal_blocks", "loss_family", "loss_slots", "relabel", "representative_slots", "search", "verify_witness"]


@dataclass
class Certificate:
    verdict: str  # "proven", "contextual" or "consistent_on_family"
    method: str  # "frame-css", "frame-css+gauge", "diagonal-blocks", "diagonal-blocks+gauge" or "exact-search"
    reference: np.ndarray | None = None
    gauge: dict | None = None  # for frame.sample(gauge=...), with the "frame-css+gauge" method
    witness: list | None = None  # [(loss pattern, [measurement indices of y], value)], sums to 0 = 1
    family_size: int = 0
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "verdict": self.verdict,
            "method": self.method,
            "reference_ones": None if self.reference is None else int(self.reference.sum()),
            "gauge_paulis": sum(len(v) for v in (self.gauge or {}).values()),
            "witness_patterns": None if self.witness is None else [w[0] for w in self.witness],
            "family_size": self.family_size,
            "notes": self.notes,
        }



# ------------------------------------------------------------------ entry point
def certify(
    circuit: stim.Circuit,
    placement: str = "2q",
    reload: bool = False,
    include_noiseless: bool = False,
    pairs: bool = False,
    n_random: int = 1000,
    max_losses: int = 4,
    seed: int = 0,
    spot_check: int = 50,
    workers: int = 1,
) -> Certificate:
    """Prove that one reference works for every loss pattern, prove that none does, or report the
    evidence.  placement / reload / include_noiseless define the loss model of the search (proofs hold
    for every model; placement 'anywhere' also allows losses before an atom's first operation and at
    `noiseless` operations); pairs / n_random / max_losses / seed choose the initial family; workers > 1
    computes the search's constraints in that many processes."""
    circuit = circuit.flattened()
    mf.validate(circuit)
    fam_placement, noiseless = ("any", True) if placement == "anywhere" else (placement, include_noiseless)
    rel = relabel(circuit)
    if rel.frame_css:
        cert = Certificate(
            "proven", "frame-css" if rel.exact else "frame-css+gauge", rel.reference, rel.gauge or None, notes=list(rel.reasons)
        )
        if rel.exact and spot_check:  # cheap self-check of the proof on a few patterns
            fam = loss_family(circuit, fam_placement, reload, noiseless, n_random=0)[:spot_check]
            bad = check_reference(circuit, rel.reference, fam, reload)
            cert.family_size = len(fam)
            if bad:
                raise AssertionError(f"frame-CSS reference fails on {bad[:3]}: a bug, please report")
        return cert
    blk = diagonal_blocks(circuit)
    if blk.proven:
        cert = Certificate(
            "proven",
            "diagonal-blocks" if not blk.gauge else "diagonal-blocks+gauge",
            blk.reference,
            blk.gauge or None,
            notes=[f"{blk.blocks} closed S/CZ blocks; not frame-CSS: {rel.reasons[0]}"],
        )
        if not blk.gauge and spot_check:
            fam = loss_family(circuit, fam_placement, reload, noiseless, n_random=0)[:spot_check]
            if bad := check_reference(circuit, blk.reference, fam, reload):
                raise AssertionError(f"diagonal-block reference fails on {bad[:3]}: a bug, please report")
            cert.family_size = len(fam)
        return cert
    srch = _Search(circuit, reload, workers)
    witness = srch.extend(loss_family(circuit, fam_placement, reload, noiseless, pairs, n_random, max_losses, seed))
    notes = [f"not frame-CSS: {'; '.join(rel.reasons[:3])}", f"no closed diagonal blocks: {'; '.join(blk.reasons[:2])}"]
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
