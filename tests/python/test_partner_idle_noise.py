"""A surviving atom idles when loss cancels its entangling gate (Liu et al., App. M1)."""
import numpy as np
import pytest
import stim

from ellis import frame as mf


@pytest.mark.parametrize("lost", (0, 1))
@pytest.mark.parametrize("gate", ("CX", "CZ"))
def test_present_partner_has_depolarizing_idle_channel(lost, gate):
    c = stim.Circuit(f"R 0 1\nI_ERROR[loss](0) 0 1\n{gate} 0 1\nM 0 1")
    survivor = 1 - lost
    samples, heralds = mf.sample(c, 20000, seed=7, forced_losses={1: [lost]}, lost_partner_idle_p=0.3)
    assert heralds[:, lost].all() and not heralds[:, survivor].any()
    # Z readout flips for X or Y, two of the three equiprobable Pauli faults.
    assert abs(samples[:, survivor].mean() - 0.2) < 0.015
    baseline, _ = mf.sample(c, 100, seed=7, forced_losses={1: [lost]})
    assert not baseline[:, survivor].any()


def test_no_partner_idle_fault_when_gate_executes_or_is_software():
    c = stim.Circuit("R 0 1\nI_ERROR[loss](0) 0\nCX 0 1\nM 0 1")
    present, _ = mf.sample(c, 100, seed=7, lost_partner_idle_p=1)
    assert not present.any()
    software = stim.Circuit("R 0 1\nI_ERROR[loss](0) 0\nCX[feedforward_t] 0 1\nM 0 1")
    missing, _ = mf.sample(software, 100, seed=7, forced_losses={1: [0]}, lost_partner_idle_p=1)
    assert np.count_nonzero(missing[:, 1]) == 0
