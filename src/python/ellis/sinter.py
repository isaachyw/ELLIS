"""sinter integration: sample with the C++ masked frame (ellis._core), decode with any sinter decoder.

    import sinter
    from ellis.sinter import masked
    stats = sinter.collect(num_workers=128, tasks=tasks, decoders=["relay-bp"], max_shots=10**7, max_errors=1000,
                           custom_decoders={"relay-bp": masked(SinterDecoder_RelayBP(...))})

`masked(decoder)` is a sinter.Sampler: each sinter worker process compiles a MaskFrameSampler for the task circuit
(loss instructions `I_ERROR[loss](p)` included) and the decoder for the task's detector error model, which Stim builds
without loss (loss-unaware decoding).  Everything else is sinter's own StimThenDecodeSampler: postselection, discards,
error classification.  Extra custom counts: `shots_with_loss` (shots with at least one heralded readout).
"""
import collections
import pathlib
import time

import numpy as np
import sinter
from sinter._decoding._stim_then_decode_sampler import _compile_decoder_with_disk_fallback, classify_discards_and_errors

from ellis import frame as mf


class MaskedFrameThenDecodeSampler(sinter.Sampler):
    def __init__(self, decoder: sinter.Decoder, tmp_dir: pathlib.Path | None = None):
        self.decoder = decoder
        self.tmp_dir = tmp_dir

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return _CompiledMaskedFrameThenDecodeSampler(self.decoder, task, self.tmp_dir)


class _CompiledMaskedFrameThenDecodeSampler(sinter.CompiledSampler):
    def __init__(self, decoder: sinter.Decoder, task: sinter.Task, tmp_dir: pathlib.Path | None):
        self.task = task
        self.compiled_decoder = _compile_decoder_with_disk_fallback(decoder, task, tmp_dir)
        self.sampler = mf.compile_sampler(task.circuit)       # seeded from the OS, one per worker
        self.num_obs = task.circuit.num_observables

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs, heralds = self.sampler.sample_detectors(suggested_shots, bit_packed=True, heralds=True)
        num_shots = dets.shape[0]
        custom_counts = collections.Counter({"shots_with_loss": int(np.count_nonzero(heralds.any(axis=1)))})
        num_discards_1 = 0
        if self.task.postselection_mask is not None:
            discarded = np.any(dets & self.task.postselection_mask, axis=1)
            num_discards_1 = int(np.count_nonzero(discarded))
            dets, actual_obs = dets[~discarded], actual_obs[~discarded]
        predictions = self.compiled_decoder.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        num_discards_2, num_errors = classify_discards_and_errors(
            actual_obs=actual_obs, predictions=predictions,
            postselected_observables_mask=self.task.postselected_observables_mask,
            out_count_observable_error_combos=None, num_obs=self.num_obs)
        return sinter.AnonTaskStats(shots=num_shots, errors=num_errors, discards=num_discards_1 + num_discards_2,
                                    seconds=time.monotonic() - t0, custom_counts=custom_counts)


def masked(decoder: sinter.Decoder, tmp_dir: pathlib.Path | None = None) -> MaskedFrameThenDecodeSampler:
    """A sinter.Sampler: masked-frame sampling (with loss) followed by `decoder`."""
    return MaskedFrameThenDecodeSampler(decoder, tmp_dir)
