"""pytest plugin: save every circuit the LightStim tests validate (via compile_detector_sampler / DEM)."""
import hashlib
import os
import re

import stim

OUT = os.environ.get("HARVEST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "lightstim"))
_seen = set()
def _save(c):
    txt = str(c)
    h = hashlib.sha1(txt.encode()).hexdigest()[:10]
    if h in _seen: return
    _seen.add(h)
    test = os.environ.get("PYTEST_CURRENT_TEST", "unknown").split(" ")[0]
    name = re.sub(r"[^A-Za-z0-9_]+", "_", test.split("::", 1)[-1])[:90]
    with open(f"{OUT}/{name}__{h}.stim", "w") as f: f.write(txt)
for meth in ("compile_detector_sampler", "detector_error_model"):
    orig = getattr(stim.Circuit, meth)
    def wrap(self, *a, _o=orig, **k):
        _save(self); return _o(self, *a, **k)
    setattr(stim.Circuit, meth, wrap)
