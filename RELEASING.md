# Releasing ellis-qec

`.github/workflows/wheels.yml` builds the sdist and the wheels on every push. It publishes them only when the
pushed ref is a tag that starts with `v`.

## Release a version

1. Set `version` in `pyproject.toml`, for example `0.1.1`, and merge the change to `main`.
2. Check that the `test` and `wheels` workflows pass on that commit of `main`.
3. Tag the commit and push the tag. The tag must be `v` followed by the version in `pyproject.toml`.

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

## What the tag starts

1. `sdist` builds `ellis_qec-<version>.tar.gz` with `uv build --sdist`.
2. `wheels` runs cibuildwheel for CPython 3.12, 3.13, and 3.14 on `ubuntu-latest` (manylinux_2_28 x86-64) and
   `macos-15` (arm64). Each wheel is tested with the test suite in `tests/python` before it is uploaded. A check
   fails the job if a wheel does not contain the expected `ellis._core` variants: `avx2` and `sse2` on Linux,
   `polyfill` on macOS.
3. `testpypi` checks that the tag names the sdist's version and uploads every file to TestPyPI. It runs in the GitHub
   environment `testpypi`.
4. `pypi` runs after `testpypi` succeeds and uploads the same files to PyPI. It runs in the GitHub environment `pypi`.

Both upload jobs use trusted publishing (OpenID Connect), so the repository stores no PyPI token. The pending publisher
on each index must name project `ellis-qec`, owner `isaachyw`, repository `ELLIS`, workflow `wheels.yml`, and the
environment of its job: `testpypi` on test.pypi.org and `pypi` on pypi.org.

If the `pypi` job fails, rerun the failed jobs of the workflow run. The `testpypi` job skips files that TestPyPI already
has. PyPI never accepts a second upload of a file, so a release that reached PyPI with a defect needs a new version.

## Check a TestPyPI release

TestPyPI does not host the dependencies, so install them from PyPI:

```bash
python3.12 -m venv /tmp/ellis-check && . /tmp/ellis-check/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ellis-qec==0.1.1
python -c "import ellis, ellis._core as c; print(c.VARIANT)"
```
