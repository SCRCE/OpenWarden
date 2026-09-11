# Publishing OpenWarden

OpenWarden publishes from GitHub Actions through PyPI Trusted Publishing. No
long-lived PyPI token belongs in the repository or GitHub secrets.

## One-Time Configuration

Create pending Trusted Publishers on TestPyPI and PyPI with:

- Owner: `SCRCE`
- Repository: `OpenWarden`
- Test workflow: `test-publish.yml`
- Production workflow: `release.yml`
- Environments: `testpypi` and `pypi` respectively

Create matching GitHub environments. Require approval for the `pypi`
environment.

## Preflight

```bash
python -m pip install -e ".[dev]"
ruff check openwarden tests examples
python -m pytest -q
python -m build
python -m twine check dist/*
```

Inspect both archives before publishing:

```bash
python -m zipfile -l dist/openwarden-*.whl
tar -tzf dist/openwarden-*.tar.gz | sort
```

The wheel must contain only the `openwarden` package and distribution
metadata. The source distribution also contains documentation, examples,
tests, license, and project governance files.

## TestPyPI

Run the `Publish to TestPyPI` workflow manually. After it succeeds:

```bash
python -m venv /tmp/openwarden-test
/tmp/openwarden-test/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  openwarden==0.1.0
/tmp/openwarden-test/bin/python -c \
  "from openwarden import GuardedOpenAI, GuardedPyTorch, __version__; print(__version__)"
```

## PyPI Release

1. Update `openwarden/__about__.py` and `CHANGELOG.md`.
2. Confirm CI passes on `main`.
3. Create a signed tag such as `v0.1.0`.
4. Publish the matching GitHub Release.
5. The `release.yml` workflow builds once and publishes through OIDC.
6. Verify `pip install openwarden==0.1.0` in a clean environment.

PyPI does not allow replacing an existing release file. Version mistakes
require a new release number.
