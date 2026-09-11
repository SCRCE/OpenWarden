# Contributing

## Development Setup

```bash
git clone https://github.com/SCRCE/OpenWarden.git
cd OpenWarden
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

Run the same checks required by CI:

```bash
ruff check openwarden tests examples scripts
python -m pytest -q
python -m build
python -m twine check dist/*
```

## Design Rules

- Keep endpoint-specific parsing in adapters.
- Keep orchestration independent of provider request formats.
- Prefer explicit resource wrappers over recursive endpoint proxying.
- Preserve native response objects for allowed calls.
- Do not log prompt or response content by default.
- Add focused tests for every behavior change.

## Pull Requests

Keep changes narrowly scoped and explain observable behavior, compatibility
impact, and tests. Never include API keys, endpoint credentials, private
prompts, customer data, or generated model-response dumps.

By contributing, you agree that your contribution is licensed under the MIT
License.
