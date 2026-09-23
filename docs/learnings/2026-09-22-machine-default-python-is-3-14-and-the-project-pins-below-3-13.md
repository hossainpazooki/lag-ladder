# The machine's default python is 3.14 and the project pins below 3.13: build the venv from an explicit 3.12

ts: 2026-09-22T23:09:31Z
commit: e79d3ad4789467cec39ffb5ca733a8b07a0c0173
session: claude-code session 77a5d0ae-ac85-4e14-8091-8829150e0696 ("async-rl-holdover-design"), scaffold step
status: verified
fact: `python -m venv .venv` on this Windows machine produces a 3.14.2 interpreter, and `pip install -e .` then refuses with "Package 'lag-ladder' requires a different Python: 3.14.2 not in '<3.13,>=3.12'". The working venv was built from `py -3.12` (3.12.14); linear-ceiling's `.venv` is the same interpreter. The pin is inherited from linear-ceiling's pyproject and is deliberate (tomllib / torch wheel compatibility on the GPU box, which also runs 3.12).
basis: `python --version` → `Python 3.14.2`; `grep -n requires-python pyproject.toml` → `5:requires-python = ">=3.12,<3.13"` (captured 2026-09-22T23:09:31Z at e79d3ad; the pip refusal was observed in the same session's first install attempt).
re-verify: python --version; grep -n requires-python pyproject.toml
