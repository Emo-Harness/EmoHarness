# Contributing

Read `docs/engineering.md` before writing code and `docs/adr/README.md` before changing a design decision. Every change goes through a pull request that passes CI (ruff, mypy --strict, pytest, codespell). Design changes need a new ADR or an amendment to an existing one; never edit a frozen bundle in `harness/` or `bundles/`: any change is a new bundle with a new content hash.
