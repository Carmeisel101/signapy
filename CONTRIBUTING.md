# Contributing to SignaPy

Thanks for your interest in SignaPy. It's an alpha library, so the API may
still change, and feedback at this stage is especially useful: bug reports,
questions about the statistics, and ideas are all welcome.

## Contact

For anything that doesn't fit a GitHub issue or pull request, email
**dev.meisel+signapy@gmail.com**.

- **Bugs and feature requests:**
  [open an issue](https://github.com/Carmeisel101/signapy/issues). Include the
  SignaPy version, your Python version, and a small example that reproduces
  the problem (a few rows of synthetic data are ideal).
- **Larger changes:** please open an issue or email first, so we can agree on
  the approach before you spend time on it.
- **Security concerns:** email rather than opening a public issue.

## Development setup

```bash
git clone https://github.com/Carmeisel101/signapy.git
cd signapy
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The `dev` extra installs the test and lint tools and the optional `plot` and
`polars` dependencies. If you use [uv](https://docs.astral.sh/uv/), the
equivalent is `uv run --extra dev pytest`.

## Before you open a pull request

CI runs the same checks, on Python 3.10 through 3.14:

```bash
pytest
ruff check .
ruff format --check .
```

Run `ruff format .` to fix formatting. Warnings are errors in the test suite,
because silent numerical problems are unacceptable in a statistics library.

## Guidelines

- **Add tests** for new behavior and for bug fixes. Tests live in `tests/`.
- **Keep the docs in step with the code.** User-facing changes go in the
  README, `docs/metrics.md` (what a statistic means and how to read it), or
  `docs/architecture.md` (design and scope), and get an entry under
  "Unreleased" in `CHANGELOG.md`.
- **Be honest about what the statistics show.** SignaPy reports associations
  in the sample, not causes. Wording in docs, docstrings, and plot text should
  not imply more than the numbers support.
- **Keep changes focused.** One concern per pull request is easier to review.
- **Match the surrounding code:** naming, comment density, and typing.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
