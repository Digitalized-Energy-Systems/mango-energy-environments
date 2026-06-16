# Contributing to mango-energy-environments

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. Please read the relevant section before making your contribution — it will make things smoother for everyone.

> If you like the project but don't have time to contribute, that's fine. Other easy ways to show appreciation:
> - Star the project
> - Mention it to colleagues
> - Send feedback to <rico.schrage@uol.de> or open a GitHub issue

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving Documentation](#improving-documentation)
- [Style Guide](#style-guide)

---

## Code of Conduct

This project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it. Report unacceptable behavior to <rico.schrage@uol.de>.

---

## I Have a Question

Before asking, search existing [issues](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/issues) that might already answer your question.

If you still need help:

- Open an issue with the label `question`.
- Provide as much context as you can (Python version, OS, relevant code, full traceback).

---

## I Want To Contribute

> **Legal notice:** By contributing you confirm that you have authored 100% of the content, have the necessary rights, and agree that the content may be provided under the project license.

### Reporting Bugs

Before filing:

- Make sure you are on the latest version.
- Check whether this is really a bug and not a mis-configuration.
- Search [existing bug reports](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/issues?q=label%3Abug).

When filing:

- Open an issue (do not label it a bug yet — we will do that after triage).
- Describe expected vs. actual behavior.
- Include a minimal reproduction: Python version, OS, installed package versions, code snippet, and full traceback.

### Suggesting Enhancements

Enhancement suggestions are tracked as [GitHub issues](https://github.com/Digitalized-Energy-Systems/mango-energy-environments/issues).

- Use a clear, descriptive title.
- Describe the current behavior and what you expect instead.
- Explain why this would be useful to most users.
- Check whether something similar already exists.

### Your First Code Contribution

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/mango-energy-environments.git
cd mango-energy-environments

# 2. Install in editable mode with dev extras
pip install -e ".[dev]"

# 3. Create a branch
git checkout -b my-feature

# 4. Make changes, write tests
# 5. Run the test suite
pytest tests/ -v

# 6. Push and open a pull request
```

Requirements for a PR to be merged:

- All existing tests pass (`pytest tests/`).
- New functionality is covered by tests.
- Code follows the style guide below.
- CI (GitHub Actions) passes.

### Improving Documentation

For minor fixes (typos, clarifications) open a PR directly.  For larger reworks, open an issue first to discuss the changes.

---

## Style Guide

This project follows [PEP 8](https://peps.python.org/pep-0008/) with the following conventions:

- **Formatter:** [`ruff format`](https://docs.astral.sh/ruff/) (or `black`).
- **Linter:** [`ruff check`](https://docs.astral.sh/ruff/).
- **Type hints:** Required for all public functions and class attributes.
- **Docstrings:** NumPy style for modules, classes, and public methods.
- **Imports:** Standard library → third-party → local, sorted alphabetically within each group.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for module-level constants.
- **Tests:** `pytest`, one file per source module, named `test_<module>.py`.

---

*This guide is based on the contributing template used by MangoEnergyEnvironments.jl.*
