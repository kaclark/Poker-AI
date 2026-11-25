# Repository Guidelines

## Project Structure & Module Organization

This Python 3.12+ project evolves strategies for heads-up fixed-limit Texas Hold'em. Modules live at the repository root:

- `poker.py`: hand ranking, observations, opponent policies, and game rules.
- `train.py`: NEAT training, model loading, evaluation, and CLI commands.
- `viewer.py`: terminal hand replays.
- `gui.py`: playable Tkinter table.
- `test_poker.py`: game-engine regression tests.
- `neat_config.ini`: network and evolution settings.

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Generated models (`*.pkl`, including `models/` snapshots) are ignored by Git. There is no separate asset directory or build step.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync`: install the locked dependencies into the local environment.
- `uv run python -m unittest discover -v`: run all unit tests.
- `uv run train.py train --generations 2 --hands 10 --watch --delay 0`: run a short training smoke check and save `winner.pkl`.
- `uv run train.py evaluate --model winner.pkl --hands 1000 --seed 9001`: evaluate a saved strategy on seeded deals.
- `uv run gui.py`: launch the table; requires a desktop display and Python with Tcl/Tk support.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Follow existing type annotations and short module/function docstrings. Keep game rules in `poker.py` so training, replays, and human play share behavior. No formatter or linter is currently configured.

## Testing Guidelines

Use standard-library `unittest`, with `test_*.py` files and `test_*` methods. Add regression coverage for rule changes, hand-ranking edge cases, legal actions, observations, and event traces. Use explicit seeds for reproducibility. No coverage threshold is configured. Keep the 12 observation inputs and three action outputs consistent with `neat_config.ini`.

## Commit & Pull Request Guidelines

The available history contains one concise, imperative commit: `Build NEAT poker trainer and playable Tk table`. Follow that style with focused subjects. PRs should explain behavior changes, include validation commands and results, and link relevant issues. Include screenshots for table changes and seeds/evaluation settings for strategy claims.

## Security & Configuration

Load only trusted pickle models: deserialization can execute code. Keep generated models and virtual environments out of commits. Evaluate strategy changes on held-out seeds; performance against the simple baseline does not establish strength against humans.
