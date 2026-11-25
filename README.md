# Poker-AI

A small, reproducible experiment in evolving poker decisions with **NEAT** (NeuroEvolution of Augmenting Topologies). Inspired by [Blackjack-AI](https://github.com/kaclark/Blackjack-AI), this is a separate project ready for its own GitHub repository.

The game is heads-up **fixed-limit Texas Hold'em**. Both players ante one chip. Each street allows one bet or raise of 2 chips before the turn and 4 chips on the turn and river. Legal choices are fold, check/call, and bet/raise. At showdown, the best five of seven cards wins. There are no stack sizes, blinds, side pots, or all-in actions; the goal is a compact strategy experiment rather than a full casino rules engine.

Each genome is a feed-forward network with 12 normalized inputs (hole-card ranks, pair and suited flags, street, current made-hand category, board pair and suit texture, call price, pot size, raise availability, and position) and three action scores. Only legal actions compete for the highest score. The opponent is a simple stochastic tight or loose heuristic that sees only its own cards and public information. Fitness is **average net chips per hand** on a shared set of seeded deals. The seed also controls NEAT's mutations. Deals change each generation; a separate held-out seed evaluates the winner.

## Run

```bash
uv sync
uv run train.py train --generations 30 --hands 100
uv run train.py evaluate --model winner.pkl --hands 1000 --seed 9001
uv run train.py watch --model winner.pkl --hands 5
```

## Play against a model

Launch the Tcl/Tk table with:

```bash
uv run gui.py
```

Select any trained `.pkl` model in the picker (or use `--model path/to/model.pkl`), then click **Deal hand**. Your cards appear at the bottom; the model's cards stay face down until showdown. Click the enabled Fold, Check/Call, or Bet/Raise button when it is your turn. The table shows the pot, board, action log, and your session winnings. You can save different training runs with `train --output another-model.pkl` and switch between them before dealing.

To play against models from different stages of learning, train with snapshots:

```bash
uv run train.py train --generations 30 --hands 100 --watch --save-every 5
```

The picker will include `models/generation_005.pkl`, `generation_010.pkl`, and so on. Snapshots and the final winner are local files and are excluded from Git.

Tkinter requires a desktop display and the system Tk package. On Ubuntu, install it with `sudo apt install python3-tk`; `uv sync` installs the Python dependency but not the operating system's Tk library. Load only model files you trust.

Training prints population fitness each generation. To **watch learning live**, add `--watch` to the training command. It also reports each generation's champion on the same fixed benchmark deals, so the scores are comparable across generations. It replays one hand from that champion, showing the AI's hole cards, community cards, both players' actions, pot, and result. The opponent's hole cards stay hidden until showdown. `watch` replays a saved winner for as many hands as you like. Set `--delay 0` for instant output or a larger number for slower playback.

For a quick smoke run, use `--generations 2 --hands 10 --watch --delay 0`. More hands reduce fitness noise, at the cost of longer runs. The training command saves `winner.pkl`; load pickle files only from trusted sources.

The baseline opponent is intentionally simple. Positive results against it do not establish strong play against skilled humans or other poker agents. For better strategy research, add stronger opponent policies and evaluate across more held-out deals.

## Publish

After creating an empty `kaclark/Poker-AI` repository on GitHub:

```bash
git remote add origin git@github.com:kaclark/Poker-AI.git
git push -u origin main
```
