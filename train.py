"""Train and evaluate a NEAT poker policy. Run ``python3 train.py --help``."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import random

import neat

from poker import play_hand
from viewer import watch_hand

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "neat_config.ini"


def load_config() -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(CONFIG))


def network_policy(network):
    def decide(features, legal):
        scores = network.activate(features)
        return max(legal, key=lambda action: scores[("fold", "call", "raise").index(action)])
    return decide


def score(network, hands: int, seed: int) -> float:
    policy = network_policy(network)
    # Both styles and positions are represented in the same evaluation set.
    return sum(play_hand(policy, seed + i, "tight" if i % 2 else "loose")
               for i in range(hands)) / hands


def train(args):
    random.seed(args.seed)
    config = load_config()
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    generation = 0

    def evaluate(genomes, current_config):
        nonlocal generation
        # All genomes face the same deals. Change deals each generation to curb memorization.
        deal_seed = args.seed + generation * args.hands
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, current_config)
            genome.fitness = score(net, args.hands, deal_seed)
        champion = max((genome for _, genome in genomes), key=lambda genome: genome.fitness)
        if args.save_every and (generation + 1) % args.save_every == 0:
            snapshot_dir = Path(args.snapshots_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            with (snapshot_dir / f"generation_{generation + 1:03d}.pkl").open("wb") as file:
                pickle.dump(champion, file)
        if args.watch:
            sample = neat.nn.FeedForwardNetwork.create(champion, current_config)
            benchmark = score(sample, 100, args.seed + 1_000_000)
            print(f"\nGeneration {generation + 1} champion: {champion.fitness:+.3f} training chips/hand; "
                  f"{benchmark:+.3f} on fixed benchmark")
            watch_hand(network_policy(sample), args.seed + 500_000 + generation,
                       "tight" if generation % 2 else "loose", args.delay)
        generation += 1

    winner = population.run(evaluate, args.generations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(winner, file)
    print(f"Saved winner to {output}")
    validation_seed = args.seed + args.generations * args.hands + 1_000_000
    print(f"Held-out mean chips/hand: {score(neat.nn.FeedForwardNetwork.create(winner, config), args.hands, validation_seed):+.3f}")


def load_network(model):
    with Path(model).open("rb") as file:
        genome = pickle.load(file)  # Load only model files you trust.
    return neat.nn.FeedForwardNetwork.create(genome, load_config())


def evaluate(args):
    network = load_network(args.model)
    print(f"Mean chips/hand over {args.hands} hands: {score(network, args.hands, args.seed):+.3f}")


def watch(args):
    policy = network_policy(load_network(args.model))
    total = 0
    for i in range(args.hands):
        style = args.style if args.style != "mixed" else "tight" if i % 2 else "loose"
        total += watch_hand(policy, args.seed + i, style, args.delay)
    print(f"\nSession result: {total:+d} chips over {args.hands} {'hand' if args.hands == 1 else 'hands'}")


def main():
    parser = argparse.ArgumentParser(description="NEAT for heads-up fixed-limit Texas Hold'em")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Evolve a strategy")
    training.add_argument("--generations", type=int, default=30)
    training.add_argument("--hands", type=int, default=100, help="hands per genome per generation")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", default=str(ROOT / "winner.pkl"))
    training.add_argument("--watch", action="store_true", help="show a champion hand after each generation")
    training.add_argument("--delay", type=float, default=0.35, help="seconds between replay events")
    training.add_argument("--save-every", type=int, default=0, help="save a champion every N generations")
    training.add_argument("--snapshots-dir", default=str(ROOT / "models"))
    testing = sub.add_parser("evaluate", help="Evaluate a saved strategy")
    testing.add_argument("--model", default=str(ROOT / "winner.pkl"))
    testing.add_argument("--hands", type=int, default=1000)
    testing.add_argument("--seed", type=int, default=9001)
    watching = sub.add_parser("watch", help="watch a saved strategy play")
    watching.add_argument("--model", default=str(ROOT / "winner.pkl"))
    watching.add_argument("--hands", type=int, default=3)
    watching.add_argument("--seed", type=int, default=9001)
    watching.add_argument("--style", choices=("mixed", "tight", "loose"), default="mixed")
    watching.add_argument("--delay", type=float, default=0.35, help="seconds between events")
    args = parser.parse_args()
    if args.hands < 1 or (args.command == "train" and args.generations < 1):
        parser.error("hands and generations must be positive")
    if args.command == "train" and args.save_every < 0:
        parser.error("save-every must be nonnegative")
    if hasattr(args, "delay") and args.delay < 0:
        parser.error("delay must be nonnegative")
    {"train": train, "evaluate": evaluate, "watch": watch}[args.command](args)


if __name__ == "__main__":
    main()
