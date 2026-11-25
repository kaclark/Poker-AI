"""Small heads-up, fixed-limit Texas Hold'em environment for NEAT experiments."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random
from typing import Callable

Card = tuple[int, int]  # rank 2..14, suit 0..3
Policy = Callable[[tuple[float, ...], tuple[str, ...]], str]
EventSink = Callable[[dict], None]
STREETS = (0, 3, 1, 1)  # community cards dealt before each betting round
BET_SIZE = (2, 2, 4, 4)
FEATURE_NAMES = (
    "high_hole_rank", "low_hole_rank", "pocket_pair", "suited",
    "street", "made_hand_category", "board_has_pair", "board_max_suit_count",
    "call_cost_over_pot", "pot_size", "raise_available", "acting_first",
)


def rank_five(cards: tuple[Card, ...]) -> tuple[int, ...]:
    """Comparable poker rank: category 0..8 followed by tie breakers."""
    ranks = [rank for rank, _ in cards]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({suit for _, suit in cards}) == 1
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    straight = max((high for high in range(5, 15)
                    if all(rank in unique for rank in range(high - 4, high + 1))), default=0)
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(rank for rank in ranks if rank not in pairs)
        return (2, *pairs, kicker)
    if pairs:
        return (1, pairs[0], *sorted((r for r in ranks if r != pairs[0]), reverse=True))
    return (0, *sorted(ranks, reverse=True))


def best_hand(cards: tuple[Card, ...]) -> tuple[int, ...]:
    if len(cards) < 5:
        raise ValueError("A poker hand needs at least five cards")
    return max(rank_five(hand) for hand in combinations(cards, 5))


def observation(hole: tuple[Card, Card], board: tuple[Card, ...], street: int,
                to_call: int, pot: int, can_raise: bool, acting_first: bool) -> tuple[float, ...]:
    ranks = sorted((hole[0][0], hole[1][0]), reverse=True)
    board_counts = Counter(rank for rank, _ in board)
    board_suits = Counter(suit for _, suit in board)
    made = best_hand(hole + board)[0] if len(board) >= 3 else 0
    return (
        ranks[0] / 14, ranks[1] / 14, float(ranks[0] == ranks[1]),
        float(hole[0][1] == hole[1][1]), street / 3, made / 8,
        float(any(count > 1 for count in board_counts.values())),
        max(board_suits.values(), default=0) / 5,
        to_call / max(pot, 1), min(pot / 40, 1),
        float(can_raise), float(acting_first),
    )


def baseline_policy(style: str, rng: random.Random) -> Policy:
    """Simple stochastic opponent. It sees only its own cards and public information."""
    if style not in {"tight", "loose"}:
        raise ValueError(f"Unknown baseline style: {style}")

    def decide(features: tuple[float, ...], legal: tuple[str, ...]) -> str:
        high, low, pair, suited, street, made = features[:6]
        strength = (high + low) / 2 + 0.28 * pair + 0.07 * suited
        if street > 0:
            strength = 0.22 * strength + 0.78 * made
        threshold = 0.54 if style == "tight" else 0.38
        if "fold" in legal and strength < threshold and rng.random() < (0.85 if style == "tight" else 0.45):
            return "fold"
        if "raise" in legal and strength > threshold + 0.19 and rng.random() < 0.65:
            return "raise"
        return "call"

    return decide


def play_hand(policy: Policy, seed: int, style: str = "tight",
              on_event: EventSink | None = None,
              opponent_policy: Policy | None = None) -> int:
    """Return agent net chips for one hand; each player antes one chip."""
    rng = random.Random(seed)
    deck = [(rank, suit) for suit in range(4) for rank in range(2, 15)]
    rng.shuffle(deck)
    holes = ((deck.pop(), deck.pop()), (deck.pop(), deck.pop()))
    board: tuple[Card, ...] = ()
    paid = [1, 1]
    pot = 2
    opponent = opponent_policy if opponent_policy is not None else baseline_policy(style, rng)

    def emit(kind: str, **details):
        if on_event is not None:
            on_event({"kind": kind, **details})

    emit("start", seed=seed, style=style, agent_hole=holes[0],
         opponent_hole=holes[1], pot=pot)

    for street, new_cards in enumerate(STREETS):
        board += tuple(deck.pop() for _ in range(new_cards))
        emit("street", street=street, board=board, pot=pot)
        actor = seed % 2  # rotate position across evaluation hands
        first_actor = actor
        street_paid = [0, 0]
        raises = 0
        checks = 0
        while True:
            other = 1 - actor
            to_call = street_paid[other] - street_paid[actor]
            legal = ("fold", "call") if to_call else ("call",)
            if raises == 0:
                legal += ("raise",)
            features = observation(holes[actor], board, street, to_call, pot,
                                   raises == 0, actor == first_actor)
            action = policy(features, legal) if actor == 0 else opponent(features, legal)
            if action not in legal:
                raise ValueError(f"Illegal action {action!r}; legal actions: {legal}")
            if action == "fold":
                result = -paid[0] if actor == 0 else pot - paid[0]
                emit("action", actor=actor, action=action, amount=0, pot=pot)
                emit("end", result=result, winner=other, showdown=False)
                return result
            if action == "raise":
                amount = to_call + BET_SIZE[street]
                street_paid[actor] += amount
                paid[actor] += amount
                pot += amount
                raises += 1
                checks = 0
            elif to_call:
                amount = to_call
                street_paid[actor] += to_call
                paid[actor] += to_call
                pot += to_call
                emit("action", actor=actor, action="call", amount=amount, pot=pot)
                break
            else:
                amount = 0
                checks += 1
                emit("action", actor=actor, action="check", amount=0, pot=pot)
                if checks == 2:
                    break
            if action == "raise":
                emit("action", actor=actor, action="raise" if to_call else "bet",
                     amount=amount, pot=pot)
            actor = other

    agent_hand = best_hand(holes[0] + board)
    opponent_hand = best_hand(holes[1] + board)
    winner = 0 if agent_hand > opponent_hand else 1 if agent_hand < opponent_hand else None
    result = (pot - paid[0] if winner == 0 else -paid[0] if winner == 1
              else pot // 2 - paid[0])
    emit("end", result=result, winner=winner, showdown=True,
         opponent_hole=holes[1], agent_hand=agent_hand, opponent_hand=opponent_hand)
    return result
