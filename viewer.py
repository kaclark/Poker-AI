"""Readable, paced terminal replay of a poker hand."""

import time

from poker import play_hand

SUITS = "♣♦♥♠"
RANKS = {11: "J", 12: "Q", 13: "K", 14: "A"}
STREET_NAMES = ("Preflop", "Flop", "Turn", "River")
HAND_NAMES = ("high card", "pair", "two pair", "three of a kind", "straight",
              "flush", "full house", "four of a kind", "straight flush")


def cards_text(cards):
    return " ".join(f"{RANKS.get(rank, rank)}{SUITS[suit]}" for rank, suit in cards)


def watch_hand(policy, seed: int, style: str = "tight", delay: float = 0.35) -> int:
    if delay < 0:
        raise ValueError("delay must be nonnegative")

    def show(event):
        kind = event["kind"]
        if kind == "start":
            print(f"\n=== Hand {event['seed']} vs {event['style']} opponent ===", flush=True)
            print(f"AI: {cards_text(event['agent_hole'])}   Opponent: ?? ??   Pot: {event['pot']}", flush=True)
        elif kind == "street":
            board = cards_text(event["board"]) or "(none)"
            print(f"\n{STREET_NAMES[event['street']]}  |  Board: {board}  |  Pot: {event['pot']}", flush=True)
        elif kind == "action":
            actor = "AI" if event["actor"] == 0 else "Opponent"
            amount = f" {event['amount']}" if event["amount"] else ""
            print(f"  {actor:<8} {event['action']}{amount}  →  pot {event['pot']}", flush=True)
        elif kind == "end":
            if event["showdown"]:
                print(f"  Opponent reveals {cards_text(event['opponent_hole'])}", flush=True)
                print(f"  AI: {HAND_NAMES[event['agent_hand'][0]]}; opponent: {HAND_NAMES[event['opponent_hand'][0]]}", flush=True)
            winner = "AI" if event["winner"] == 0 else "opponent" if event["winner"] == 1 else "tie"
            print(f"  Winner: {winner}  |  AI net: {event['result']:+d} chips", flush=True)
        if delay and kind != "turn":
            time.sleep(delay)

    return play_hand(policy, seed, style, on_event=show)
