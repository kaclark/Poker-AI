"""Tkinter poker table: play against a saved NEAT genome."""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import random
import threading
import time

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:
    tk = None

from poker import play_hand
from train import ROOT, load_network, network_policy
from viewer import HAND_NAMES, RANKS, SUITS


class PokerWindow:
    def __init__(self, root: tk.Tk, initial_model: Path):
        self.root = root
        root.title("Poker-AI — Play against NEAT")
        root.geometry("940x800")
        root.minsize(800, 700)
        root.configure(bg="#12251e")

        self.model_path = tk.StringVar(value=str(initial_model))
        self.speed = tk.DoubleVar(value=0.45)
        self.status = tk.StringVar(value="Choose a model and deal a hand.")
        self.score = tk.StringVar(value="Session: 0 chips over 0 hands")
        self.events: queue.Queue[dict] = queue.Queue()
        self.choices: queue.Queue[str] | None = None
        self.busy = False
        self.hand_number = 0
        self.seed_base = random.randrange(100_000_000)
        self.total = 0
        self.board = ()
        self.human_hole = ()
        self.model_hole = ()
        self.revealed = False
        self.pot = 0
        self.legal: tuple[str, ...] = ()

        top = ttk.Frame(root, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Model").grid(row=0, column=0, padx=(0, 8))
        self.model_box = ttk.Combobox(top, textvariable=self.model_path, width=55)
        self.model_box.grid(row=0, column=1, sticky="ew")
        self.model_box["values"] = tuple(str(path) for path in
                                          sorted((*ROOT.glob("*.pkl"), *ROOT.glob("models/*.pkl"))))
        ttk.Button(top, text="Browse…", command=self.browse_model).grid(row=0, column=2, padx=6)
        self.deal_button = ttk.Button(top, text="Deal hand", command=self.deal)
        self.deal_button.grid(row=0, column=3)
        top.columnconfigure(1, weight=1)

        self.table = tk.Canvas(root, width=900, height=525, bg="#176d4b", highlightthickness=0)
        self.table.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.table.bind("<Configure>", lambda _event: self.draw_table())

        controls = ttk.Frame(root, padding=(12, 0, 12, 8))
        controls.pack(fill="x")
        self.buttons = {}
        for column, (action, label) in enumerate((("fold", "Fold"), ("call", "Check / Call"),
                                                  ("raise", "Bet / Raise"))):
            button = ttk.Button(controls, text=label, command=lambda a=action: self.choose(a))
            button.grid(row=0, column=column, sticky="ew", padx=4)
            controls.columnconfigure(column, weight=1)
            self.buttons[action] = button
        self.update_buttons()

        info = ttk.Frame(root, padding=(12, 0, 12, 8))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.status).pack(anchor="w")
        ttk.Label(info, textvariable=self.score).pack(anchor="w")
        ttk.Label(info, text="Replay pace (seconds per move)").pack(anchor="w")
        ttk.Scale(info, from_=0, to=1.5, variable=self.speed, orient="horizontal").pack(fill="x")

        self.log = tk.Text(root, height=6, bg="#10241d", fg="#e9f5ee", insertbackground="white",
                           state="disabled", wrap="word", relief="flat", padx=12, pady=8)
        self.log.pack(fill="x", padx=12, pady=(0, 12))
        self.draw_table()
        root.after(40, self.poll_events)

    def browse_model(self):
        selected = filedialog.askopenfilename(title="Choose a trained model", initialdir=ROOT,
                                              filetypes=(("NEAT model", "*.pkl"), ("All files", "*")))
        if selected:
            self.model_path.set(selected)

    def deal(self):
        if self.busy:
            return
        model = Path(self.model_path.get()).expanduser()
        try:
            policy = network_policy(load_network(model))
        except Exception as error:
            messagebox.showerror("Cannot load model", str(error))
            return
        self.busy = True
        self.deal_button.configure(state="disabled")
        self.model_box.configure(state="disabled")
        self.choices = queue.Queue()
        self.legal = ()
        self.board = ()
        self.human_hole = ()
        self.model_hole = ()
        self.revealed = False
        self.pot = 2
        self.status.set("Dealing…")
        self.clear_log()
        self.update_buttons()
        self.draw_table()
        seed = self.seed_base + self.hand_number
        self.hand_number += 1
        pace = self.speed.get()
        thread = threading.Thread(target=self.run_hand, args=(policy, seed, pace), daemon=True)
        thread.start()

    def run_hand(self, model_policy, seed: int, pace: float):
        def send(event: dict):
            self.events.put(event)
            if event["kind"] in ("street", "action") and pace:
                time.sleep(pace)

        def human_policy(_features, legal):
            self.events.put({"kind": "decision", "legal": legal})
            assert self.choices is not None
            return self.choices.get()

        try:
            play_hand(model_policy, seed, "human", on_event=send, opponent_policy=human_policy)
        except Exception as error:
            self.events.put({"kind": "error", "message": str(error)})

    def poll_events(self):
        try:
            while True:
                self.handle_event(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(40, self.poll_events)

    def handle_event(self, event: dict):
        kind = event["kind"]
        if kind == "start":
            self.model_hole = event["agent_hole"]
            self.human_hole = event["opponent_hole"]
            self.pot = event["pot"]
            self.status.set("Your cards are ready.")
            self.write_log(f"Hand {self.hand_number} started. Both players ante 1.")
        elif kind == "street":
            self.board = event["board"]
            self.pot = event["pot"]
            name = ("Preflop", "Flop", "Turn", "River")[event["street"]]
            self.status.set(name)
            self.write_log(f"— {name} —")
        elif kind == "action":
            self.pot = event["pot"]
            actor = "Model" if event["actor"] == 0 else "You"
            amount = f" {event['amount']}" if event["amount"] else ""
            self.write_log(f"{actor}: {event['action']}{amount}. Pot {self.pot}.")
        elif kind == "decision":
            self.legal = event["legal"]
            self.status.set("Your turn — choose an action.")
            self.update_buttons()
        elif kind == "end":
            self.revealed = event["showdown"]
            self.total -= event["result"]  # engine result is from the model's perspective
            self.score.set(f"Session: {self.total:+d} chips over {self.hand_number} hands")
            winner = "You" if event["winner"] == 1 else "Model" if event["winner"] == 0 else "Tie"
            self.status.set(f"{winner} wins. Your net: {-event['result']:+d} chips. Deal again?")
            if self.revealed:
                self.write_log(f"Showdown: your {HAND_NAMES[event['opponent_hand'][0]]} vs "
                               f"model's {HAND_NAMES[event['agent_hand'][0]]}.")
            self.write_log(f"{winner} wins. Your net: {-event['result']:+d} chips.")
            self.finish_hand()
        elif kind == "error":
            self.finish_hand()
            messagebox.showerror("Game error", event["message"])
        self.draw_table()

    def finish_hand(self):
        self.busy = False
        self.legal = ()
        self.deal_button.configure(state="normal")
        self.model_box.configure(state="normal")
        self.update_buttons()

    def choose(self, action: str):
        if action not in self.legal or self.choices is None:
            return
        self.legal = ()
        self.update_buttons()
        self.status.set("Model is thinking…")
        self.choices.put(action)

    def update_buttons(self):
        for action, button in self.buttons.items():
            button.configure(state="normal" if action in self.legal else "disabled")
        self.buttons["call"].configure(text="Call" if "fold" in self.legal else "Check")
        self.buttons["raise"].configure(text="Raise" if "fold" in self.legal else "Bet")

    def write_log(self, line: str):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def draw_card(self, x: float, y: float, card=None, hidden=False):
        width, height = 72, 96
        fill = "#244b8c" if hidden else "#faf8f1"
        self.table.create_rectangle(x, y, x + width, y + height, fill=fill,
                                    outline="#d7dfd5", width=2)
        if hidden:
            self.table.create_text(x + width / 2, y + height / 2, text="★",
                                   fill="#e9dca4", font=("DejaVu Sans", 30))
        elif card is not None:
            rank, suit = card
            color = "#c52e38" if suit in (1, 2) else "#182b29"
            self.table.create_text(x + width / 2, y + height / 2,
                                   text=f"{RANKS.get(rank, rank)}{SUITS[suit]}",
                                   fill=color, font=("DejaVu Sans", 23, "bold"))

    def draw_table(self):
        self.table.delete("all")
        width = max(self.table.winfo_width(), 800)
        height = max(self.table.winfo_height(), 500)
        center = width / 2
        self.table.create_oval(50, 20, width - 50, height - 20, outline="#83b895", width=3)
        self.table.create_text(center, 40, text="NEAT MODEL", fill="#f7f4df",
                               font=("DejaVu Sans", 17, "bold"))
        for index in range(2):
            self.draw_card(center - 80 + index * 88, 65,
                           self.model_hole[index] if self.revealed and self.model_hole else None,
                           hidden=bool(self.model_hole and not self.revealed))
        self.table.create_text(center, height / 2 - 84, text=f"POT  {self.pot} chips",
                               fill="#ffe6a6", font=("DejaVu Sans", 19, "bold"))
        for index in range(5):
            self.draw_card(center - 212 + index * 88, height / 2 - 50,
                           self.board[index] if index < len(self.board) else None)
        self.table.create_text(center, height - 174, text="YOU", fill="#f7f4df",
                               font=("DejaVu Sans", 17, "bold"))
        for index in range(2):
            self.draw_card(center - 80 + index * 88, height - 153,
                           self.human_hole[index] if self.human_hole else None)


def main():
    parser = argparse.ArgumentParser(description="Play heads-up poker against a saved NEAT model")
    parser.add_argument("--model", type=Path, default=ROOT / "winner.pkl")
    args = parser.parse_args()
    if tk is None:
        raise SystemExit("Tkinter is required. On Ubuntu, install python3-tk before running gui.py.")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit(f"Cannot open a Tk window: {error}. Run this on a desktop with a display.") from error
    PokerWindow(root, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
