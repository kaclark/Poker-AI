"""Tkinter poker table: play against a saved NEAT genome."""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import random
import sys
import threading
import time

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ModuleNotFoundError:
    tk = None

from poker import Action, BET_SIZE, MAX_BET, play_hand
from train import ROOT, load_network, network_policy
from viewer import HAND_NAMES, RANKS, SUITS


class PokerWindow:
    def __init__(self, root: tk.Tk, initial_model: Path):
        self.root = root
        root.title("Poker-AI — Play against NEAT")
        root.geometry("1120x740")
        root.minsize(1000, 660)
        root.configure(bg="#0e1d19")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self.configure_styles()

        self.model_path = tk.StringVar(value=self.model_label(initial_model))
        self.speed = tk.DoubleVar(value=0.45)
        self.status = tk.StringVar(value="Choose a model and deal a hand.")
        self.score = tk.StringVar(value="Session: 0 chips over 0 hands")
        self.bet_amount = tk.StringVar(value="2")
        self.bet_hint = tk.StringVar(value="One bet per street · no stacks or all-ins")
        self.animations = tk.BooleanVar(value=True)
        self.min_bet = BET_SIZE[0]
        self.to_call = 0
        self.overlay = None
        self.overlay_job = None
        self.chip_job = None
        self.chip = None
        self.events: queue.Queue[dict] = queue.Queue()
        self.choices: queue.Queue[Action] | None = None
        self.busy = False
        self.hand_number = 0
        self.seed_base = random.randrange(100_000_000)
        self.total = 0
        self.board = ()
        self.human_hole = ()
        self.model_hole = ()
        self.revealed = False
        self.pot = 0
        self.street_name = "READY TO PLAY"
        self.legal: tuple[str, ...] = ()

        top = ttk.Frame(root, padding=(22, 16))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="POKER / AI", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Heads-up Hold’em · Play against an evolved strategy",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(top, text="OPPONENT", style="Caption.TLabel").grid(row=0, column=1, sticky="w")
        self.model_box = ttk.Combobox(top, textvariable=self.model_path, width=27)
        self.model_box.grid(row=1, column=1, sticky="ew", pady=(4, 0))
        self.model_box["values"] = tuple(self.model_label(path) for path in
                                          sorted((*ROOT.glob("*.pkl"), *ROOT.glob("models/*.pkl"))))
        ttk.Button(top, text="Browse…", command=self.browse_model).grid(row=1, column=2, padx=(10, 0))
        top.columnconfigure(0, weight=1)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", padx=18)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.table = tk.Canvas(body, width=720, height=540, bg="#123c32", highlightthickness=0)
        self.table.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.table.bind("<Configure>", lambda _event: self.draw_table())

        sidebar = ttk.Frame(body, padding=18, style="Panel.TFrame", width=304)
        sidebar.grid(row=0, column=1, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, minsize=58)
        sidebar.rowconfigure(9, weight=1)
        ttk.Label(sidebar, text="YOUR MOVE", style="PanelCaption.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self.status, wraplength=268, justify="left",
                  style="Status.TLabel").grid(row=1, column=0, sticky="ew", pady=(8, 10))
        ttk.Label(sidebar, textvariable=self.score, style="PanelMuted.TLabel").grid(row=2, column=0, sticky="w")

        sizing = ttk.Frame(sidebar, style="Panel.TFrame")
        sizing.grid(row=3, column=0, sticky="ew", pady=(14, 8))
        sizing.columnconfigure(0, weight=1)
        ttk.Label(sizing, text="BET AMOUNT", style="PanelCaption.TLabel").grid(row=0, column=0, sticky="w")
        self.bet_entry = ttk.Spinbox(sizing, from_=2, to=MAX_BET, increment=1,
                                     textvariable=self.bet_amount, width=6)
        self.bet_entry.grid(row=0, column=1, sticky="e")
        presets = ttk.Frame(sidebar, style="Panel.TFrame")
        presets.grid(row=4, column=0, sticky="ew")
        self.bet_presets = []
        for label, amount in (("Min", None), ("10", 10), ("25", 25), ("100", 100)):
            column = len(self.bet_presets)
            button = ttk.Button(presets, text=label, width=4, style="Preset.TButton",
                                command=lambda n=amount: self.bet_amount.set(str(n or self.min_bet)))
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
            presets.columnconfigure(column, weight=1)
            self.bet_presets.append(button)
        ttk.Label(sidebar, textvariable=self.bet_hint, style="PanelMuted.TLabel").grid(row=5, column=0, sticky="w", pady=(7, 10))

        controls = ttk.Frame(sidebar, style="Panel.TFrame")
        controls.grid(row=6, column=0, sticky="ew")
        self.buttons = {}
        for column, (action, label) in enumerate((("fold", "Fold"), ("call", "Check"), ("raise", "Bet"))):
            button = ttk.Button(controls, text=label, width=5,
                                style="Bet.TButton" if action == "raise" else "TButton",
                                command=lambda a=action: self.choose(a))
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
            controls.columnconfigure(column, weight=1, uniform="actions")
            self.buttons[action] = button
        self.deal_button = ttk.Button(sidebar, text="Deal hand  →", style="Primary.TButton", command=self.deal)
        self.deal_button.grid(row=7, column=0, sticky="ew", pady=(10, 14))
        self.update_buttons()

        ttk.Label(sidebar, text="HAND HISTORY", style="PanelCaption.TLabel").grid(row=8, column=0, sticky="w", pady=(0, 8))
        history = ttk.Frame(sidebar, style="Panel.TFrame")
        history.grid(row=9, column=0, sticky="nsew")
        history.columnconfigure(0, weight=1)
        history.rowconfigure(0, weight=1)
        self.log = tk.Text(history, width=26, height=4, bg="#132620", fg="#bed1c6",
                           insertbackground="white", state="disabled", wrap="word", relief="flat",
                           font=("Helvetica", 11), padx=10, pady=10, spacing3=4, highlightthickness=0)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(history, command=self.log.yview, style="Vertical.TScrollbar")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        info = ttk.Frame(sidebar, style="Panel.TFrame")
        info.grid(row=10, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(info, text="Animate table updates", variable=self.animations).pack(anchor="w")
        pace = ttk.Frame(info, style="Panel.TFrame")
        pace.pack(fill="x", pady=(8, 0))
        ttk.Label(pace, text="Pace", style="PanelMuted.TLabel").pack(side="left", padx=(0, 10))
        ttk.Scale(pace, from_=0, to=1.5, variable=self.speed, orient="horizontal").pack(side="left", fill="x", expand=True)
        ttk.Label(root, text="1 CHIP ANTE    /    ONE BET PER STREET    /    NO STACKS OR ALL-INS",
                  style="Caption.TLabel", padding=(22, 12)).grid(row=2, column=0, sticky="w")
        self.draw_table()
        root.after(40, self.poll_events)

    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Helvetica", 11), background="#0e1d19", foreground="#edf3e9")
        style.configure("TFrame", background="#0e1d19")
        style.configure("TLabel", background="#0e1d19")
        style.configure("Title.TLabel", font=("Helvetica", 21, "bold"), foreground="#eed29a")
        style.configure("Muted.TLabel", foreground="#91aa9e")
        style.configure("Caption.TLabel", font=("Helvetica", 9, "bold"), foreground="#91aa9e")
        style.configure("Panel.TFrame", background="#1a3028")
        style.configure("PanelCaption.TLabel", background="#1a3028", foreground="#a9c1b2", font=("Helvetica", 9, "bold"))
        style.configure("PanelMuted.TLabel", background="#1a3028", foreground="#a9c1b2", font=("Helvetica", 10))
        style.configure("Status.TLabel", background="#1a3028", font=("Helvetica", 12), foreground="#edf3e9")
        style.configure("TButton", padding=(8, 9), background="#2b493c", foreground="#edf3e9", borderwidth=0)
        style.map("TButton", background=[("disabled", "#223b30"), ("active", "#3b5e4c")],
                  foreground=[("disabled", "#8ba396")])
        style.configure("Primary.TButton", background="#e5c185", foreground="#19281f", font=("Helvetica", 12, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#405044"), ("active", "#f2d69f")],
                  foreground=[("disabled", "#a2b3a8")])
        style.configure("Bet.TButton", background="#397657")
        style.map("Bet.TButton", background=[("disabled", "#223b30"), ("active", "#4b906c")])
        style.configure("Preset.TButton", padding=(3, 6))
        for widget in ("TCombobox", "TSpinbox"):
            style.configure(widget, fieldbackground="#12251e", background="#2b493c", foreground="#edf3e9",
                            arrowcolor="#c9dccc", bordercolor="#405d4c", lightcolor="#405d4c",
                            darkcolor="#405d4c", padding=6)
            style.map(widget, fieldbackground=[("disabled", "#1b3027")], foreground=[("disabled", "#8ba396")])
        style.configure("TCheckbutton", background="#1a3028", foreground="#a9c1b2")
        style.map("TCheckbutton", background=[("active", "#1a3028")], indicatorbackground=[("selected", "#e5c185")])
        style.configure("Horizontal.TScale", background="#75917e", troughcolor="#10251c",
                        bordercolor="#1a3028", lightcolor="#75917e", darkcolor="#75917e")
        style.configure("Vertical.TScrollbar", background="#375243", troughcolor="#132620",
                        bordercolor="#132620", lightcolor="#375243", darkcolor="#375243",
                        arrowcolor="#a9c1b2", borderwidth=0, arrowsize=10, width=10)
        style.map("Vertical.TScrollbar", background=[("active", "#52715c")])
        self.root.option_add("*TCombobox*Listbox.background", "#1a3028")
        self.root.option_add("*TCombobox*Listbox.foreground", "#edf3e9")

    @staticmethod
    def model_label(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def browse_model(self):
        selected = filedialog.askopenfilename(title="Choose a trained model", initialdir=ROOT,
                                              filetypes=(("NEAT model", "*.pkl"), ("All files", "*")))
        if selected:
            self.model_path.set(self.model_label(Path(selected)))

    def deal(self):
        if self.busy:
            return
        model = Path(self.model_path.get()).expanduser()
        if not model.is_absolute():
            model = ROOT / model
        try:
            policy = network_policy(load_network(model))
        except Exception as error:
            self.show_overlay("Cannot load model", str(error), "#f3a6a6", duration=6000)
            return
        self.clear_effects()
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
        self.street_name = "DEALING"
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
            self.street_name = name.upper()
            self.status.set(name)
            self.write_log(f"— {name} —")
            self.show_overlay(name, "Choose your bet when it is your turn.", duration=1000)
        elif kind == "action":
            self.pot = event["pot"]
            actor = "Model" if event["actor"] == 0 else "You"
            amount = f" {event['amount']}" if event["amount"] else ""
            self.write_log(f"{actor}: {event['action']}{amount}. Pot {self.pot}.")
            self.status.set(f"{actor}: {event['action']}{amount}.")
            if event["amount"]:
                self.animate_chips(event["actor"], event["amount"])
        elif kind == "turn" and event["actor"] == 1:
            self.min_bet = event["min_bet"]
            self.to_call = event["to_call"]
            self.bet_entry.configure(from_=self.min_bet)
            try:
                amount = int(self.bet_amount.get())
                self.bet_amount.set(str(max(self.min_bet, min(MAX_BET, amount))))
            except ValueError:
                self.bet_amount.set(str(self.min_bet))
        elif kind == "decision":
            self.legal = event["legal"]
            self.status.set("Your turn — choose an action.")
            self.update_buttons()
        elif kind == "end":
            self.revealed = event["showdown"]
            self.total -= event["result"]  # engine result is from the model's perspective
            self.score.set(f"Session: {self.total:+d} chips over {self.hand_number} hands")
            winner = "You win" if event["winner"] == 1 else "Model wins" if event["winner"] == 0 else "Split pot"
            self.status.set(f"{winner}. Your net: {-event['result']:+d} chips. Deal again?")
            detail = "Hand ended by a fold."
            if self.revealed:
                detail = (f"Your {HAND_NAMES[event['opponent_hand'][0]]} vs "
                          f"model's {HAND_NAMES[event['agent_hand'][0]]}.")
                self.write_log(f"Showdown: your {HAND_NAMES[event['opponent_hand'][0]]} vs "
                               f"model's {HAND_NAMES[event['agent_hand'][0]]}.")
            self.write_log(f"{winner}. Your net: {-event['result']:+d} chips.")
            self.finish_hand()
            color = "#9aefbd" if event["winner"] == 1 else "#f3a6a6" if event["winner"] == 0 else "#ffe6a6"
            self.show_overlay(winner, f"Your net: {-event['result']:+d} chips\n{detail}", color, duration=5500)
        elif kind == "error":
            self.finish_hand()
            self.status.set(f"Game error: {event['message']}")
            self.show_overlay("Game error", event["message"], "#f3a6a6", duration=6000)
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
        choice: Action = action
        if action == "raise":
            try:
                amount = int(self.bet_amount.get())
                if not self.min_bet <= amount <= MAX_BET:
                    raise ValueError
            except ValueError:
                self.status.set(f"Enter a whole-chip bet from {self.min_bet} to {MAX_BET}.")
                self.bet_entry.focus_set()
                return
            choice = (action, amount)
        self.dismiss_overlay()
        self.legal = ()
        self.update_buttons()
        self.status.set("Model is thinking…")
        self.choices.put(choice)

    def update_buttons(self):
        for action, button in self.buttons.items():
            button.configure(state="normal" if action in self.legal else "disabled")
        self.buttons["call"].configure(text=f"Call {self.to_call}" if "fold" in self.legal else "Check")
        self.buttons["raise"].configure(text="Raise" if "fold" in self.legal else "Bet")
        state = "normal" if "raise" in self.legal else "disabled"
        self.bet_entry.configure(state=state)
        for button in self.bet_presets:
            button.configure(state=state)
        self.bet_hint.set(f"{self.min_bet}–{MAX_BET} chips · whole amounts only")

    def dismiss_overlay(self):
        if self.overlay_job is not None:
            self.root.after_cancel(self.overlay_job)
            self.overlay_job = None
        self.overlay = None
        self.table.delete("overlay")

    def show_overlay(self, title: str, detail: str, color="#ffe6a6", duration=1000):
        """Slide in a dismissible canvas notice without taking focus or a grab."""
        self.dismiss_overlay()
        self.overlay = (title, detail, color, 0.0 if self.animations.get() else 1.0)
        started = time.monotonic()

        def tick():
            elapsed = (time.monotonic() - started) * 1000
            if elapsed >= duration:
                self.overlay_job = None
                self.dismiss_overlay()
                return
            progress = min(1.0, elapsed / 220) if self.animations.get() else 1.0
            self.overlay = (title, detail, color, progress)
            self.draw_overlay()
            self.overlay_job = self.root.after(16 if progress < 1 else 100, tick)

        tick()

    def draw_overlay(self):
        self.table.delete("overlay")
        if self.overlay is None or self.table.winfo_width() < 10:
            return
        title, detail, color, progress = self.overlay
        center = self.table.winfo_width() / 2
        y = self.table.winfo_height() / 2 - 70 - 24 * (1 - progress) ** 3
        half_width = min(240, self.table.winfo_width() / 2 - 24)
        self.rounded_rect(center - half_width, y, center + half_width, y + 140,
                          fill="#102b25", outline=color, width=2, tags="overlay")
        self.table.create_text(center, y + 28, text=title, fill=color,
                               font=("Helvetica", 22, "bold"), tags="overlay")
        self.table.create_text(center, y + 74, text=detail, fill="#f7f4df", width=half_width * 2 - 28,
                               justify="center", font=("Helvetica", 12), tags="overlay")
        self.table.create_text(center, y + 120, text="Click to dismiss · or keep playing",
                               fill="#a8c6b9", font=("Helvetica", 10), tags="overlay")
        self.table.tag_bind("overlay", "<Button-1>", lambda _event: self.dismiss_overlay())

    def animate_chips(self, actor: int, amount: int):
        if self.chip_job is not None:
            self.root.after_cancel(self.chip_job)
            self.chip_job = None
        self.chip = None
        self.table.delete("chips")
        if not self.animations.get():
            return
        started = time.monotonic()

        def tick():
            progress = min(1.0, (time.monotonic() - started) / 0.4)
            self.chip = (actor, amount, progress)
            self.draw_chips()
            if progress < 1:
                self.chip_job = self.root.after(16, tick)
            else:
                self.chip_job = None
                self.chip = None
                self.table.delete("chips")

        tick()

    def draw_chips(self):
        self.table.delete("chips")
        if self.chip is None:
            return
        actor, amount, progress = self.chip
        width = self.table.winfo_width()
        height = self.table.winfo_height()
        x = width / 2 + min(160, width * 0.28)
        start = height * (0.22 if actor == 0 else 0.82)
        target = height / 2 - min(72, height * 0.12) * 0.69 - 36
        y = start + (target - start) * (1 - (1 - progress) ** 3)
        self.table.create_oval(x - 23, y - 23, x + 23, y + 23, fill="#e0b65d",
                               outline="#fff0c0", width=3, tags="chips")
        self.table.create_text(x, y, text=str(amount), fill="#332709",
                               font=("Helvetica", 12, "bold"), tags="chips")
        self.table.tag_raise("overlay")

    def clear_effects(self):
        self.dismiss_overlay()
        if self.chip_job is not None:
            self.root.after_cancel(self.chip_job)
            self.chip_job = None
        self.chip = None
        self.table.delete("chips")

    def write_log(self, line: str):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def rounded_rect(self, x1, y1, x2, y2, radius=12, **options):
        points = (x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
                  x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
                  x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1)
        return self.table.create_polygon(points, smooth=True, **options)

    def draw_card(self, x: float, y: float, card=None, hidden=False, width=72, tags="card"):
        height = width * 1.38
        if card is None and not hidden:
            self.rounded_rect(x, y, x + width, y + height, radius=8,
                              fill="#123c32", outline="#3a5e4e", width=1, dash=(3, 5), tags=tags)
            return
        self.rounded_rect(x + 2, y + 3, x + width + 2, y + height + 3, radius=8,
                          fill="#0b2c24", outline="", tags=tags)
        self.rounded_rect(x, y, x + width, y + height, radius=8,
                          fill="#234c3c" if hidden else "#faf7ed", outline="#8fae94" if hidden else "#fffdf6", tags=tags)
        if hidden:
            self.rounded_rect(x + 6, y + 6, x + width - 6, y + height - 6, radius=5,
                              fill="", outline="#52735b", tags=tags)
            self.table.create_text(x + width / 2, y + height / 2, text="★",
                                   fill="#a7b28a", font=("Helvetica", int(width * 0.4)), tags=tags)
        elif card is not None:
            rank, suit = card
            color = "#b34544" if suit in (1, 2) else "#203e30"
            self.table.create_text(x + 10, y + 9, anchor="nw", text=str(RANKS.get(rank, rank)),
                                   fill=color, font=("Helvetica", int(width * 0.25), "bold"), tags=tags)
            self.table.create_text(x + width / 2, y + height * 0.62, text=SUITS[suit],
                                   fill=color, font=("Helvetica", int(width * 0.43)), tags=tags)

    def draw_table(self):
        self.table.delete("all")
        width = self.table.winfo_width()
        height = self.table.winfo_height()
        if min(width, height) < 10:
            return
        center = width / 2
        card_width = min(72, height * 0.12, width / 9)
        card_height = card_width * 1.38
        gap = 12
        model_y = height * 0.11
        board_y = height / 2 - card_height / 2 + 12
        human_y = height - card_height - height * 0.09
        self.table.create_oval(18, 18, width - 18, height - 18, outline="#577653", width=2)
        self.table.create_oval(25, 25, width - 25, height - 25, outline="#294e3d", width=1)
        self.table.create_text(center, model_y - 22, text="NEAT MODEL", fill="#a9c5b0",
                               font=("Helvetica", 10, "bold"), tags="model_label")
        for index in range(2):
            self.draw_card(center - card_width - gap / 2 + index * (card_width + gap), model_y,
                           self.model_hole[index] if self.revealed and self.model_hole else None,
                           hidden=bool(self.model_hole and not self.revealed), width=card_width, tags="model_cards")
        pot_y = board_y - 48
        self.rounded_rect(center - 78, pot_y - 18, center + 78, pot_y + 18,
                          fill="#243f2e", outline="#6b7950", tags="pot")
        self.table.create_text(center, pot_y, text=f"POT   {self.pot} chips", fill="#f0d49b",
                               font=("Helvetica", 13, "bold"), tags="pot")
        start = center - (5 * card_width + 4 * gap) / 2
        for index in range(5):
            self.draw_card(start + index * (card_width + gap), board_y,
                           self.board[index] if index < len(self.board) else None,
                           width=card_width, tags="board_cards")
        self.table.create_text(center, board_y + card_height + 24, text=self.street_name, fill="#86ab94",
                               font=("Helvetica", 9, "bold"), tags="street_label")
        self.table.create_text(center, human_y - 22, text="YOU", fill="#a9c5b0",
                               font=("Helvetica", 10, "bold"), tags="human_label")
        for index in range(2):
            self.draw_card(center - card_width - gap / 2 + index * (card_width + gap), human_y,
                           self.human_hole[index] if self.human_hole else None, width=card_width, tags="human_cards")
        self.draw_chips()
        self.draw_overlay()


def main():
    parser = argparse.ArgumentParser(description="Play heads-up poker against a saved NEAT model")
    parser.add_argument("--model", type=Path, default=ROOT / "winner.pkl")
    args = parser.parse_args()
    if tk is None:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.platform == "darwin":
            hint = (f"For Homebrew Python, run: brew install python-tk@{version}. "
                    "Otherwise, use a Python installation with Tcl/Tk support.")
        elif sys.platform.startswith("linux"):
            hint = ("On Ubuntu/Debian, install python3-tk (matching your Python version). "
                    "Use a Python installation with Tcl/Tk support.")
        else:
            hint = "Install Python with Tcl/Tk support enabled."
        raise SystemExit(f"Tkinter is required for Python {version} ({sys.executable}). {hint}")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit(f"Cannot open a Tk window: {error}. Run this on a desktop with a display.") from error
    PokerWindow(root, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
