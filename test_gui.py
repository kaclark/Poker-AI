"""Table interaction tests; require Tk and a desktop display."""

from pathlib import Path
import queue
import time
import unittest

from gui import PokerWindow, tk


class TableTests(unittest.TestCase):
    def setUp(self):
        if tk is None:
            self.skipTest("Tkinter is unavailable")
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("A desktop display is unavailable")
        self.root.withdraw()
        self.app = PokerWindow(self.root, Path("winner.pkl"))

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.clear_effects()
        if hasattr(self, "root"):
            self.root.destroy()

    def test_invalid_bet_keeps_turn_and_valid_bet_submits_once(self):
        app = self.app
        app.choices = queue.Queue()
        app.legal = ("call", "raise")
        for value in ("", "abc", "2.5", "0", "101"):
            app.bet_amount.set(value)
            app.choose("raise")
            self.assertEqual(app.legal, ("call", "raise"))
            self.assertTrue(app.choices.empty())
        app.bet_amount.set("25")
        app.choose("raise")
        app.choose("raise")
        self.assertEqual(app.choices.get_nowait(), ("raise", 25))
        self.assertTrue(app.choices.empty())

    def test_call_cost_and_street_minimum_are_visible(self):
        app = self.app
        app.handle_event({"kind": "turn", "actor": 1, "min_bet": 4, "to_call": 25})
        app.handle_event({"kind": "decision", "legal": ("fold", "call")})
        self.assertEqual(app.bet_amount.get(), "4")
        self.assertEqual(app.buttons["call"].cget("text"), "Call 25")
        self.assertTrue(app.bet_entry.instate(["disabled"]))

    def test_result_overlay_leaves_controls_available_and_can_be_dismissed(self):
        app = self.app
        for winner, result, title in ((1, -12, "You win"), (0, 12, "Model wins"),
                                      (None, 0, "Split pot")):
            app.handle_event({"kind": "end", "winner": winner, "result": result, "showdown": False})
            self.assertEqual(app.overlay[0], title)
            self.assertFalse(app.deal_button.instate(["disabled"]))
            self.assertIsNone(self.root.grab_current())
            app.dismiss_overlay()
            self.assertIsNone(app.overlay)

    def test_animation_uses_event_loop_and_effects_can_be_cleared(self):
        app = self.app
        app.show_overlay("Flop", "Three community cards", duration=60)
        app.animate_chips(1, 25)
        callback_ran = []
        self.root.after(5, lambda: callback_ran.append(True))
        deadline = time.monotonic() + 1
        while app.overlay is not None and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.005)
        self.assertTrue(callback_ran)
        self.assertIsNone(app.overlay)
        app.clear_effects()
        self.assertIsNone(app.chip_job)
        self.assertIsNone(app.chip)
        app.animations.set(False)
        app.animate_chips(0, 10)
        self.assertIsNone(app.chip_job)

    def test_table_and_controls_fit_at_minimum_and_default_sizes(self):
        app = self.app
        self.root.deiconify()
        app.model_hole = ((14, 0), (13, 0))
        app.human_hole = ((12, 1), (11, 1))
        app.board = ((2, 0), (3, 1), (4, 2), (5, 3), (6, 0))
        app.revealed = True
        for size in ("1000x660", "1120x740"):
            with self.subTest(size=size):
                self.root.geometry(size)
                self.root.update()
                app.draw_table()
                tags = ("model_label", "model_cards", "pot", "board_cards",
                        "street_label", "human_label", "human_cards")
                boxes = [app.table.bbox(tag) for tag in tags]
                for first, second in zip(boxes, boxes[1:]):
                    self.assertLess(first[3], second[1])
                self.assertLess(boxes[-1][3], app.table.winfo_height())
                self.assertGreaterEqual(app.log.winfo_height(), 70)
                self.assertTrue(app.deal_button.winfo_ismapped())


if __name__ == "__main__":
    unittest.main()
