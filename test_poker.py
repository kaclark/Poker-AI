import unittest

from poker import BET_SIZE, MAX_BET, best_hand, observation, play_hand, rank_five


class PokerTests(unittest.TestCase):
    def test_hand_ranking_and_wheel(self):
        wheel = ((14, 0), (2, 0), (3, 0), (4, 0), (5, 0))
        higher_straight = ((2, 1), (3, 2), (4, 0), (5, 3), (6, 0))
        self.assertEqual(rank_five(wheel), (8, 5))
        self.assertEqual(rank_five(higher_straight), (4, 6))
        self.assertGreater(rank_five(wheel), rank_five(higher_straight))
        seven = wheel + ((13, 1), (13, 2))
        self.assertEqual(best_hand(seven), (8, 5))

    def test_reproducible_play(self):
        def call(features, legal):
            self.assertEqual(len(features), 12)
            return "call"

        for seed in range(20):
            self.assertEqual(play_hand(call, seed), play_hand(call, seed))

    def test_observation_does_not_need_opponent_cards(self):
        features = observation(((14, 0), (14, 1)), (), 0, 0, 2, True, True)
        self.assertEqual(features[2], 1.0)
        self.assertEqual(features[4], 0.0)

    def test_trace_result_matches_hand_result(self):
        events = []
        result = play_hand(lambda features, legal: "call", 8, on_event=events.append)
        self.assertEqual(events[0]["kind"], "start")
        self.assertEqual(events[-1]["kind"], "end")
        self.assertEqual(events[-1]["result"], result)
        self.assertEqual(sum(event["kind"] == "street" for event in events), 4)

    def test_human_opponent_uses_same_game_rules(self):
        events = []
        turns = []

        def human(features, legal):
            turns.append((features, legal))
            return "call"

        result = play_hand(lambda features, legal: "call", 13, "human",
                           on_event=events.append, opponent_policy=human)
        self.assertTrue(turns)
        self.assertEqual(len(events[0]["opponent_hole"]), 2)
        self.assertEqual(events[-1]["result"], result)
        self.assertEqual(events[-1]["kind"], "end")

    def test_sized_bets_are_matched_and_keep_accounting(self):
        events = []

        def human(features, legal):
            return ("raise", 25) if "raise" in legal else "call"

        result = play_hand(lambda features, legal: "call", 13, "human",
                           on_event=events.append, opponent_policy=human)
        actions = [event for event in events if event["kind"] == "action"]
        self.assertEqual([event["amount"] for event in actions if event["action"] == "bet"], [25] * 4)
        self.assertEqual([event["amount"] for event in actions if event["action"] == "call"], [25] * 4)
        self.assertEqual(actions[-1]["pot"], 202)
        self.assertIn(result, (-101, 0, 101))
        self.assertEqual(events[-1]["result"], result)

    def test_fold_to_sized_bet_only_loses_ante(self):
        events = []
        result = play_hand(lambda features, legal: "fold" if "fold" in legal else "call",
                           13, "human", on_event=events.append,
                           opponent_policy=lambda features, legal: ("raise", MAX_BET))
        self.assertEqual(result, -1)
        self.assertFalse(events[-1]["showdown"])
        self.assertEqual(events[-1]["winner"], 1)

    def test_invalid_sized_bets_are_rejected(self):
        for choice in (("raise", 0), ("raise", -1), ("raise", MAX_BET + 1),
                       ("raise", 2.5), ("raise", True), ("call", 10), ("raise",)):
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                play_hand(lambda features, legal: choice, 0)
        with self.assertRaises(ValueError):
            play_hand(lambda features, legal: "call", 13, "human",
                      opponent_policy=lambda features, legal: ("raise", 2))

    def test_default_bets_remain_fixed_and_turns_report_call_cost(self):
        events = []
        play_hand(lambda features, legal: "raise" if "raise" in legal else "call", 0,
                  on_event=events.append, opponent_policy=lambda features, legal: "call")
        self.assertEqual([event["amount"] for event in events if event["kind"] == "action"
                          and event["action"] == "bet"], list(BET_SIZE))
        calls = [event for event in events if event["kind"] == "turn" and event["actor"] == 1]
        self.assertEqual([event["to_call"] for event in calls], list(BET_SIZE))
        self.assertTrue(all("raise" not in event["legal"] for event in calls))


if __name__ == "__main__":
    unittest.main()
