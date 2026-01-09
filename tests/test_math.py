import unittest
from src.strategy import RiskManager

class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(bankroll=1000.0)

    def test_kelly_positive_edge(self):
        # We think 60% chance win, Market pays 1:1 (50% implied)
        # Edge is positive. Should bet positive amount.
        bet = self.rm.calculate_kelly_bet(market_price=0.50, model_prob=0.60)
        print(f"Positive Edge Bet: ${bet}")
        self.assertTrue(bet > 0)

    def test_kelly_negative_edge(self):
        # We think 40% chance win, Market prices at 50%
        # Edge is negative. Should bet 0.
        bet = self.rm.calculate_kelly_bet(market_price=0.50, model_prob=0.40)
        self.assertEqual(bet, 0.0)

    def test_kelly_hard_cap(self):
        # Huge edge, but should limit to MAX_POSITION_SIZE (50.0)
        bet = self.rm.calculate_kelly_bet(market_price=0.10, model_prob=0.99)
        self.assertEqual(bet, 50.0)

if __name__ == '__main__':
    unittest.main()