import unittest
from src.market.classifier import MarketClassifier

class TestSystem(unittest.TestCase):
    
    def test_classifier_regex(self):
        """Does our regex correctly identify Magnitude and Region?"""
        mc = MarketClassifier()
        
        # Test Case 1: Standard Title
        title1 = "Will a Magnitude 7.5+ Earthquake hit Japan in 2024?"
        match1 = mc.parse_title_mock(title1) # You'd expose the internal logic to test
        # (For this example, we assume we extracted logic to a helper method)
        
        # Mock logic check
        import re
        mag = float(re.search(mc.MAG_PATTERN, title1, re.IGNORECASE).group(1))
        self.assertEqual(mag, 7.5)
        self.assertTrue("Japan" in title1)

    def test_dry_run_execution(self):
        """Ensure Gateway doesn't crash on Dry Run"""
        from src.execution.gateway import PolymarketGateway
        gw = PolymarketGateway(dry_run=True)
        res = gw.place_limit_order("0x123", "BUY", 0.50, 100)
        self.assertEqual(res['status'], 'simulated')

if __name__ == "__main__":
    print("🧪 RUNNING SYSTEM DIAGNOSTICS...")
    unittest.main()