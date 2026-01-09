import unittest
import numpy as np
from src.analytics.tgr_math import TGRPhysics
from src.analytics.forecasting import SeismicForecaster

class TestTGRMath(unittest.TestCase):
    
    def test_magnitude_moment_conversion(self):
        # M=6.0 should equal approx 1.0e18 Nm
        mag = 6.0
        moment = TGRPhysics.magnitude_to_moment(mag)
        # 1.5 * 6.0 + 9.0 = 18.0 -> 10^18
        self.assertAlmostEqual(moment, 1.0e18)
        
        # Inverse check
        calc_mag = TGRPhysics.moment_to_magnitude(moment)
        self.assertAlmostEqual(calc_mag, 6.0)

    def test_probability_decay(self):
        """
        Test that probability of M >= 8.0 is lower than M >= 7.0
        """
        forecaster = SeismicForecaster()
        alpha = 100.0 # 100 quakes per year > M5
        
        prob_7 = forecaster.probability_at_least_one(alpha, 5.0, 7.0, 1.0)
        prob_8 = forecaster.probability_at_least_one(alpha, 5.0, 8.0, 1.0)
        
        print(f"Prob > 7.0: {prob_7:.5f}")
        print(f"Prob > 8.0: {prob_8:.5f}")
        
        self.assertTrue(prob_8 < prob_7)

if __name__ == '__main__':
    unittest.main()