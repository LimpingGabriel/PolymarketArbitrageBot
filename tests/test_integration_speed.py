import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from obspy.core import Trace, Stats, UTCDateTime
from main_engine import on_seismic_event
import config.settings

class TestSpeedLayer(unittest.TestCase):
    
    @patch('main_engine.gateway')
    @patch('main_engine.active_opportunities')
    def test_snipe_trigger(self, mock_ops, mock_gateway):
        """
        Simulate a sudden spike in seismic amplitude and verify the bot fires.
        """
        # 1. Setup Active Markets (Pre-condition: We are watching a market)
        # Mocking dictionary: {market_id: magnitude_threshold}
        mock_ops.items.return_value = [("0xTargetMarket123", 5.0)]
        mock_ops.__bool__.return_value = True # ensure if active_opportunities: is True
        
        # 2. Create Synthetic Seismic Data (The "Injection")
        # Generate 15 seconds of noise followed by a massive SPIKE
        sampling_rate = 40.0
        n_samples = int(15 * sampling_rate)
        data = np.random.normal(0, 0.1, n_samples) # Background noise
        
        # Inject "Earthquake" at the end (Last 1 second is huge)
        data[-40:] = np.random.normal(10, 2, 40) 
        
        stats = Stats()
        stats.sampling_rate = sampling_rate
        stats.station = "TEST_STATION"
        trace = Trace(data=data, header=stats)

        # 3. Inject into Engine
        print("\n🧪 INJECTING SYNTHETIC P-WAVE...")
        on_seismic_event(trace)

        # 4. Verification
        # Did we try to place an order?
        mock_gateway.place_limit_order.assert_called_with(
            '0xTargetMarket123', 'BUY', 0.99, 100
        )
        print("✅ SUCCESS: Snipe order fired on synthetic signal.")

if __name__ == '__main__':
    unittest.main()