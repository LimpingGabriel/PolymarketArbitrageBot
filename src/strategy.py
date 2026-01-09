# src/strategy.py
import numpy as np
from src.analytics.forecasting import SeismicForecaster
from src.analytics.tgr_math import SeismicityRate
from config.settings import KELLY_MULTIPLIER, MAX_POSITION_SIZE, CAPITAL

class RiskManager:
    def __init__(self, historical_catalog_df, bankroll=CAPITAL):
        """
        Args:
            historical_catalog_df: Pandas DF of historical quakes (loaded via ObsPy)
            bankroll: Total USDC available.
        """
        self.bankroll = bankroll
        self.forecaster = SeismicForecaster()
        
        # 1. Calibrate the Baseline Rate (Alpha) immediately upon load
        # We calculate the rate of M >= 5.0 events over the last ~40 years
        # This acts as our "Prior"
        self.alpha_baseline = SeismicityRate.calculate_alpha0(
            historical_catalog_df, 
            mag_threshold=5.0, 
            time_window_years=44.0 # Approx duration of NDK catalog
        )
        print(f"✅ Strategy Calibrated: Baseline Rate (M>=5.0) = {self.alpha_baseline:.4f}/year")

    def get_fair_value(self, target_mag, days_remaining):
        """
        Returns the Fair Probability (0.0 to 1.0) of a quake >= target_mag 
        occurring within days_remaining.
        """
        if days_remaining <= 0:
            return 0.0

        prob = self.forecaster.probability_at_least_one(
            alpha_threshold=self.alpha_baseline,
            mag_threshold=5.0,        # Our calibrated baseline
            target_mag=target_mag,    # The market's target (e.g. 7.0)
            forecast_duration_years=days_remaining / 365.25
        )
        return prob

    def calculate_position_size(self, market_price, target_mag, days_remaining):
        """
        Decides how much to bet on 'YES' using Kelly Criterion.
        """
        # 1. Get Model Probability (Fair Value)
        fair_prob = self.get_fair_value(target_mag, days_remaining)
        
        # 2. Check for Edge
        if fair_prob <= market_price:
            return 0.0, fair_prob # No edge

        # 3. Kelly Criterion
        # b = Net Odds = (1 / market_price) - 1
        b = (1.0 / market_price) - 1.0
        p = fair_prob
        q = 1.0 - p
        
        kelly_fraction = (b * p - q) / b
        
        # 4. Safety Multipliers
        safe_fraction = kelly_fraction * KELLY_MULTIPLIER
        bet_amount = self.bankroll * safe_fraction
        
        # 5. Hard Cap
        final_size = min(bet_amount, MAX_POSITION_SIZE)
        return max(0.0, final_size), fair_prob