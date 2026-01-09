import numpy as np
from src.analytics.forecasting import SeismicForecaster
from src.analytics.tgr_math import SeismicityRate
from config.settings import KELLY_MULTIPLIER, MAX_POSITION_SIZE

class RiskManager:
    def __init__(self, historical_catalog_df, bankroll):
        # Initial Bankroll (used as fallback or denominator for Kelly)
        self.initial_bankroll = bankroll 
        self.forecaster = SeismicForecaster()
        
        # Calibrate Baseline Rate (Alpha)
        self.alpha_baseline = SeismicityRate.calculate_alpha0(
            historical_catalog_df, 
            mag_threshold=5.0, 
            time_window_years=44.0
        )

    def calculate_position_size(self, market_price, target_mag, days_remaining, 
                              current_exposure=0.0, available_capital=None):
        """
        Calculates position size respecting Max Position limits and Available Capital.
        """
        # 0. Safety Checks
        if days_remaining <= 0 or market_price >= 0.99:
            return 0.0, 0.0

        # Use actual available capital if provided, else static bankroll
        current_bankroll = available_capital if available_capital is not None else self.initial_bankroll

        # 1. Get Model Probability (Fair Value)
        fair_prob = self.forecaster.probability_at_least_one(
            alpha_threshold=self.alpha_baseline,
            mag_threshold=5.0,
            target_mag=target_mag,
            forecast_duration_years=days_remaining / 365.25
        )
        
        # 2. Check for Edge
        if fair_prob <= market_price:
            return 0.0, fair_prob

        # 3. Kelly Criterion
        b = (1.0 / market_price) - 1.0
        p = fair_prob
        q = 1.0 - p
        kelly_fraction = (b * p - q) / b
        
        # 4. Sizing
        # Apply fractional Kelly (safety factor)
        target_size = current_bankroll * kelly_fraction * KELLY_MULTIPLIER
        
        # 5. Constraints & Risk Limits
        
        # A. Cap at Max Position Size (Global Config)
        # If we already have $30 exposed and Max is $50, we can only bet $20 more.
        remaining_room = MAX_POSITION_SIZE - current_exposure
        if remaining_room <= 0:
            return 0.0, fair_prob # Position Full
            
        final_size = min(target_size, remaining_room)
        
        # B. Cap at Available Cash (Can't spend what we don't have)
        final_size = min(final_size, current_bankroll * 0.95) # Keep 5% buffer
        
        return max(0.0, final_size), fair_prob
    
    def calculate_exit_logic(self, market_price, fair_value, position_size_shares):
        """
        Determines if we should sell existing 'Yes' shares to recycle capital.
        Returns: (Action, Size_to_Sell)
           Action: 'HOLD', 'SELL_PROFIT', 'SELL_STOP'
        """
        if position_size_shares <= 0:
            return 'HOLD', 0.0

        # 1. Take Profit (Arb Exit)
        # If the market price is significantly higher than our model's probability,
        # the market is over-reacting. We sell to capture the spread.
        # Example: Market is paying 50c, but we think chance is only 40%.
        spread = market_price - fair_value
        if spread > 0.05: # 5 cent profit margin
            return 'SELL_PROFIT', position_size_shares

        # 2. Stop Loss / Time Decay
        # If we held a position because we thought a quake was coming, but time passed
        # and the probability dropped (fair_value < entry), we might want to exit.
        # However, purely probabilistic models usually hold to expiry. 
        # We only sell if the probability has collapsed near zero to salvage dust.
        if fair_value < 0.01 and market_price > 0.02:
            return 'SELL_STOP', position_size_shares

        return 'HOLD', 0.0