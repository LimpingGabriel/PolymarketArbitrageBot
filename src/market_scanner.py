# src/market_scanner.py
import pandas as pd
from datetime import datetime
from src.market.classifier import MarketClassifier

class MarketScanner:
    def __init__(self, risk_manager):
        self.rm = risk_manager
        self.classifier = MarketClassifier()

    def find_best_opportunities(self):
        """
        Scans active earthquake markets and ranks them by Kelly Edge.
        """
        # 1. Fetch REAL active markets
        markets = self.classifier.fetch_active_markets()
        
        opportunities = []

        for m in markets:
            # 2. Skip low liquidity markets (optional safety check)
            if m.liquidity < 100: 
                continue

            # 3. Calculate days remaining
            try:
                # ISO Format: 2024-12-31T23:59:59Z or 2024-12-31
                # Truncate to YYYY-MM-DD for simplicity
                expiry_str = m.expiry_date.split('T')[0]
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
                days_left = (expiry - datetime.now()).days
            except ValueError:
                continue
            
            if days_left <= 0: continue

            # 4. Strategy Calculation
            # We access attributes with dot notation (m.price_yes) because it's a Dataclass
            bet_size, fair_val = self.rm.calculate_position_size(
                market_price=m.price_yes,
                target_mag=m.min_magnitude,
                days_remaining=days_left
            )
            
            # 5. Filter for positive EV
            if bet_size > 0:
                print(f"💰 OPPORTUNITY: {m.question} | Edge: {fair_val - m.price_yes:.2f}")
                opportunities.append({
                    'market_id': m.id,
                    'question': m.question,
                    'bet_size': bet_size,
                    'market_price': m.price_yes,
                    'fair_value': fair_val,
                    'roi': (fair_val - m.price_yes) / m.price_yes
                })

        # Sort by best ROI
        df = pd.DataFrame(opportunities)
        if not df.empty:
            return df.sort_values(by='roi', ascending=False)
        return df