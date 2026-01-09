# src/market_scanner.py
import pandas as pd
from datetime import datetime
from src.execution.gateway import PolymarketGateway # You'd write this wrapper

class MarketScanner:
    def __init__(self, risk_manager):
        self.rm = risk_manager
        self.gateway = PolymarketGateway() # Connects to API

    def find_best_opportunities(self):
        """
        Scans active earthquake markets and ranks them by Kelly Edge.
        """
        # 1. Fetch active markets (Mocked for structure)
        # markets = self.gateway.fetch_markets(tag="Earthquake")
        
        # MOCK DATA for logic verification
        markets = [
            {'id': '1', 'region': 'California', 'mag': 6.0, 'expiry': '2026-02-01', 'price_yes': 0.15},
            {'id': '2', 'region': 'California', 'mag': 7.0, 'expiry': '2026-02-01', 'price_yes': 0.02},
            {'id': '3', 'region': 'Japan', 'mag': 7.0, 'expiry': '2026-03-01', 'price_yes': 0.05},
        ]
        
        opportunities = []

        for m in markets:
            # Calculate days remaining
            expiry = datetime.strptime(m['expiry'], "%Y-%m-%d")
            days_left = (expiry - datetime.now()).days
            
            if days_left <= 0: continue

            # ASK STRATEGY: What is this worth?
            bet_size, fair_val = self.rm.calculate_position_size(
                market_price=m['price_yes'],
                target_mag=m['mag'],
                days_remaining=days_left
            )
            
            if bet_size > 0:
                opportunities.append({
                    'market_id': m['id'],
                    'bet_size': bet_size,
                    'edge': fair_val - m['price_yes'],
                    'roi': (fair_val - m['price_yes']) / m['price_yes']
                })

        # Sort by best ROI
        df = pd.DataFrame(opportunities)
        if not df.empty:
            return df.sort_values(by='roi', ascending=False)
        return df