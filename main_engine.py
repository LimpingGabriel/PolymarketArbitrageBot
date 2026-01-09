import threading
import time
import os
from dotenv import load_dotenv

from obspy.clients.seedlink.easyseedlink import create_client
from obspy.signal.trigger import classic_sta_lta

from src.data.loader import SeismicDataLoader
from src.strategy import RiskManager
from src.market_scanner import MarketScanner
from src.execution.gateway import PolymarketGateway
from config.settings import STA_LTA_THRESHOLD, NDK_PATH, CAPITAL

# Global State
active_opportunities = {}
gateway = None
risk_engine = None

def on_seismic_event(trace):
    # ... [Same as before] ...
    pass 

def passive_scanner_loop(scanner):
    """Batch Layer: Portfolio Management, Exits, and Entries"""
    global active_opportunities
    
    while True:
        print("\n🔄 SYNCING PORTFOLIO...")
        
        # 1. Fetch State
        cash = gateway.get_usdc_balance()
        # positions is dict: {market_id: current_exposure_value_usdc}
        # We need share count, but for now we approximate or fetch detail if needed.
        # Ideally gateway returns {market_id: {'shares': 100, 'avg_price': 0.10}}
        positions = gateway.get_portfolio_positions() 
        
        print(f"   💰 Balance: ${cash:.2f} | 📜 Active Positions: {len(positions)}")

        # 2. Fetch Active Markets
        markets = scanner.classifier.fetch_active_markets()
        market_map = {m.id: m for m in markets}

        # --- EXIT LOGIC (New) ---
        print("📉 Checking for Exit Opportunities...")
        for m_id, pos_data in positions.items():
            if m_id not in market_map: continue # Market closed or not found
            
            m = market_map[m_id]
            
            # Calculate current Fair Value
            days_left = (m.expiry_date_dt - datetime.now()).days
            fair_prob = risk_engine.get_fair_value(m.min_magnitude, days_left)
            
            # Decide
            action, amount = risk_engine.calculate_exit_logic(
                market_price=m.price_yes, 
                fair_value=fair_prob,
                position_size_shares=pos_data['shares'] 
            )
            
            if action != 'HOLD':
                print(f"   👋 EXECUTING {action}: {m.question}")
                print(f"      Price: {m.price_yes:.2f} | Fair: {fair_prob:.2f}")
                gateway.place_limit_order(m_id, 'SELL', m.price_yes - 0.01, amount)

        # --- ENTRY LOGIC ---
        print("🔍 Scanning for Entries...")
        for m in markets:
            existing_exposure = positions.get(m.id, {}).get('exposure', 0.0)
            
            bet_size, fair_val = risk_engine.calculate_position_size(
                market_price=m.price_yes,
                target_mag=m.min_magnitude,
                days_remaining=(m.expiry_date_dt - datetime.now()).days,
                current_exposure=existing_exposure,
                available_capital=cash
            )

            if bet_size > 5.0:
                print(f"✅ VALUE FOUND: {m.question}")
                gateway.place_limit_order(m.id, 'BUY', m.price_yes + 0.01, bet_size)
                active_opportunities[m.id] = m.min_magnitude

        time.sleep(3600)


def main():
    global gateway, risk_engine
    load_dotenv()
    
    # Init
    df = SeismicDataLoader.load_ndk(NDK_PATH)
    gateway = PolymarketGateway(dry_run=os.getenv("DRY_RUN") == "True")
    risk_engine = RiskManager(df, bankroll=CAPITAL)
    scanner = MarketScanner(risk_engine)
    
    # Threads
    t = threading.Thread(target=passive_scanner_loop, args=(scanner,), daemon=True)
    t.start()
    
    # SeedLink
    client = create_client('rtserve.iris.washington.edu', on_seismic_event)
    client.select_stream('CI', 'PAS', 'HHZ')
    client.run()

if __name__ == "__main__":
    main()