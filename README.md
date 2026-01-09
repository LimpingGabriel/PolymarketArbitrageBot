# Seismic Alpha: Low-Latency Arbitrage System

**A hybrid Market-Making and High-Frequency Trading (HFT) engine that arbitrages the latency delta between physical seismographic sensors and the UMA Oracle on Polymarket.**

## 1. Executive Summary

**Seismic Alpha** exploits a structural inefficiency in earthquake prediction markets: **Information Asymmetry in Time.**

* **The Oracle Lag:** Prediction markets resolve based on USGS/EMSC reports, which have a publication latency of **15–60 seconds** after an event.
* **The Physical Signal:** Raw P-waves (Primary waves) travel at ~6 km/s. By listening directly to the **SeedLink** TCP protocol from seismographic stations (e.g., `CI.PAS`), we detect events **1.5s to 15s** before the market price reacts.
* **The Strategy:**
    1.  **Passive (Market Making):** Uses a **Tapered Gutenberg-Richter (TGR)** model to calculate the fair value of "No Earthquake" contracts based on historical Poisson arrival rates.
    2.  **Active (Sniping):** Listens to live telemetry. When a P-wave signal-to-noise ratio (STA/LTA) exceeds 4.0σ, the bot executes an IOC (Immediate-or-Cancel) Buy order on "Yes".

---

## 2. System Architecture

The system follows a **Lambda Architecture**:
* **Speed Layer (Ingestion):** Connects to IRIS Ring Servers via TCP (SeedLink).
* **Batch Layer (Analytics):** Calibrates statistical models on 40+ years of NDK/CMT data.
* **Execution Layer (Trade):** Interfaces with the Polymarket CLOB (Central Limit Order Book).

### File Structure
```text
seismic_algo/
├── config/
│   └── settings.py          # Credentials & Risk Parameters
├── src/
│   ├── analytics/
│   │   ├── tgr_math.py      # Tapered Gutenberg-Richter Physics
│   │   └── forecasting.py   # Probability Models
│   ├── data/
│   │   └── loader.py        # ObsPy NDK Parser
│   ├── market/
│   │   └── classifier.py    # Market Discovery & Regex
│   ├── execution/
│   │   └── gateway.py       # Polymarket CLOB Client
│   ├── strategy.py          # Kelly Criterion & Logic
│   └── market_scanner.py    # Opportunity Finder
└── main_engine.py           # The Event Loop
```

---

## 3. Implementation Details

### A. Configuration (`config/settings.py`)
```python
import os
from dotenv import load_dotenv

load_dotenv()

# --- CREDENTIALS ---
CLOB_API_KEY = os.getenv("CLOB_API_KEY")
CLOB_SECRET = os.getenv("CLOB_SECRET")
CLOB_PASS_PHRASE = os.getenv("CLOB_PASS_PHRASE")
PRIVATE_KEY = os.getenv("PK")  # Polygon Wallet Private Key

# --- RISK MANAGEMENT ---
STARTING_CAPITAL = 500.0  # USDC
KELLY_MULTIPLIER = 0.25   # Safety Factor (Quarter-Kelly)
MAX_POSITION_SIZE = 50.0  # Max USDC per trade hard cap

# --- SIGNAL TRIGGERS ---
STA_LTA_THRESHOLD = 4.0   # Signal-to-Noise Ratio to trigger SNIPE
COOLDOWN_SECONDS = 300    # Prevent double-firing on same event
```

### B. The Physics Layer (`src/analytics/tgr_math.py`)
```python
import numpy as np

# Constants from Kagan & Jackson (Standard Seismology)
C_MOMENT = 9.0
CORNER_MAG_DEFAULT = 9.0

class TGRPhysics:
    """Implements Tapered Gutenberg-Richter Physics."""

    @staticmethod
    def magnitude_to_moment(mag):
        """Converts Mw to Scalar Moment (Nm)."""
        return 10.0 ** (1.5 * np.array(mag) + C_MOMENT)

    @staticmethod
    def survivor_function(moment, moment_threshold, beta=0.678, corner_moment=None):
        """
        The Fraction of earthquakes > moment.
        G(M) = (Mt / M)^beta * exp((Mt - M) / Mcm)
        """
        if corner_moment is None:
            corner_moment = TGRPhysics.magnitude_to_moment(CORNER_MAG_DEFAULT)
        
        term1 = (moment_threshold / moment) ** beta
        term2 = np.exp((moment_threshold - moment) / corner_moment)
        return term1 * term2

class SeismicityRate:
    @staticmethod
    def calculate_alpha0(df_catalog, mag_threshold, time_window_years):
        """Calculates observed annual rate (Alpha)."""
        count = len(df_catalog[df_catalog['magnitude'] >= mag_threshold])
        return count / time_window_years if time_window_years > 0 else 0.0
```

### C. The Forecasting Layer (`src/analytics/forecasting.py`)
```python
import numpy as np
from src.analytics.tgr_math import TGRPhysics

class SeismicForecaster:
    def __init__(self, beta=0.678):
        self.beta = beta
        self.corner_moment = TGRPhysics.magnitude_to_moment(9.0)

    def probability_at_least_one(self, alpha_threshold, mag_threshold, target_mag, forecast_duration_years):
        """
        Calculates P(N >= 1) using Poisson process derived from TGR rate.
        """
        mt = TGRPhysics.magnitude_to_moment(mag_threshold)
        mx = TGRPhysics.magnitude_to_moment(target_mag)

        # 1. Extrapolate Rate
        survivor_prob = TGRPhysics.survivor_function(mx, mt, self.beta, self.corner_moment)
        lambda_target = alpha_threshold * survivor_prob

        # 2. Poisson Probability
        expected_events = lambda_target * forecast_duration_years
        return 1.0 - np.exp(-expected_events)
```

### D. The Market Discovery Layer (`src/market/classifier.py`)
```python
import re
import requests
from dataclasses import dataclass

@dataclass
class SeismicMarket:
    id: str
    question: str
    region: str
    min_magnitude: float
    expiry_date: str
    price_yes: float

class MarketClassifier:
    MAG_PATTERN = r"(?:magnitude|mag|M)\s?(\d+(?:\.\d+)?)"
    REGIONS = ['California', 'Japan', 'Global']

    def fetch_active_markets(self):
        """Scrapes Polymarket Gamma API for Earthquake markets."""
        url = "[https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&tag_id=101](https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&tag_id=101)"
        try:
            data = requests.get(url).json()
        except: return []

        markets = []
        for item in data:
            q = item.get('question', '')
            if 'earthquake' not in q.lower(): continue
            
            # Extract Magnitude
            mag_match = re.search(self.MAG_PATTERN, q, re.IGNORECASE)
            if not mag_match: continue
            
            # Extract Region
            region = next((r for r in self.REGIONS if r in q), 'Global')
            
            markets.append(SeismicMarket(
                id=item.get('conditionId'),
                question=q,
                region=region,
                min_magnitude=float(mag_match.group(1)),
                expiry_date=item.get('endDate'),
                price_yes=0.0 # Requires CLOB lookup in prod
            ))
        return markets
```

### E. The Execution Gateway (`src/execution/gateway.py`)
```python
import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, ApiCreds
from py_clob_client.constants import AMOY, POLYGON
from py_clob_client.order_builder.constants import BUY

class PolymarketGateway:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.creds = ApiCreds(
            os.getenv("CLOB_API_KEY"), 
            os.getenv("CLOB_SECRET"), 
            os.getenv("CLOB_PASS_PHRASE")
        )
        chain = AMOY if dry_run else POLYGON
        if not dry_run:
            self.client = ClobClient("[https://clob.polymarket.com](https://clob.polymarket.com)", key=os.getenv("PK"), chain_id=chain, creds=self.creds)

    def snipe_yes(self, condition_id, size):
        if self.dry_run:
            print(f"[DRY RUN] SNIPE BUY | ID: {condition_id} | Size: {size}")
            return
        
        try:
            order = self.client.create_and_post_order(
                OrderArgs(price=0.99, size=int(size), side=BUY, token_id=condition_id)
            )
            print(f"✅ EXECUTION: {order}")
        except Exception as e:
            print(f"❌ EXECUTION FAILED: {e}")
```

### F. Strategy Logic (`src/strategy.py`)
```python
from config.settings import KELLY_MULTIPLIER, MAX_POSITION_SIZE
from src.analytics.forecasting import SeismicForecaster
from src.analytics.tgr_math import SeismicityRate

class RiskManager:
    def __init__(self, historical_catalog_df, bankroll):
        self.bankroll = bankroll
        self.forecaster = SeismicForecaster()
        # Calibrate baseline on 44 years of history (1976-2020)
        self.alpha_baseline = SeismicityRate.calculate_alpha0(historical_catalog_df, 5.0, 44.0)

    def calculate_position_size(self, market_price, target_mag, days_remaining):
        # 1. Fair Value
        fair_prob = self.forecaster.probability_at_least_one(
            self.alpha_baseline, 5.0, target_mag, days_remaining/365.25
        )
        
        # 2. Edge Check
        if fair_prob <= market_price: return 0.0

        # 3. Kelly Criterion
        b = (1.0 / market_price) - 1.0
        f = (b * fair_prob - (1-fair_prob)) / b
        
        # 4. Sizing
        size = self.bankroll * f * KELLY_MULTIPLIER
        return min(size, MAX_POSITION_SIZE)
```

### G. Main Event Loop (`main_engine.py`)
```python
import threading
import time
from obspy.clients.seedlink.easyseedlink import create_client
from obspy.signal.trigger import classic_sta_lta
from src.data.loader import SeismicDataLoader
from src.strategy import RiskManager
from src.market.classifier import MarketClassifier
from src.execution.gateway import PolymarketGateway

# --- STATE ---
active_target_id = None # Best market to snipe
risk_engine = None
gateway = None

def on_data(trace):
    """HOT PATH: SeedLink Callback"""
    # Ring Buffer Logic
    data = trace.data
    sr = trace.stats.sampling_rate
    cft = classic_sta_lta(data, int(1*sr), int(10*sr))
    
    if cft[-1] > 4.0 and active_target_id:
        print(f"🔴 EARTHQUAKE DETECTED! Ratio: {cft[-1]}")
        gateway.snipe_yes(active_target_id, size=100) # Dynamic sizing in prod

def scanner_loop():
    """Finds the best market to target"""
    global active_target_id
    classifier = MarketClassifier()
    while True:
        markets = classifier.fetch_active_markets()
        if markets:
            # Simple logic: Target the first active market found
            active_target_id = markets[0].id 
            print(f"🎯 Targeted Market: {markets[0].question}")
        time.sleep(3600)

if __name__ == "__main__":
    print("⚡ SYSTEM START ⚡")
    
    # Init
    # Ensure 'data/historical/jan76_dec20.ndk' exists
    df = SeismicDataLoader.load_ndk("data/historical/jan76_dec20.ndk")
    risk_engine = RiskManager(df, bankroll=500)
    gateway = PolymarketGateway(dry_run=True)
    
    # Threads
    t = threading.Thread(target=scanner_loop, daemon=True)
    t.start()
    
    # Listener
    client = create_client('rtserve.iris.washington.edu', on_data)
    client.select_stream('CI', 'PAS', 'HHZ')
    client.run()
```