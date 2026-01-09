import json
import re
from dataclasses import dataclass
from typing import Optional, List
import requests

@dataclass
class SeismicMarket:
    id: str
    question: str
    region: str          # 'California', 'Japan', 'Global'
    min_magnitude: float # 5.0, 7.0
    expiry_date: str     # YYYY-MM-DD
    price_yes: float
    liquidity: float

class MarketClassifier:
    """
    Parses messy Polymarket titles into structured seismic parameters.
    """
    
    # Regex patterns to extract data from titles
    # Capture groups: 1=Magnitude
    MAG_PATTERN = r"(?:magnitude|mag|M)\s?(\d+(?:\.\d+)?)"
    
    # Region keywords mapped to NDK coordinates (approximate bounding boxes)
    REGIONS = {
        'California': {'lat_min': 32, 'lat_max': 42, 'lon_min': -125, 'lon_max': -114},
        'Japan': {'lat_min': 30, 'lat_max': 46, 'lon_min': 128, 'lon_max': 146},
        'Global': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
        'Worldwide': {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180},
    }

    def fetch_active_markets(self) -> List[SeismicMarket]:
        """
        Queries Polymarket Gamma API for all 'Earthquake' tagged markets.
        """
        print("🌐 Querying Polymarket Gamma API...")
        url = "https://gamma-api.polymarket.com/events?limit=50&active=true&closed=false&tag_id=101" # 101 is roughly Science/Climate, simpler to search text
        # Better approach: Search by query text
        url = "https://gamma-api.polymarket.com/markets?limit=100&active=true&closed=false&order=volume&ascending=false"
        
        try:
            # We fetch a broad list then filter client-side for "Earthquake"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ API Error: {e}")
            return []

        clean_markets = []
        
        for item in data:
            question = item.get('question', '')
            
            # 1. Filter: Must be about Earthquakes
            if 'earthquake' not in question.lower():
                continue

            # 2. Parse Magnitude
            mag_match = re.search(self.MAG_PATTERN, question, re.IGNORECASE)
            if not mag_match:
                continue # Skip if we can't determine magnitude
            magnitude = float(mag_match.group(1))

            # 3. Parse Region
            region = 'Global' # Default
            for r_name in self.REGIONS.keys():
                if r_name.lower() in question.lower():
                    region = r_name
                    break

            # 4. Extract Price (Token ID 1 is usually 'YES' for binary markets)
            # Gamma API structure varies, this is a simplified access pattern
            try:
                # Depending on the API response structure (checking Outcome or CLOB)
                outcome_prices = json.loads(item.get('outcomePrices', '["0", "0"]'))
                price_yes = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.0
            except:
                price_yes = 0.0

            m = SeismicMarket(
                id=item.get('conditionId'), # Crucial for CLOB execution
                question=question,
                region=region,
                min_magnitude=magnitude,
                expiry_date=item.get('endDate'), # ISO Format
                price_yes=price_yes,
                liquidity=float(item.get('liquidity', 0))
            )
            clean_markets.append(m)
            
        print(f"✅ Found {len(clean_markets)} active earthquake markets.")
        return clean_markets

    def get_region_bounds(self, region_name):
        return self.REGIONS.get(region_name, self.REGIONS['Global'])