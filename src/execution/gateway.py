import os
import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, AssetType, OrderArgs  # <--- Added OrderArgs
from py_clob_client.constants import AMOY, POLYGON
from py_clob_client.order_builder.constants import BUY, SELL
from eth_account import Account  # Requires: pip install eth-account

class PolymarketGateway:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        
        # Load Creds
        self.host = "https://clob.polymarket.com"
        self.key = os.getenv("PK") 
        self.creds = ApiCreds(
            api_key=os.getenv("CLOB_API_KEY"),
            api_secret=os.getenv("CLOB_SECRET"),
            api_passphrase=os.getenv("CLOB_PASS_PHRASE"),
        )
        self.chain_id = AMOY if dry_run else POLYGON
        
        if not self.dry_run:
            self.client = ClobClient(self.host, key=self.key, chain_id=self.chain_id, creds=self.creds)
            # Derive public address from Private Key for Data API queries
            self.address = Account.from_key(self.key).address
        else:
            print("🛡️ DRY RUN MODE: No orders will be sent.")
            self.address = "0x0000000000000000000000000000000000000000"

    def get_usdc_balance(self):
        """
        Fetches available USDC collateral.
        """
        if self.dry_run:
            return 500.0 # Mock balance
            
        try:
            # Fetch collateral balance (USDC)
            # Note: returns { 'balance': '...', 'allowance': '...' }
            resp = self.client.get_balance_allowance(
                params={'asset_type': AssetType.COLLATERAL}
            )
            return float(resp.get('balance', 0.0))
        except Exception as e:
            print(f"⚠️ Failed to fetch balance: {e}")
            return 0.0

    def get_portfolio_positions(self):
        """
        Returns: {market_id: {'shares': float, 'exposure': float}}
        """
        if self.dry_run: return {}
        url = "https://data-api.polymarket.com/positions"
        try:
            resp = requests.get(url, params={"user": self.address, "limit": 100})
            resp.raise_for_status()
            positions = {}
            for pos in resp.json():
                size = float(pos.get('size', 0))
                if size > 1.0:
                    positions[pos.get('market')] = {
                        'shares': size,
                        'exposure': size * float(pos.get('avgPrice', 0))
                    }
            return positions
        except Exception:
            return {}
    
    def place_limit_order(self, condition_id, side, price, size):
        # ... (Existing code from previous turn) ...
        if self.dry_run:
            print(f"[DRY RUN] LIMIT {side} | ID: {condition_id[:8]}... | Price: {price} | Size: {size}")
            return {"status": "simulated", "orderID": "mock_123"}

        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY if side == 'BUY' else SELL,
                token_id=condition_id,
            )
            resp = self.client.create_and_post_order(order_args)
            return resp
        except Exception as e:
            print(f"❌ ORDER FAILED: {e}")
            return None