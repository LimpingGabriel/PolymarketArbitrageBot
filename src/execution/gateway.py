import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, ApiCreds
from py_clob_client.constants import AMOY, POLYGON
from py_clob_client.order_builder.constants import BUY, SELL

class PolymarketGateway:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        
        # Load Creds
        self.host = "https://clob.polymarket.com"
        self.key = os.getenv("PK") # Your Polygon Private Key
        self.creds = ApiCreds(
            api_key=os.getenv("CLOB_API_KEY"),
            api_secret=os.getenv("CLOB_SECRET"),
            api_passphrase=os.getenv("CLOB_PASS_PHRASE"),
        )
        
        if not self.dry_run:
            print("⚠️ LIVE TRADING ENABLED - CONNECTING TO POLYGON MAINNET")
            # Use POLYGON for real money, AMOY for testnet
            self.client = ClobClient(self.host, key=self.key, chain_id=POLYGON, creds=self.creds)
            print("✅ CLOB Client Connected.")
        else:
            print("🛡️ DRY RUN MODE: No orders will be sent.")

    def place_limit_order(self, condition_id, side, price, size):
        """
        Places a Maker order (Limit).
        side: 'BUY' or 'SELL'
        """
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