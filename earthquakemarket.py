import os, json

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from dotenv import load_dotenv
from py_clob_client.constants import AMOY

import requests


def get_market_ids_by_slug(slug):
    request = requests.get(f"https://gamma-api.polymarket.com/events/slug/{slug}")
    if request.status_code == 200:
        market_data = json.loads(request.text)

        ids = []
        for market in market_data["markets"]:
            ids.append(market["conditionId"])
        
        return ids

class Market():
    def __init__(self, ):
        pass

    def update(self):
        pass

class EarthquakeOccurrenceMarket():
    pass

if __name__ == "__main__":
    host = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
    key = os.getenv("PK")
    creds = ApiCreds(
        api_key=os.getenv("CLOB_API_KEY"),
        api_secret=os.getenv("CLOB_SECRET"),
        api_passphrase=os.getenv("CLOB_PASS_PHRASE"),
    )
    chain_id = AMOY
    client = ClobClient(host, key=key, chain_id=chain_id, creds=creds)

    #print(client.get_markets())
    #print(client.get_simplified_markets())
    #print(client.get_sampling_markets())
    #print(client.get_sampling_simplified_markets())
    condition_ids = get_market_ids_by_slug("another-7pt0-or-above-earthquake-by-october-31-951")
    print(condition_ids)
    #print(client.get_market(condition_id=condition_id))