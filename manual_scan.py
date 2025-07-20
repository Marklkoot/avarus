# manual_scan.py
import logging
import yaml
import ccxt
import os
import sys

# If your fundamental engine is in fundamental_engine.py:
from fundamental_engine import FundamentalEngine

# If your DB connection code is in db/connection.py or something similar:
from db.connection import get_db_connection

def main():
    # 1) Load config
    config_path = "config/config.yaml"  # adjust if your config is elsewhere
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"[manual_scan] Could not load config => {e}")
        sys.exit(1)

    # 2) Create DB connection
    db = get_db_connection()
    if not db:
        logging.error("[manual_scan] DB connection failed => exit.")
        sys.exit(1)

    # 3) Create ccxt exchange instance (Kraken or whichever is specified in config)
    exch_name = config["exchange"]["name"]
    exchange_class = getattr(ccxt, exch_name)
    exchange_obj = exchange_class({
        "apiKey": config["exchange"]["apiKey"],
        "secret": config["exchange"]["secret"],
        "enableRateLimit": True
    })

    # 4) Instantiate FundamentalEngine
    fund_engine = FundamentalEngine(db=db, exchange=exchange_obj, config=config)

    # 5) Force a scan_new_coins
    logging.info("[manual_scan] => start scanning new coins...")
    fund_engine.scan_new_coins()
    logging.info("[manual_scan] => done scanning coins.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
