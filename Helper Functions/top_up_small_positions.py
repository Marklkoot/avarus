#!/usr/bin/env python3

"""
top_up_small_positions.py

A stand-alone script that:
1) Loads your config + DB connection (same as Avarus).
2) Asks how much new stable (USD) you're investing.
3) Asks which coins to top up.
4) Splits that new stable equally among those coins, places buy orders on Kraken,
   and records the trades in DB so meltdown anchor can protect them.
5) (Optionally) updates meltdown_anchor right away to include this new capital.
"""

import os
import sys
import logging
import yaml
import datetime
from decimal import Decimal

import ccxt

# Adjust import paths to match your folder structure
from db.connection import get_db_connection
from portfolio_manager import PortfolioManager

def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1) Load config
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2) Decide how much new stable to invest
    #    Hardcode or prompt user input. For example:
    total_new_stable = Decimal("2000")  # <-- change this as needed
    
    # 3) Which coins to top up (list them)
    coins_to_top_up = ["SUI", "BCH", "DOT", "ICP", "RENDER", "UNI", "ETH", "ONDO", "SOL", "ETC", "BTC", "TAO", "LINK", "AAVE", "XRP", "POL", "APT", "SHIB", "DOGE", "TON", "KAS", "ADA", "LTC", "TRX", "XLM"]
    
    # 4) Connect DB
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # 5) Initialize ccxt Kraken
    exch_cfg = config["exchange"]  # e.g. { name: "kraken", apiKey:..., secret:..., ... }
    exchange_class = getattr(ccxt, exch_cfg["name"])
    kraken = exchange_class({
        'apiKey': exch_cfg["apiKey"],
        'secret': exch_cfg["secret"],
        'enableRateLimit': exch_cfg.get("enableRateLimit", True)
    })
    kraken.load_markets()
    
    # 6) Create portfolio manager (so we can record trades in DB)
    portfolio_mgr = PortfolioManager(db, kraken, config)
    
    # 7) Check how many coins we top up => divide equally
    num_coins = len(coins_to_top_up)
    if num_coins == 0:
        logging.warning("[TopUp] No coins_to_top_up => nothing to do.")
        return
    
    per_coin_allocation = total_new_stable / Decimal(num_coins)
    logging.info(f"[TopUp] We have {total_new_stable} USD to invest among {num_coins} coins => {per_coin_allocation} each.")
    
    # 8) For each coin, place a buy & record
    for coin in coins_to_top_up:
        symbol = f"{coin}/USD"
        if symbol not in kraken.markets:
            logging.warning(f"[TopUp] {symbol} not in kraken => skip coin {coin}.")
            continue
        
        # fetch ask price
        try:
            ticker = kraken.fetch_ticker(symbol)
        except Exception as e:
            logging.warning(f"[TopUp] fetch_ticker fail => {symbol} => {e}")
            continue
        
        ask_price = Decimal(str(ticker.get("ask", 0)))
        if ask_price <= 0:
            logging.warning(f"[TopUp] {symbol} => invalid ask => skip buy.")
            continue
        
        # how many units to buy
        units = per_coin_allocation / ask_price
        if units < Decimal("0.00001"):
            logging.warning(f"[TopUp] {symbol} => computed units={units} => too small => skip.")
            continue
        
        logging.info(f"[TopUp] Buying ~{per_coin_allocation} USD => {units} {coin} at price={ask_price}")
        
        # place a limit buy at ask price
        try:
            order = kraken.create_limit_buy_order(str(symbol), float(units), float(ask_price))
            logging.info(f"[TopUp] order => {order}")
            
            # If we want meltdown anchor to protect these funds => record in DB
            # Attempt to parse fill price or fee from 'order' if available
            # For now, naive approach:
            fill_price = ask_price
            fee = Decimal("1.0")  # or parse from order['fee'] or order['fees']
            
            # record trade in DB
            portfolio_mgr.record_trade(
                coin,          # e.g. "DOGE"
                "buy",
                float(units),
                float(fill_price),
                float(fee),
                order_obj=order
            )
            
        except Exception as e:
            logging.warning(f"[TopUp] fail => {coin} => {e}")
    
    # 9) Optional => forcibly update meltdown anchor (if you want meltdown to reflect new capital *immediately*)
    #    Typically the meltdown logic will update anchor next time it sees a new net_value>anchor.
    #    But if you want to do it right now:
    _maybe_update_meltdown_anchor(db, portfolio_mgr, config)
    
    logging.info("[TopUp] Done distributing capital and recording trades. You may now restart Avarus.")

def _maybe_update_meltdown_anchor(db, portfolio_mgr, config):
    """
    If meltdown logic is stored in meta_parameters under param_name='meltdown_anchor',
    we can forcibly recalc the net_value and set meltdown_anchor to it if it's higher.
    """
    # 1) get meltdown_anchor
    q = """SELECT param_value FROM meta_parameters WHERE param_name='meltdown_anchor'"""
    cursor = db.cursor(dictionary=True)
    cursor.execute(q)
    row = cursor.fetchone()
    meltdown_anchor_str = row["param_value"] if row else None
    
    if meltdown_anchor_str is None:
        logging.info("[TopUp] meltdown_anchor not found => skipping forced anchor update.")
        return
    
    meltdown_anchor = Decimal(meltdown_anchor_str)
    
    # 2) calc current net value => sum stable + coin_value
    current_val = _calc_portfolio_value(db, portfolio_mgr, config)
    if current_val > meltdown_anchor:
        logging.info(f"[TopUp] meltdown_anchor => updating from {meltdown_anchor} to {current_val}")
        # store new meltdown_anchor
        now_ = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        q2 = """REPLACE INTO meta_parameters(param_name, param_value, last_updated)
                VALUES('meltdown_anchor', %s, %s)"""
        cursor.execute(q2, (str(current_val), now_))
        db.commit()
    else:
        logging.info(f"[TopUp] meltdown_anchor={meltdown_anchor}, current_val={current_val} => no update needed.")

def _calc_portfolio_value(db, portfolio_mgr, config):
    """
    Minimal approach => stable + (each coin's quantity * last price).
    If you want the same code as strategy uses, replicate it here or you could
    do a partial approach. We'll do a simple approach with ccxt tickers.
    """
    pf = portfolio_mgr.get_portfolio()
    stable = pf.get(portfolio_mgr.stable_symbol, 0)
    total_val = Decimal(str(stable))
    
    # connect ccxt
    exch_cfg = config["exchange"]
    exchange_class = getattr(ccxt, exch_cfg["name"])
    kraken = exchange_class({
        'apiKey': exch_cfg["apiKey"],
        'secret': exch_cfg["secret"],
        'enableRateLimit': exch_cfg.get("enableRateLimit", True)
    })
    kraken.load_markets()
    
    for c, amt in pf.items():
        if c == portfolio_mgr.stable_symbol or amt <= 0:
            continue
        pair = f"{c}/USD"
        if pair in kraken.markets:
            try:
                t = kraken.fetch_ticker(pair)
                last_price = Decimal(str(t.get("last", t.get("close", 0.0))))
                if last_price>0:
                    total_val += Decimal(str(amt)) * last_price
            except:
                pass
    return total_val


if __name__ == "__main__":
    main()
