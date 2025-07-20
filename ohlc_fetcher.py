# ohlc_fetcher.py

import ccxt
import logging
import datetime
import time
import mysql.connector
from decimal import Decimal
import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator

class OhlcFetcher:
    def __init__(self, db, exchange_cfg, coin_list):
        """
        exchange_cfg => dict from config, e.g. { "name":"kraken", "apiKey":"...", "secret":"..." }
        coin_list => e.g. ["BTC","ETH","XRP"]
        """
        exchange_class = getattr(ccxt, exchange_cfg["name"])
        self.exchange = exchange_class({
            'apiKey': exchange_cfg["apiKey"],
            'secret': exchange_cfg["secret"],
            'enableRateLimit': exchange_cfg.get("enableRateLimit", True)
        })
        self.db = db
        self.coin_list = coin_list
        self.logger = logging.getLogger(__name__)

    def fetch_and_store_ohlc(self, timeframe='1h', limit=100):
        cursor = self.db.cursor()
        for coin in self.coin_list:
            if coin.upper() in ["USDC", "USDT", "DAI", "EUR", "USD", "XMR"]:
                self.logger.info(f"[OhlcFetcher] Skipping stablecoin: {coin}")
                continue
            symbol = f"{coin}/USD"
            try:
                ohlc = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not ohlc:
                    continue

                # 1) Insert or Update the raw OHLC data as you already do:
                for candle in ohlc:
                    ts = candle[0]
                    dt_utc = datetime.datetime.utcfromtimestamp(ts/1000)
                    _open  = Decimal(str(candle[1]))
                    _high  = Decimal(str(candle[2]))
                    _low   = Decimal(str(candle[3]))
                    _close = Decimal(str(candle[4]))
                    _vol   = Decimal(str(candle[5]))

                    sql = """INSERT INTO ohlc_data 
                             (coin, timeframe, timestamp, open, high, low, close, volume)
                             VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                             ON DUPLICATE KEY UPDATE
                               open=%s, high=%s, low=%s, close=%s, volume=%s
                          """
                    cursor.execute(sql, (
                        coin, timeframe, dt_utc, _open, _high, _low, _close, _vol,
                        _open, _high, _low, _close, _vol
                    ))
                self.db.commit()

                # 2) Now fetch the latest data from DB into a Pandas DataFrame:
                #    (We want to compute Bollinger, MACD, etc. for all 'fresh' rows.)
                #    Typically we do it for the last 'limit' bars.
                q = """SELECT timestamp, open, high, low, close, volume
                       FROM ohlc_data
                       WHERE coin=%s AND timeframe=%s
                       ORDER BY timestamp DESC
                       LIMIT %s
                    """
                cursor.execute(q, (coin, timeframe, limit))
                rows = cursor.fetchall()
                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
                # Sort ascending so indicators compute properly
                df = df.sort_values("timestamp").reset_index(drop=True)

                # Compute indicators with the "ta" library:
                bb = BollingerBands(close=df["close"], window=20, window_dev=2)
                df["boll_up"] = bb.bollinger_hband()
                df["boll_down"] = bb.bollinger_lband()

                macd_ = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
                df["macd"] = macd_.macd()
                df["macd_signal"] = macd_.macd_signal()
                df["macd_diff"] = macd_.macd_diff()

                ema_short = EMAIndicator(close=df["close"], window=10)
                df["ema_10"] = ema_short.ema_indicator()

                ema_long = EMAIndicator(close=df["close"], window=50)
                df["ema_50"] = ema_long.ema_indicator()

                df.dropna(inplace=True)

                # 3) UPDATE each row in DB with these computed indicator values:
                for i, row_ in df.iterrows():
                    tstamp = row_["timestamp"]  # same type as DB (datetime)
                    upd_sql = """
                      UPDATE ohlc_data
                      SET boll_up=%s, boll_down=%s, macd=%s, macd_signal=%s, macd_diff=%s,
                          ema_10=%s, ema_50=%s
                      WHERE coin=%s AND timeframe=%s AND timestamp=%s
                    """
                    cursor.execute(upd_sql, (
                        float(row_["boll_up"]) if not pd.isna(row_["boll_up"]) else None,
                        float(row_["boll_down"]) if not pd.isna(row_["boll_down"]) else None,
                        float(row_["macd"]) if not pd.isna(row_["macd"]) else None,
                        float(row_["macd_signal"]) if not pd.isna(row_["macd_signal"]) else None,
                        float(row_["macd_diff"]) if not pd.isna(row_["macd_diff"]) else None,
                        float(row_["ema_10"]) if not pd.isna(row_["ema_10"]) else None,
                        float(row_["ema_50"]) if not pd.isna(row_["ema_50"]) else None,
                        coin, timeframe, row_["timestamp"]
                    ))
                self.db.commit()

                self.logger.info(f"[OhlcFetcher] Inserted {len(ohlc)} raw candles + updated indicators for {coin}:{timeframe}")

            except Exception as e:
                self.logger.warning(f"[OhlcFetcher] Error fetching {symbol} => {e}")