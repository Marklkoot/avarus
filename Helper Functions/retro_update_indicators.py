import mysql.connector
import pandas as pd
import datetime
import time
from decimal import Decimal

from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator

def retro_update_for_coin_tf(db, coin, timeframe, start_ts, end_ts):
    """
    Fetch all rows in [start_ts, end_ts] for this coin/timeframe,
    compute Bollinger, MACD, etc., then update the DB so none are NULL.
    """
    cursor = db.cursor(dictionary=True)
    
    # 1) Fetch older bars
    sql_fetch = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlc_data
        WHERE coin=%s
          AND timeframe=%s
          AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
    """
    cursor.execute(sql_fetch, (coin, timeframe, start_ts, end_ts))
    rows = cursor.fetchall()
    if not rows:
        print(f"[RetroUpdate] coin={coin}, tf={timeframe} => No rows in {start_ts} to {end_ts}")
        return
    
    df = pd.DataFrame(rows)
    df.rename(columns={"timestamp":"time"}, inplace=True)
    df = df.sort_values("time").reset_index(drop=True)

    # 2) Compute the indicators
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["boll_up"] = bb.bollinger_hband()
    df["boll_down"] = bb.bollinger_lband()
    df["boll_mavg"] = bb.bollinger_mavg()

    macd_ = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_.macd()
    df["macd_signal"] = macd_.macd_signal()
    df["macd_diff"] = macd_.macd_diff()

    ema_short = EMAIndicator(close=df["close"], window=10)
    df["ema_10"] = ema_short.ema_indicator()

    ema_long = EMAIndicator(close=df["close"], window=50)
    df["ema_50"] = ema_long.ema_indicator()

    # 3) Update
    upd_sql = """
        UPDATE ohlc_data
        SET boll_up=%s,
            boll_down=%s,
            boll_mavg=%s,
            macd=%s,
            macd_signal=%s,
            macd_diff=%s,
            ema_10=%s,
            ema_50=%s
        WHERE coin=%s
          AND timeframe=%s
          AND timestamp=%s
    """
    cursor2 = db.cursor()
    updates = 0
    for i, row_ in df.iterrows():
        t_ = row_["time"]
        bu = row_["boll_up"]
        bd = row_["boll_down"]
        mavg = row_["boll_mavg"] 
        mc = row_["macd"]
        ms = row_["macd_signal"]
        md = row_["macd_diff"]
        e10= row_["ema_10"]
        e50= row_["ema_50"]
        cursor2.execute(upd_sql, (
            float(bu)  if pd.notna(bu)  else None,
            float(bd)  if pd.notna(bd)  else None,
            float(mavg) if pd.notna(mavg) else None,
            float(mc)  if pd.notna(mc)  else None,
            float(ms)  if pd.notna(ms)  else None,
            float(md)  if pd.notna(md)  else None,
            float(e10) if pd.notna(e10) else None,
            float(e50) if pd.notna(e50) else None,
            coin, timeframe, t_
        ))
        updates += 1
    db.commit()
    print(f"[RetroUpdate] coin={coin}, tf={timeframe}, rows={len(df)}, updated={updates}")

def universal_retro_update(db):
    """
    1) Gather all distinct (coin, timeframe) from ohlc_data
    2) For each, find min(timestamp), max(timestamp)
    3) Retro-update that entire timespan
    """
    c = db.cursor(dictionary=True)
    c.execute("""
      SELECT coin, timeframe,
             MIN(timestamp) AS min_ts,
             MAX(timestamp) AS max_ts
      FROM ohlc_data
      GROUP BY coin, timeframe
    """)
    combos = c.fetchall()
    if not combos:
        print("No coin/timeframe combos in ohlc_data!")
        return
    
    for row in combos:
        coin = row["coin"]
        tf   = row["timeframe"]
        start_ts = row["min_ts"]
        end_ts   = row["max_ts"]
        print(f"== Retro for coin={coin}, tf={tf}, from {start_ts} to {end_ts}")
        retro_update_for_coin_tf(db, coin, tf, start_ts, end_ts)

if __name__=="__main__":
    db = mysql.connector.connect(
        host="localhost",
        user="avarus_user",
        password="someStrongPassword",
        database="avarus2"
    )

    universal_retro_update(db)
    db.close()
