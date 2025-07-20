import logging
import datetime
from decimal import Decimal
import pandas as pd
from ml_engine.ai_signals_margin import AIPricePredictorMargin


class VirtualPortfolio:
    def __init__(self, stable_symbol="EUR", initial_investment=800.0):
        self.stable_symbol = stable_symbol
        self.balances = {stable_symbol: Decimal(str(initial_investment))}
        self.trades = []
        self.initial_val = Decimal(str(initial_investment))
        self.highest_equity = Decimal(str(initial_investment))
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = Decimal("0")

        # track average cost basis
        self.coin_cost_basis = {}

    def get_balance(self, coin):
        return self.balances.get(coin, Decimal("0"))

    def record_trade(self, coin, side, amount, price, fee, ts):
        cost = Decimal(str(amount)) * Decimal(str(price))
        fee_ = Decimal(str(fee))

        self.trades.append({
            "coin": coin,
            "side": side,
            "amount": float(amount),
            "price": float(price),
            "fee": float(fee),
            "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S')
        })

        old_coin_qty = self.get_balance(coin)
        old_val = self.total_portfolio_value_for_trade()

        if side.lower() == "buy":
            self._adjust_balance(self.stable_symbol, -(cost + fee_))
            self._adjust_balance(coin, Decimal(str(amount)))
        else:
            self._adjust_balance(coin, -Decimal(str(amount)))
            net_gain = cost - fee_
            self._adjust_balance(self.stable_symbol, net_gain)

        new_val = self.total_portfolio_value_for_trade()

        # cost basis update
        new_coin_qty = self.get_balance(coin)
        if side.lower() == "buy":
            if old_coin_qty <= 0:
                self.coin_cost_basis[coin] = Decimal(str(price))
            else:
                old_basis = self.coin_cost_basis.get(coin, Decimal("0"))
                old_value = old_coin_qty * old_basis
                new_value = Decimal(str(amount)) * Decimal(str(price))
                combined_qty = old_coin_qty + Decimal(str(amount))
                self.coin_cost_basis[coin] = (old_value + new_value) / combined_qty
        elif side.lower() == "sell":
            if new_coin_qty <= 0:
                self.coin_cost_basis[coin] = Decimal("0")

        if side.lower() == "sell":
            if new_val > old_val:
                self.win_count += 1
            else:
                self.loss_count += 1

    def total_portfolio_value(self, prices):
        total_val = Decimal("0")
        for c, qty in self.balances.items():
            if c == self.stable_symbol:
                total_val += qty
            else:
                price = prices.get(c, Decimal("0"))
                if qty > 0 and price > 0:
                    total_val += (qty * price)
        return total_val

    def total_portfolio_value_for_trade(self):
        stable_val = self.balances.get(self.stable_symbol, Decimal("0"))
        return stable_val

    def _adjust_balance(self, coin, delta):
        old_q = self.get_balance(coin)
        new_q = old_q + delta
        self.balances[coin] = new_q

class BacktesterMargin:
    """
    A copy of your original backtester, but now named "BacktesterMargin" 
    for the margin meltdown scenario. 
    We keep meltdown logic the same. 
    This code references the param_map from meltdown or partial-sell logic 
    specifically for margin approach.
    """
    def __init__(self, db, config, stable_symbol="EUR", initial_investment=500.0):
        self.db = db
        self.config = config
        self.stable_symbol = stable_symbol
        self.initial_investment = initial_investment
        self.logger = logging.getLogger(__name__)
        self.meltdown_stage = 0
        self.meltdown_low = None
        self.ai_predictor_margin = AIPricePredictorMargin(db, config)

    def run_backtest(self, param_map, coin_list, start_date, end_date, timeframe="1h"):
        self.logger.info(f"[BacktestMargin] Start run_backtest => param_map={param_map}")
        from decimal import Decimal
        from pandas import to_datetime
        
        self.portfolio = VirtualPortfolio(
            stable_symbol=self.stable_symbol,
            initial_investment=self.initial_investment
        )

        self.meltdown_stage = 0
        self.meltdown_low = None

        historical_data = {}

        # 1) Fetch DataFrame per coin
        for coin in coin_list:
            df = self._fetch_ohlc(coin, timeframe, start_date, end_date)
            if df is None or df.empty:
                continue
            # Ensure 'time' is a datetime
            df["time"] = to_datetime(df["time"], utc=True, errors="coerce")
            # Sort ascending
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)
            historical_data[coin] = df

        # 2) Build a combined timeline of all unique 'time' points
        #    Now each 'time' is a Timestamp, not a string.
        all_ts = set()
        for coin in coin_list:
            if coin not in historical_data:
                continue
            # historical_data[coin] is a DF => we can just union them
            ts_values = historical_data[coin]["time"].dropna().unique()
            all_ts.update(ts_values)  # each is a pd.Timestamp

        # Convert to sorted list
        self.logger.debug(f"[BacktestMargin] loaded data => coin_list={coin_list}")
        timeline = sorted(list(all_ts))

        meltdown_anchor = None

        # Rebalancing
        raw_allocs = param_map.get("rebalance_target_allocations", "{}")
        try:
            rebalance_allocs = eval(raw_allocs) if isinstance(raw_allocs, str) else raw_allocs
        except:
            rebalance_allocs = {}
        rebalance_interval_days = 7
        last_rebalance_ts = None

        for ts in timeline:
            # ts is now a pd.Timestamp (datetime), not a str.

            # 3) Build prices_at_ts from your per-coin DataFrame
            prices_at_ts = {}
            for coin in coin_list:
                if coin not in historical_data:
                    prices_at_ts[coin] = Decimal("0")
                    continue
                row = self._find_latest_ohlc(historical_data[coin], ts)
                if row is not None:
                    # row is presumably a Series or something => row[4] might still be "close"
                    # If row is a Series with named columns, do row["close"] or row.iloc[4]
                    # Adjust as needed. We'll do row.iloc[4] for consistency with old code.
                    prices_at_ts[coin] = Decimal(str(row.iloc[4]))
                else:
                    prices_at_ts[coin] = Decimal("0")

            meltdown_anchor = self._update_meltdown_anchor(param_map, meltdown_anchor, prices_at_ts)
            meltdown_triggered = self._check_meltdown(param_map, meltdown_anchor, prices_at_ts, ts)

            if not meltdown_triggered:
                # 4) If meltdown not triggered, do AI logic
                for coin in coin_list:
                    if coin not in historical_data:
                        continue
                    recent_candles = self._get_recent(historical_data[coin], ts, lookback=50)
                    if len(recent_candles) < 2:
                        continue

                    # AI
                    ai_candles = self._get_recent(historical_data[coin], ts, lookback=20)
                    if len(ai_candles) >= 20:
                        ai_signal = self._ai_generate_signal(coin, ai_candles)
                        if ai_signal == "buy":
                            stable_bal = self.portfolio.get_balance(self.stable_symbol)
                            if stable_bal > 5:
                                spend = stable_bal * self.ai_buy_fraction
                                if spend < 5:
                                    spend = Decimal("5")
                                if spend > stable_bal:
                                    spend = stable_bal
                                last_close = Decimal(str(ai_candles.iloc[-1, 4]))
                                if last_close > 0 and spend > 0:
                                    buy_amt = spend / last_close
                                    fee = Decimal("0.3")
                                    # ts is datetime => pass it or convert it to str if needed
                                    self.portfolio.record_trade(coin, "buy", buy_amt, last_close, fee, ts)
                        elif ai_signal == "sell":
                            coin_bal = self.portfolio.get_balance(coin)
                            if coin_bal > 0:
                                sell_amt = coin_bal * self.ai_sell_fraction
                                last_close = Decimal(str(ai_candles.iloc[-1, 4]))
                                if sell_amt > 0 and last_close > 0:
                                    fee = Decimal("0.3")
                                    self.portfolio.record_trade(coin, "sell", sell_amt, last_close, fee, ts)

            # 5) Rebalancing check => do date arithmetic on Timestamps
            if last_rebalance_ts is None:
                last_rebalance_ts = ts
            else:
                # (ts - last_rebalance_ts) => Timedelta => .days is fine
                days_diff = (ts - last_rebalance_ts).days
                if days_diff >= rebalance_interval_days:
                    self._rebalance_portfolio(param_map, rebalance_allocs, prices_at_ts, ts)
                    last_rebalance_ts = ts

            # 6) Drawdown tracking
            current_val = self._calc_portfolio_value(prices_at_ts)
            if current_val > self.portfolio.highest_equity:
                self.portfolio.highest_equity = current_val
            dd_ratio = (self.portfolio.highest_equity - current_val) / (
                self.portfolio.highest_equity if self.portfolio.highest_equity > 0 else Decimal("1")
            )
            if dd_ratio > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = dd_ratio

        # 7) final check
        final_prices = {}
        if timeline:
            last_ts = timeline[-1]
            for coin in coin_list:
                if coin not in historical_data:
                    final_prices[coin] = Decimal("0")
                    continue
                row = self._find_latest_ohlc(historical_data[coin], last_ts)
                if row is not None:
                    final_prices[coin] = Decimal(str(row.iloc[4]))
                else:
                    final_prices[coin] = Decimal("0")

        final_val = self.portfolio.total_portfolio_value(final_prices)
        final_coins = {}
        total_coins_float = 0.0
        for c, bal in self.portfolio.balances.items():
            if c != self.stable_symbol and bal > 0:
                final_coins[c] = float(bal)
                total_coins_float += float(bal)

        return {
            "final_value": float(final_val),
            "win_count": self.portfolio.win_count,
            "loss_count": self.portfolio.loss_count,
            "max_dd": float(self.portfolio.max_drawdown),
            "accumulated_coins": final_coins,
            "total_coins": total_coins_float
        }

    def _ai_generate_signal(self, coin, df_candles):
        """
        df_candles is a DataFrame with open/high/low/close/volume etc.
        We'll do: for idx, row in df_candles.iterrows()
        """
        if df_candles is None or df_candles.empty:
            return "hold"

        # e.g. build a data list from each row
        data = []
        for idx, row in df_candles.iterrows():
            data.append({
                "time":   row["time"],
                "close":  float(row["close"]),
                "volume": float(row["volume"])
                # ...
            })

        # Then pass 'df_candles' or 'data' to your AI predictor or do logic:
        ai_result = self.ai_predictor_margin.generate_signal(coin, df_candles, seq_len=20, threshold_pct=0.01)
        return ai_result


    def _fetch_ohlc(self, coin, timeframe, start_date, end_date):
        cursor = self.db.cursor()
        q = """
            SELECT 
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                boll_up,
                boll_down,
                macd,
                macd_signal,
                macd_diff,
                ema_10,
                ema_50
            FROM ohlc_data
            WHERE coin=%s
              AND timeframe=%s
              AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
        """
        cursor.execute(q, (coin, timeframe, start_date, end_date))
        rows = cursor.fetchall()

        out = []
        for r in rows:
            # r is a tuple => (timestamp, open, high, low, close, volume, boll_up, boll_down, macd, ...)
            out.append(r)

        # Now build a DataFrame with explicit columns
        df = pd.DataFrame(out, columns=[
            "timestamp","open","high","low","close","volume",
            "boll_up","boll_down","macd","macd_signal","macd_diff",
            "ema_10","ema_50"
        ])

        # Rename 'timestamp' -> 'time'
        df.rename(columns={"timestamp":"time"}, inplace=True)

        # 1) Convert 'time' column to datetime type:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

        # 2) Optionally sort ascending by 'time':
        df.sort_values("time", inplace=True, ascending=True)

        return df


    def _find_latest_ohlc(self, df, ts):
        best = None
        for idx, row in df.iterrows():
            if row["time"] <= ts:
                best = row
            else:
                break
        return best

    def _get_recent(self, df, ts, lookback=50):
        """
        Return a sub-DataFrame of the rows where df["time"] <= ts,
        then take the last 'lookback' rows.
        Assumes df is already sorted ascending by time.
        """
        if df is None or df.empty:
            return pd.DataFrame()  # empty fallback

        # filter => all rows with time <= ts
        subset = df[df["time"] <= ts]
        if subset.empty:
            return subset  # or pd.DataFrame()

        # pick last 'lookback' rows
        return subset.iloc[-lookback:].copy()


    def _find_latest_ohlc(self, df, ts):
        """
        Return a single row (a Series) for the 'latest' candle
        whose df["time"] <= ts.
        If none found, return None.
        """
        if df is None or df.empty:
            return None

        subset = df[df["time"] <= ts]
        if subset.empty:
            return None
        # the last row => 'latest' time up to 'ts'
        return subset.iloc[-1]


    def _update_meltdown_anchor(self, pm, meltdown_anchor, prices_at_ts):
        current_val = self._calc_portfolio_value(prices_at_ts)
        meltdown_anchor_buffer = Decimal(str(pm.get("meltdown_anchor_buffer", "0.02")))
        if meltdown_anchor is None:
            meltdown_anchor = current_val
        else:
            threshold_new_anchor = meltdown_anchor * (Decimal("1") + meltdown_anchor_buffer)
            if current_val > threshold_new_anchor:
                meltdown_anchor = current_val
        return meltdown_anchor


    def _check_meltdown(self, pm, meltdown_anchor, prices_at_ts, ts):
        self.logger.debug(f"[CheckMeltdown] meltdown_stage={self.meltdown_stage}, ts={ts}")
        meltdown_tiers = pm.get("meltdown_tiers", [])
        meltdown_reentry_pct = Decimal(str(pm.get("meltdown_reentry_pct", "0.10")))
        meltdown_threshold = Decimal(str(pm.get("meltdown_threshold", 0.30)))

        current_val = self._calc_portfolio_value(prices_at_ts)
        if meltdown_anchor is None or meltdown_anchor == Decimal("0"):
            return False

        if self.meltdown_low is None:
            self.meltdown_low = current_val

        dd = (meltdown_anchor - current_val) / meltdown_anchor if meltdown_anchor>0 else Decimal("0")
        meltdown_triggered = False

        if self.meltdown_stage < len(meltdown_tiers):
            next_threshold = Decimal(str(meltdown_tiers[self.meltdown_stage]["drawdown"]))
            next_sell_ratio = Decimal(str(meltdown_tiers[self.meltdown_stage]["sell_ratio"]))

            if dd >= next_threshold:
                self.logger.warning(f"[Backtest Meltdown] dd={dd:.2%} => meltdown_stage={self.meltdown_stage}, ratio={next_sell_ratio}")
                self._partial_liquidate_portfolio(next_sell_ratio, prices_at_ts, ts, pm)
                self.meltdown_stage += 1
                meltdown_triggered = True
                self.meltdown_low = min(self.meltdown_low, current_val)
        else:
            if dd > meltdown_threshold:
                self.logger.warning(f"[Backtest Meltdown] dd={dd:.2%} => final meltdown => catastrophic => sell everything")
                self._liquidate_portfolio(prices_at_ts, ts, pm, catastrophic=True)
                meltdown_triggered = True
                self.meltdown_stage = len(meltdown_tiers)

        if self.meltdown_stage > 0:
            rebound_level = self.meltdown_low * (Decimal("1") + meltdown_reentry_pct)
            if current_val > rebound_level:
                old_stage = self.meltdown_stage
                self.meltdown_stage -= 1
                ratio = Decimal("0.0")
                if (old_stage-1) < len(meltdown_tiers):
                    ratio = Decimal(str(meltdown_tiers[old_stage-1]["sell_ratio"]))
                if ratio > Decimal("0"):
                    self.logger.warning(f"[Backtest ReEntry] meltdown_stage -> {self.meltdown_stage}, ratio={ratio}")
                    self._partial_rebuy_portfolio(ratio, prices_at_ts, ts)
                self.meltdown_low = current_val

        return meltdown_triggered


    def _partial_liquidate_portfolio(self, ratio, prices_at_ts, ts, pm):
        core_ratio = Decimal(str(pm.get("core_ratio", 0.70)))

        for coin, qty in list(self.portfolio.balances.items()):
            if coin == self.stable_symbol:
                continue
            full_amt = self.portfolio.get_balance(coin)
            if full_amt > 0:
                core_qty = full_amt * core_ratio
                trading_qty = full_amt - core_qty
                if trading_qty <= 0:
                    continue
                sell_qty = trading_qty * ratio
                if sell_qty <= 0:
                    continue
                price = prices_at_ts.get(coin, Decimal("0"))
                if price > 0 and sell_qty > 0:
                    fee = price * sell_qty * Decimal("0.0035")  # e.g. 0.35% fee
                    self.portfolio.record_trade(coin, "sell", sell_qty, price, fee, ts)
        self.logger.warning("[PartialMeltdown] Done partial liquidation in backtest => only trading portion sold.")


    def _partial_rebuy_portfolio(self, ratio, prices_at_ts, ts):
        stable_bal = self.portfolio.get_balance(self.stable_symbol)
        coins = [c for c in self.portfolio.balances.keys() if c != self.stable_symbol]
        if not coins or stable_bal <= 0:
            return
        per_coin_spend = (stable_bal * ratio) / Decimal(str(len(coins)))
        for c in coins:
            if per_coin_spend < 5:
                continue
            price = prices_at_ts.get(c, Decimal("0"))
            if price > 0:
                buy_qty = per_coin_spend / price
                fee = buy_qty * price * Decimal("0.0035")
                self.portfolio.record_trade(c, "buy", buy_qty, price, fee, ts)
        self.logger.warning("[PartialRebuy] Done partial re-buy in backtest.")


    def _liquidate_portfolio(self, prices_at_ts, ts, pm, catastrophic=False):
        core_ratio = Decimal(str(pm.get("core_ratio", 0.70)))
        for coin, qty in list(self.portfolio.balances.items()):
            if coin == self.stable_symbol:
                continue
            full_amt = self.portfolio.get_balance(coin)
            if full_amt <= 0:
                continue

            if catastrophic:
                sell_qty = full_amt
            else:
                core_qty = full_amt * core_ratio
                sell_qty = full_amt - core_qty

            if sell_qty <= 0:
                continue
            price = prices_at_ts.get(coin, Decimal("0"))
            if price > 0:
                fee = sell_qty * price * Decimal("0.0035")
                self.portfolio.record_trade(coin, "sell", sell_qty, price, fee, ts)

        if catastrophic:
            self.logger.warning("[Backtest Meltdown] => Catastrophic meltdown => sold everything (core included).")
        else:
            self.logger.warning("[Backtest Meltdown] => partial meltdown => sold only trading portion.")


    def _calc_portfolio_value(self, prices_at_ts):
        return self.portfolio.total_portfolio_value(prices_at_ts)


    def _rebalance_portfolio(self, pm, target_allocs, prices_at_ts, ts):
        if self.meltdown_stage > 0:
            self.logger.info("[Backtest Rebalance] meltdown_stage>0 => skip rebalancing.")
            return
        current_val = self._calc_portfolio_value(prices_at_ts)
        if current_val <= 0:
            return

        stable_bal = self.portfolio.get_balance(self.stable_symbol)
        core_ratio = Decimal(str(pm.get("core_ratio", 0.70)))
        threshold = Decimal("0.20")

        all_coins = set(list(self.portfolio.balances.keys()) + list(target_allocs.keys()))
        if self.stable_symbol in all_coins:
            all_coins.remove(self.stable_symbol)

        for c in all_coins:
            target_w = Decimal(str(target_allocs.get(c, 0.0)))
            price = prices_at_ts.get(c, Decimal("0"))
            if price <= 0:
                continue
            hold_qty = self.portfolio.get_balance(c)
            hold_val = hold_qty * price

            if target_w == 0:
                if hold_qty > 0:
                    fee = hold_qty * price * Decimal("0.0035")
                    self.portfolio.record_trade(c, "sell", hold_qty, price, fee, ts)
                continue

            target_abs_val = current_val * target_w
            if target_abs_val <= 0:
                continue

            diff_ratio = (hold_val - target_abs_val) / (target_abs_val if target_abs_val>0 else Decimal("1"))

            if diff_ratio > threshold:
                core_amt = hold_qty * core_ratio
                free_amt = hold_qty - core_amt
                if free_amt <= 0:
                    continue
                excess_val = hold_val - target_abs_val
                sell_val = min(excess_val, free_amt * price)
                if sell_val > 0:
                    sell_qty = sell_val / price
                    fee = sell_qty * price * Decimal("0.0035")
                    self.portfolio.record_trade(c, "sell", sell_qty, price, fee, ts)

            elif diff_ratio < -threshold:
                short_val = target_abs_val - hold_val
                if short_val > 0:
                    if short_val > stable_bal:
                        short_val = stable_bal
                    if short_val > 0:
                        buy_qty = short_val / price
                        fee = buy_qty * price * Decimal("0.0035")
                        self.portfolio.record_trade(c, "buy", buy_qty, price, fee, ts)
                        stable_bal -= short_val
