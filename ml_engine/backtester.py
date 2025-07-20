import logging
import datetime
from decimal import Decimal
import pandas as pd

class VirtualPortfolio:
    def __init__(self, stable_symbol="USD", initial_investment=5500.0):
        self.stable_symbol = stable_symbol
        self.balances = {stable_symbol: Decimal(str(initial_investment))}
        self.trades = []
        self.initial_val = Decimal(str(initial_investment))
        self.highest_equity = Decimal(str(initial_investment))
        self.win_count = 0
        self.loss_count = 0
        self.max_drawdown = Decimal("0")

        # --- ADDED: store average cost basis per coin ---
        self.coin_cost_basis = {}  # e.g. { "BTC": Decimal("30000.0"), ... }

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

        # Store old balance *before* adjusting
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

        # --- ADDED: update cost basis after the trade ---
        new_coin_qty = self.get_balance(coin)
        if side.lower() == "buy":
            # if we had zero (or negative) coins, set cost basis to current price
            if old_coin_qty <= 0:
                self.coin_cost_basis[coin] = Decimal(str(price))
            else:
                # Weighted average cost
                old_basis = self.coin_cost_basis.get(coin, Decimal("0"))
                old_value = old_coin_qty * old_basis
                new_value = Decimal(str(amount)) * Decimal(str(price))
                combined_qty = old_coin_qty + Decimal(str(amount))
                self.coin_cost_basis[coin] = (old_value + new_value) / combined_qty

        elif side.lower() == "sell":
            # if we fully exited the position, reset cost basis to 0
            if new_coin_qty <= 0:
                self.coin_cost_basis[coin] = Decimal("0")
        # --- END ADDED ---

        # Track wins/losses
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
        # Returns just the stable coin part (useful for comparing net after a trade).
        stable_val = self.balances.get(self.stable_symbol, Decimal("0"))
        return stable_val

    def _adjust_balance(self, coin, delta):
        old_q = self.get_balance(coin)
        new_q = old_q + delta
        self.balances[coin] = new_q
        
# === AI ADD ===
from ml_engine.ai_signals import AIPricePredictor
# === END AI ADD ===

class Backtester:
    def __init__(self, db, config, stable_symbol="USD", initial_investment=5500.0):
        self.db = db
        self.config = config
        self.stable_symbol = stable_symbol
        self.initial_investment = initial_investment
        self.logger = logging.getLogger(__name__)
        self.meltdown_stage = 0
        self.meltdown_low = None
        
        # === AI ADD ===
        self.ai_predictor = AIPricePredictor(db, config)
        # === END AI ADD ===

    def run_backtest(self, param_map, coin_list, start_date, end_date, timeframe="1h"):
        self.portfolio = VirtualPortfolio(
            stable_symbol=self.stable_symbol,
            initial_investment=self.initial_investment
        )

        # meltdown reset
        self.meltdown_stage = 0
        self.meltdown_low = None
        
        # === AI ADD ===
        self.ai_buy_fraction = Decimal(str(param_map.get("ai_buy_fraction","0.02")))
        self.ai_sell_fraction= Decimal(str(param_map.get("ai_sell_fraction","0.25")))
        # === END AI ADD ===
       
        historical_data = {}
        for coin in coin_list:
            rows = self._fetch_ohlc(coin, timeframe, start_date, end_date)
            rows.sort(key=lambda x: x[0])
            historical_data[coin] = rows

        # create a sorted list of all timestamps
        all_ts = set()
        for coin in coin_list:
            for row in historical_data[coin]:
                all_ts.add(row[0])
        timeline = sorted(list(all_ts))

        meltdown_anchor = None

        # rebalancing
        raw_allocs = param_map.get("rebalance_target_allocations", "{}")
        try:
            rebalance_allocs = eval(raw_allocs) if isinstance(raw_allocs, str) else raw_allocs
        except:
            rebalance_allocs = {}

        rebalance_interval_days = 7
        last_rebalance_ts = None

        for ts in timeline:
            prices_at_ts = {}
            for coin in coin_list:
                row = self._find_latest_ohlc(historical_data[coin], ts)
                if row:
                    prices_at_ts[coin] = Decimal(str(row[4]))
                else:
                    prices_at_ts[coin] = Decimal("0")

            meltdown_anchor = self._update_meltdown_anchor(param_map, meltdown_anchor, prices_at_ts)
            meltdown_triggered = self._check_meltdown(param_map, meltdown_anchor, prices_at_ts, ts)
            if not meltdown_triggered:
                # if meltdown not triggered, we do normal dip buy, partial sell, momentum, AI
                for coin in coin_list:
                    recent_candles = self._get_recent(historical_data[coin], ts, lookback=50)
                    if len(recent_candles) < 2:
                        continue
                    self._check_dip_buy(param_map, coin, recent_candles, ts)
                    self._check_partial_sell(param_map, coin, recent_candles, ts)
                    self._check_momentum_buy(param_map, coin, recent_candles, ts)

                    # AI
                    ai_candles = self._get_recent(historical_data[coin], ts, lookback=20)
                    if len(ai_candles)>=20:
                        ai_signal = self._ai_generate_signal(coin, ai_candles)
                        if ai_signal=="buy":
                            stable_bal = self.portfolio.get_balance(self.stable_symbol)
                            if stable_bal>5:
                                spend = stable_bal*self.ai_buy_fraction
                                if spend<5:
                                    spend= Decimal("5")
                                if spend> stable_bal:
                                    spend= stable_bal
                                last_close= Decimal(str(ai_candles[-1][4]))
                                if last_close>0 and spend>0:
                                    buy_amt= spend/ last_close
                                    # CHANGED: realistic fee for buy trade
                                    cost_of_trade = buy_amt * last_close
                                    fee_percent   = Decimal("0.0035")  # 0.35%
                                    fee           = cost_of_trade * fee_percent
                                    self.portfolio.record_trade(coin, "buy", buy_amt, last_close, fee, ts)
                        elif ai_signal=="sell":
                            coin_bal = self.portfolio.get_balance(coin)
                            if coin_bal>0:
                                sell_amt= coin_bal*self.ai_sell_fraction
                                last_close= Decimal(str(ai_candles[-1][4]))
                                if sell_amt>0 and last_close>0:
                                    cost_of_trade = sell_amt * last_close
                                    fee_percent   = Decimal("0.0035")  # 0.35%
                                    fee           = cost_of_trade * fee_percent
                                    self.portfolio.record_trade(coin,"sell",sell_amt,last_close,fee,ts)

            # check if we do weekly rebalance
            if last_rebalance_ts is None:
                last_rebalance_ts = ts
            else:
                days_diff = (ts - last_rebalance_ts).days
                if days_diff >= rebalance_interval_days:
                    self._rebalance_portfolio(param_map, rebalance_allocs, prices_at_ts, ts)
                    last_rebalance_ts = ts

            # track drawdown
            current_val = self._calc_portfolio_value(prices_at_ts)
            if current_val > self.portfolio.highest_equity:
                self.portfolio.highest_equity = current_val
            dd_ratio = (self.portfolio.highest_equity - current_val) / (
                self.portfolio.highest_equity if self.portfolio.highest_equity>0 else Decimal("1")
            )
            if dd_ratio > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = dd_ratio

        # final portfolio value
        final_prices = {}
        if timeline:
            last_ts = timeline[-1]
            for coin in coin_list:
                row = self._find_latest_ohlc(historical_data[coin], last_ts)
                if row:
                    final_prices[coin] = Decimal(str(row[4]))
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

    def _ai_generate_signal(self, coin, last_20_candles):
        if len(last_20_candles) < 20:
            return "hold"
        data=[]
        for r in last_20_candles:
            data.append({
                "time": r[0],
                "close": float(r[4]),
                "volume": float(r[5])
            })
        df= pd.DataFrame(data).sort_values("time").reset_index(drop=True)
        return self.ai_predictor.generate_signal(coin, df, seq_len=20, threshold_pct=0.01)

    def _fetch_ohlc(self, coin, timeframe, start_date, end_date):
        cursor = self.db.cursor()
        q = """
            SELECT 
                timestamp,
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
            out.append((
                r[0],  # timestamp
                None,  # open
                None,  # high
                None,  # low
                r[1],  # close
                r[2],  # volume
                r[3],  # boll_up
                r[4],  # boll_down
                r[5],  # macd
                r[6],  # macd_signal
                r[7],  # macd_diff
                r[8],  # ema_10
                r[9],  # ema_50
            ))
        return out

    def _find_latest_ohlc(self, rows, ts):
        best = None
        for r in rows:
            if r[0] <= ts:
                best = r
            else:
                break
        return best

    def _get_recent(self, rows, ts, lookback=50):
        subset = []
        for r in rows:
            if r[0] <= ts:
                subset.append(r)
        return subset[-lookback:]

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
        """
        Sells only the 'trading' portion of each coin, preserving the core ratio.
        """
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
                    # CHANGED: realistic 0.35% fee
                    cost_of_trade = sell_qty * price
                    fee_percent   = Decimal("0.0035")
                    fee           = cost_of_trade * fee_percent
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
                # CHANGED: 0.35% fee for buy
                cost_of_trade = buy_qty * price
                fee_percent   = Decimal("0.0035")
                fee           = cost_of_trade * fee_percent
                self.portfolio.record_trade(c, "buy", buy_qty, price, fee, ts)
        self.logger.warning("[PartialRebuy] Done partial re-buy in backtest.")

    def _liquidate_portfolio(self, prices_at_ts, ts, pm, catastrophic=False):
        """
        If catastrophic=True, sells the *entire* coin amount (core + trading).
        If catastrophic=False, sells only the 'trading portion', preserving the core ratio.
        """
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
                # CHANGED: 0.35% fee
                cost_of_trade = sell_qty * price
                fee_percent   = Decimal("0.0035")
                fee           = cost_of_trade * fee_percent
                self.portfolio.record_trade(coin, "sell", sell_qty, price, fee, ts)

        if catastrophic:
            self.logger.warning("[Backtest Meltdown] => Catastrophic meltdown => sold everything (core included).")
        else:
            self.logger.warning("[Backtest Meltdown] => Non-catastrophic => sold only trading portion.")

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
                # fully sell if we have holdings
                if hold_qty > 0:
                    cost_of_trade = hold_qty * price  # CHANGED
                    fee_percent   = Decimal("0.0035")
                    fee           = cost_of_trade * fee_percent
                    self.portfolio.record_trade(c, "sell", hold_qty, price, fee, ts)
                continue

            target_abs_val = current_val * target_w
            if target_abs_val <= 0:
                continue

            diff_ratio = (hold_val - target_abs_val) / (target_abs_val if target_abs_val>0 else Decimal("1"))

            if diff_ratio > threshold:
                # we have more than target => sell partial
                core_amt = hold_qty * core_ratio
                free_amt = hold_qty - core_amt
                if free_amt <= 0:
                    continue
                excess_val = hold_val - target_abs_val
                sell_val = min(excess_val, free_amt * price)
                if sell_val > 0:
                    sell_qty = sell_val / price
                    cost_of_trade = sell_qty * price  # CHANGED
                    fee_percent   = Decimal("0.0035")
                    fee           = cost_of_trade * fee_percent
                    self.portfolio.record_trade(c, "sell", sell_qty, price, fee, ts)

            elif diff_ratio < -threshold:
                # we have less than target => buy
                short_val = target_abs_val - hold_val
                if short_val > 0:
                    if short_val > stable_bal:
                        short_val = stable_bal
                    if short_val > 0:
                        buy_qty = short_val / price
                        cost_of_trade = buy_qty * price  # CHANGED
                        fee_percent   = Decimal("0.0035")
                        fee           = cost_of_trade * fee_percent
                        self.portfolio.record_trade(c, "buy", buy_qty, price, fee, ts)
                        stable_bal -= short_val

        self.logger.info("[Backtest Rebalance] completed a rebalance step.")

    def _check_dip_buy(self, pm, coin, recent_candles, ts):
        if len(recent_candles) < 2:
            return
        dip_buy_pct = Decimal(str(pm.get("dip_buy_pct_drop", 0.05)))
        last_close = Decimal(str(recent_candles[-1][4]))
        prev_close = Decimal(str(recent_candles[-2][4]))
        if prev_close > 0:
            pct = (last_close - prev_close) / prev_close
            if pct < -dip_buy_pct:
                stable_amt = self.portfolio.get_balance(self.stable_symbol)
                spend = stable_amt * Decimal("0.02")
                if spend < 5:
                    spend = Decimal("5")
                if spend > stable_amt:
                    spend = stable_amt
                if spend > 0 and last_close>0:
                    buy_qty = spend / last_close
                    cost_of_trade = buy_qty * last_close  # CHANGED
                    fee_percent   = Decimal("0.0035")
                    fee           = cost_of_trade * fee_percent
                    self.portfolio.record_trade(coin, "buy", buy_qty, last_close, fee, ts)

    def _check_partial_sell(self, pm, coin, recent_candles, ts):
        if len(recent_candles) < 20:
            return

        cost_basis = self.portfolio.coin_cost_basis.get(coin, Decimal("0"))
        if cost_basis <= 0:
            return

        last_close = Decimal(str(recent_candles[-1][4]))
        gain_ratio = (last_close - cost_basis) / cost_basis
        if gain_ratio < 0:
            return

        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands

        closes = [Decimal(str(x[4])) for x in recent_candles]
        df = pd.DataFrame(closes, columns=["close"])

        rsi_obj = RSIIndicator(df["close"], window=14)
        current_rsi = Decimal(str(rsi_obj.rsi().iloc[-1]))
        rsi_overbought = Decimal(str(pm.get("rsi_overbought", 70)))

        boll_period = int(pm.get("bollinger_period", 20))
        boll_stddev = float(pm.get("bollinger_stddev", 2.0))
        if len(df) >= boll_period:
            bb = BollingerBands(df["close"], window=boll_period, window_dev=boll_stddev)
            upper = Decimal(str(bb.bollinger_hband().iloc[-1]))
            overbought = (current_rsi > rsi_overbought) or (last_close > upper)
        else:
            overbought = (current_rsi > rsi_overbought)

        if not overbought:
            return

        partial_sell_tiers = pm.get("partial_sell_tiers", [])
        if not partial_sell_tiers:
            return

        core_ratio = Decimal(str(pm.get("core_ratio", 0.70)))
        total_qty = self.portfolio.get_balance(coin)
        trading_qty = total_qty - (total_qty * core_ratio)
        if trading_qty <= 0:
            return

        partial_sell_tiers = sorted(partial_sell_tiers, key=lambda x: Decimal(str(x["gain"])))
        leftover_trading = trading_qty
        for tier in partial_sell_tiers:
            tgain  = Decimal(str(tier["gain"]))
            tratio = Decimal(str(tier["ratio"]))
            if gain_ratio >= tgain:
                sell_qty = leftover_trading * tratio
                if sell_qty < Decimal("0.000001"):
                    continue
                # CHANGED: 0.35% fee
                cost_of_trade = sell_qty * last_close
                fee_percent   = Decimal("0.0035")
                fee           = cost_of_trade * fee_percent
                self.portfolio.record_trade(coin, "sell", sell_qty, last_close, fee, ts)
                leftover_trading -= sell_qty
                if leftover_trading <= Decimal("0.000001"):
                    break

    def _check_momentum_buy(self, pm, coin, recent_candles, ts):
        if len(recent_candles) < 20:
            return

        from ta.momentum import RSIIndicator
        closes = [Decimal(str(x[4])) for x in recent_candles]
        df = pd.DataFrame(closes, columns=["close"])
        rsi_obj= RSIIndicator(df["close"], window=14)
        rsi_val= Decimal(str(rsi_obj.rsi().iloc[-1]))
        thr= Decimal(str(pm.get("momentum_rsi", 60)))
        frac= Decimal(str(pm.get("momentum_fraction", 0.05)))
        if rsi_val> thr:
            stable_amt= self.portfolio.get_balance(self.stable_symbol)
            spend= stable_amt* frac
            if spend<10:
                spend= Decimal("10")
            if spend> stable_amt:
                spend= stable_amt
            if spend>0:
                last_close= df["close"].iloc[-1]
                buy_qty= Decimal(str(spend))/ Decimal(str(last_close))
                cost_of_trade = buy_qty * Decimal(str(last_close))  # CHANGED
                fee_percent   = Decimal("0.0035")
                fee           = cost_of_trade * fee_percent
                self.portfolio.record_trade(coin, "buy", buy_qty, Decimal(str(last_close)), fee, ts)
