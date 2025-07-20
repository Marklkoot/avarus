"""
strategy_margin.py

- Respects meltdown & stop-loss as fallback (catastrophic protection).
- AI overrides DIP buy / partial sells unless meltdown triggers.
- Dynamic position sizing: if AI says "buy_strong", we invest more; if "buy" normal, etc.
- Allows short selling if AI says "sell" or "sell_strong".
"""

import json
import logging
import pandas as pd
import datetime
from decimal import Decimal
import os

# Dynamic symbol lookup
from fundamental_engine import find_kraken_symbol_for_coin

from ml_engine.ai_signals_margin import AIPricePredictorMargin

LOG_PATH = r"C:\Users\markl\Avarus2\logs\margin_trading.log"

margin_logger = logging.getLogger("margin_trading")


class StrategyMargin:
    def __init__(self, db, exchange, config, portfolio_mgr):
        self.db = db
        self.cursor = db.cursor(dictionary=True)
        self.exchange = exchange
        self.config = config
        self.portfolio_mgr = portfolio_mgr
        
        self.stable_symbol = self.config.get("margin_portfolio", {}).get("stable_symbol", "EUR")

        self.params = self._load_dynamic_params()

        # meltdown
        self.meltdown_threshold   = Decimal(str(self.params.get("meltdown_threshold", 0.15)))
        self.max_position_fraction= Decimal(str(self.params.get("max_position_fraction", 0.30)))

        # normal strategy thresholds
        self.core_ratio          = Decimal(str(self.params.get("core_ratio", 0.70)))
        self.momentum_rsi_threshold = Decimal(str(self.params.get("momentum_rsi", 60)))
        self.trailing_stop_pct   = Decimal(str(self.params.get("trailing_stop_pct", 0.04)))
        self.stop_loss_pct       = Decimal(str(self.params.get("stop_loss_pct", 0.02)))

        # DIP buy
        self.dip_buy_pct_drop    = Decimal(str(self.params.get("dip_buy_pct_drop", 0.03)))

        # (A) AI => 10-feature ensemble with confidence
        self.ai_predictor = AIPricePredictorMargin(db, config)

        # We'll define thresholds for "strong" vs. normal position sizes
        self.strong_position_mult = Decimal("0.10")  # 10% of stable if AI strong
        self.normal_position_mult = Decimal("0.05")  # 5% of stable if AI normal
        self.short_position_mult  = Decimal("0.04")  # if AI says "sell"

    def check_short_term_trades(self):
        self.params = self._load_dynamic_params()
        meltdown_triggered = self._meltdown_check()
        if meltdown_triggered:
            margin_logger.info("[StrategyMargin] meltdown => skip normal trades.")
            return

        # 1) Perform stop-loss checks first
        self._perform_stop_loss_checks()

        # 2) Gather *all* coins from ohlc_data plus coins in the portfolio
        #    so we can open new positions for coins we don't yet hold.
        distinct_coins = set()
        cursor = self.db.cursor()
        cursor.execute("SELECT DISTINCT coin FROM ohlc_data WHERE timeframe='1h'")
        rows = cursor.fetchall()
        for r in rows:
            distinct_coins.add(r[0])  # e.g. "APT","AVAX",...

        # also include coins in your portfolio
        pf = self.portfolio_mgr.get_portfolio()  # coin -> qty
        all_coins = distinct_coins.union(pf.keys())

        # 3) AI-based trades for each coin in that combined set
        for coin in all_coins:
            qty = pf.get(coin, Decimal("0"))

            # Skip stable symbol
            if coin == self.stable_symbol:
                continue
            # Skip certain fiats
            if coin in ["GBP", "USD"]:
                margin_logger.info(f"[StrategyMargin] Skipping fiat coin => {coin}")
                continue

            df_ai = self._fetch_ohlcv_for_ai(coin, '1h', limit=50)
            if df_ai is None or len(df_ai) < 20:
                # If we don't have at least 20 bars for the AI, skip
                continue

            # Use seq_len=20 for inference since you changed it in training
            ai_signal = self.ai_predictor.generate_signal(coin, df_ai, seq_len=20, threshold_pct=0.01)
            margin_logger.info(f"[StrategyMargin] coin={coin}, AI => {ai_signal}")

            current_qty = Decimal(str(qty))  # net position for this coin
            symbol = self._map_coin(coin)
            if not symbol:
                continue

            # (A) "Buy" or "Buy_strong"
            if ai_signal in ("buy", "buy_strong"):
                if current_qty < 0:
                    # We are net SHORT => a "buy" closes/reduces that short
                    buy_amt = abs(current_qty)
                    if buy_amt > 0:
                        order = self._place_order_with_min_check(
                            self.exchange, "market", symbol,
                            amount=buy_amt, side="buy"
                        )
                        if order:
                            fill_price = self._fetch_last_price(symbol)
                            fee = buy_amt * fill_price * Decimal("0.0035")
                            self.portfolio_mgr.record_trade(
                                coin, "buy", float(buy_amt),
                                float(fill_price), float(fee),
                                trade_table="trade_history_margin"
                            )
                else:
                    # current_qty >= 0 => flat or already long => open/add to long
                    stable_amt = pf.get(self.stable_symbol, Decimal("0"))
                    if stable_amt > 5:
                        if ai_signal == "buy_strong":
                            spend = stable_amt * self.strong_position_mult
                        else:
                            spend = stable_amt * self.normal_position_mult

                        spend = self._enforce_position_size(spend)
                        if spend > stable_amt:
                            spend = stable_amt

                        last_close = Decimal(str(df_ai.iloc[-1]["close"]))
                        if spend > 0 and last_close > 0:
                            buy_amt = spend / last_close
                            order = self._place_order_with_min_check(
                                self.exchange, "market", symbol,
                                amount=buy_amt, side="buy"
                            )
                            if order is not None:
                                fill_price = self._fetch_last_price(symbol)
                                fee = buy_amt * fill_price * Decimal("0.0035")
                                self.portfolio_mgr.record_trade(
                                    coin, "buy", float(buy_amt),
                                    float(fill_price), float(fee),
                                    trade_table="trade_history_margin"
                                )

            # (B) "Sell" or "Sell_strong"
            elif ai_signal in ("sell", "sell_strong"):
                if current_qty > 0:
                    # net LONG => "sell" means reduce or close
                    if ai_signal == "sell_strong":
                        sell_amt = current_qty
                    else:
                        sell_amt = current_qty * Decimal("0.50")

                    if sell_amt > 0:
                        order = self._place_order_with_min_check(
                            self.exchange, "market", symbol,
                            amount=sell_amt, side="sell"
                        )
                        if order:
                            fill_price = self._fetch_last_price(symbol)
                            fee = sell_amt * fill_price * Decimal("0.0035")
                            self.portfolio_mgr.record_trade(
                                coin, "sell", float(sell_amt),
                                float(fill_price), float(fee),
                                trade_table="trade_history_margin"
                            )
                else:
                    # net SHORT or flat => open or increase short
                    stable_amt = pf.get(self.stable_symbol, Decimal("0"))
                    if ai_signal == "sell_strong":
                        short_spend = stable_amt * Decimal("0.30")
                    else:
                        short_spend = stable_amt * self.short_position_mult

                    short_spend = self._enforce_position_size(short_spend)
                    if short_spend > stable_amt:
                        short_spend = stable_amt

                    last_close = Decimal(str(df_ai.iloc[-1]["close"]))
                    if short_spend > 0 and last_close > 0:
                        short_qty = short_spend / last_close
                        order = self._place_order_with_min_check(
                            self.exchange, "market", symbol,
                            amount=short_qty, side="sell"
                        )
                        if order:
                            fill_price = self._fetch_last_price(symbol)
                            fee = short_qty * fill_price * Decimal("0.0035")
                            self.portfolio_mgr.record_trade(
                                coin, "sell", float(short_qty),
                                float(fill_price), float(fee),
                                trade_table="trade_history_margin"
                            )

        # 3) trailing stops
        self._perform_trailing_stops()

        # 4) now perform the comprehensive profit-taking check
        self._take_profit_check()


    def _perform_stop_loss_checks(self):
        pf = self.portfolio_mgr.get_portfolio()
        for coin, qty in pf.items():
            if coin == self.stable_symbol:
                continue

            # skip GBP, USD so no fallback warnings
            if coin in ["GBP", "USD"]:
                continue

            self._per_coin_stop_loss_check(coin)

    def _perform_trailing_stops(self):
        pf = self.portfolio_mgr.get_portfolio()
        for c, qty_ in pf.items():
            if c == self.stable_symbol or qty_ <= 0:
                continue

            if c in ["GBP", "USD"]:
                continue

            self._check_trailing_stop(c)

    # ============================
    # (**) COOLDOWN HELPER: store meltdown cooldown time
    def _save_meltdown_cooldown_time(self, until_dt):
        param_name = 'meltdown_cooldown_until'
        dt_str = until_dt.strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)"""
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute(q, (param_name, dt_str, now_))
        self.db.commit()

    # (**) COOLDOWN HELPER: get meltdown cooldown time
    def _get_meltdown_cooldown_time(self):
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name='meltdown_cooldown_until'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            dt_str = row["param_value"]
            return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        else:
            return None
    # ============================

    # (C) meltdown fallback
    def _meltdown_check(self):
        # (**) Check meltdown cooldown at the start
        cooldown_until = self._get_meltdown_cooldown_time()
        now_ = datetime.datetime.now()
        if cooldown_until and now_ < cooldown_until:
            margin_logger.info(f"[Meltdown] => on cooldown until {cooldown_until}, skip meltdown.")
            return False

        meltdown_anchor = self._get_meltdown_anchor()
        if meltdown_anchor is None:
            val = self._calc_net_value()
            self._save_meltdown_anchor(val)
            return False

        current_val = self._calc_net_value()
        if meltdown_anchor > 0:
            drawdown = (meltdown_anchor - current_val) / meltdown_anchor
        else:
            drawdown = Decimal("0")

        meltdown_stage = self._get_meltdown_stage()
        meltdown_tiers = self.params.get("meltdown_tiers", [])
        meltdown_reentry_pct = Decimal(str(self.params.get("meltdown_reentry_pct", "0.10")))
        meltdown_low = self._get_meltdown_low()
        if meltdown_low is None:
            meltdown_low = current_val
            self._save_meltdown_low(meltdown_low)

        triggered = False
        if meltdown_stage < len(meltdown_tiers):
            next_threshold = Decimal(str(meltdown_tiers[meltdown_stage]["drawdown"]))
            next_sell_ratio = Decimal(str(meltdown_tiers[meltdown_stage]["sell_ratio"]))
            if drawdown >= next_threshold:
                margin_logger.warning(f"[Meltdown] stage={meltdown_stage}, ratio={next_sell_ratio}")
                self._partial_liquidate_portfolio(next_sell_ratio)
                meltdown_stage += 1
                self._save_meltdown_stage(meltdown_stage)
                meltdown_low = min(meltdown_low, current_val)
                self._save_meltdown_low(meltdown_low)
                triggered = True
        else:
            if drawdown > self.meltdown_threshold:
                margin_logger.warning(f"[Meltdown] => catastrophic => full liquidation.")
                self._liquidate_portfolio()
                triggered = True
                meltdown_stage = len(meltdown_tiers)
                self._save_meltdown_stage(meltdown_stage)

                # (**) AFTER FULL MELTDOWN => RESET meltdown_anchor to new val
                new_val = self._calc_net_value()
                self._save_meltdown_anchor(new_val)
                margin_logger.warning(f"[Meltdown] => anchor reset => {new_val}")

                # (**) SET meltdown cooldown => e.g. 60 minutes
                cooldown_ends = now_ + datetime.timedelta(minutes=60)
                self._save_meltdown_cooldown_time(cooldown_ends)
                margin_logger.warning(f"[Meltdown] => meltdown cooldown => skip meltdown until {cooldown_ends}")

        # re-entry
        if meltdown_stage > 0:
            if current_val > meltdown_low * (Decimal("1") + meltdown_reentry_pct):
                old_stage = meltdown_stage
                meltdown_stage -= 1
                if meltdown_stage < 0:
                    meltdown_stage = 0
                self._save_meltdown_stage(meltdown_stage)
                ratio = Decimal("0")
                if meltdown_stage < len(meltdown_tiers):
                    ratio = Decimal(str(meltdown_tiers[old_stage - 1]["sell_ratio"]))
                if ratio > Decimal("0"):
                    margin_logger.warning(f"[Meltdown Reentry] => meltdown_stage => {meltdown_stage}, ratio={ratio}")
                    self._partial_rebuy_portfolio(ratio)
                meltdown_low = current_val
                self._save_meltdown_low(meltdown_low)

        anchor_buf = Decimal(str(self.params.get("meltdown_anchor_buffer", "0.02")))
        threshold_new_anchor = meltdown_anchor * (Decimal("1") + anchor_buf)
        if current_val > threshold_new_anchor:
            self._save_meltdown_anchor(current_val)

        return triggered

    def _liquidate_portfolio(self):
        """
        Sells all net long positions and buys all net short positions to bring them to zero.
        This is the “catastrophic meltdown,” triggered when drawdown is beyond meltdown_threshold
        or if we’ve run out of meltdown tiers.
        """
        pf = self.portfolio_mgr.get_portfolio()
        stable = self.stable_symbol

        all_closed = True

        for coin, amt in pf.items():
            # skip stable symbol & fiat coins we can’t margin trade
            if coin == stable or coin in ["GBP","USD"]:
                continue

            position_qty = Decimal(str(amt))
            if position_qty == 0:
                continue

            sym = self._map_coin(coin)
            if not sym:
                continue

            # If we have a net long (>0), meltdown => "sell"
            # If net short (<0), meltdown => "buy" the absolute qty to close the short
            if position_qty > 0:
                side = "sell"
                liquidation_amt = position_qty
            else:
                side = "buy"
                liquidation_amt = abs(position_qty)

            order = self._place_order_with_min_check(
                self.exchange, "market", sym, amount=liquidation_amt, side=side
            )
            if order is not None:
                fill_price = self._fetch_last_price(sym)
                fee = fill_price * liquidation_amt * Decimal("0.0035")
                self.portfolio_mgr.record_trade(
                    coin, side, float(liquidation_amt),
                    float(fill_price), float(fee),
                    trade_table="trade_history_margin"
                )
            else:
                # if order is None, meltdown couldn't close that position (e.g. margin restricted)
                margin_logger.warning(f"[MeltdownSell] => {side} {liquidation_amt} {coin} => order failed.")
                all_closed = False

        if all_closed:
            margin_logger.warning("[Meltdown] => fully stable now (all positions closed).")
        else:
            margin_logger.warning("[Meltdown] => meltdown attempted, but some positions remain.")

    def _partial_liquidate_portfolio(self, ratio):
        """
        Sells 'ratio' fraction of net long positions and
        buys 'ratio' fraction of net short positions.
        For meltdown tiers, e.g. ratio=0.30 means close 30% of each position.
        """
        pf = self.portfolio_mgr.get_portfolio()
        stable = self.stable_symbol
        any_failed = False

        for coin, amt in pf.items():
            if coin == stable or coin in ["GBP","USD"]:
                continue

            position_qty = Decimal(str(amt))
            if position_qty == 0:
                continue

            sym = self._map_coin(coin)
            if not sym:
                continue

            # meltdown partial => ratio% of each net position
            if position_qty > 0:
                # net LONG => "sell" ratio
                side = "sell"
                close_amt = position_qty * Decimal(str(ratio))
            else:
                # net SHORT => "buy" ratio
                side = "buy"
                close_amt = abs(position_qty) * Decimal(str(ratio))

            if close_amt <= 0:
                continue

            order = self._place_order_with_min_check(
                self.exchange, "market", sym,
                amount=close_amt, side=side
            )
            if order is not None:
                fill_price = self._fetch_last_price(sym)
                fee = fill_price * close_amt * Decimal("0.0035")
                self.portfolio_mgr.record_trade(
                    coin, side, float(close_amt),
                    float(fill_price), float(fee),
                    trade_table="trade_history_margin"
                )
            else:
                margin_logger.warning(f"[PartialMeltdown] => {side} {close_amt} {coin} => order failed.")
                any_failed = True

        if not any_failed:
            margin_logger.warning("[PartialMeltdown] => done partial sells/buys.")
        else:
            margin_logger.warning("[PartialMeltdown] => attempted, but some positions not fully closed.")

    def _partial_rebuy_portfolio(self, ratio):
        pf = self.portfolio_mgr.get_portfolio()
        stable_bal = pf.get(self.stable_symbol, Decimal("0"))
        coins = [c for c in pf.keys() if c != self.stable_symbol]
        if stable_bal <= 0 or not coins:
            return
        per_coin_spend = stable_bal * ratio / Decimal(str(len(coins)))
        for c in coins:
            if c in ["GBP","USD"]:
                continue
            if per_coin_spend < 5:
                continue
            sym = self._map_coin(c)
            if not sym:
                continue
            px = Decimal(str(self._fetch_last_price(sym)))
            if px > 0:
                buy_qty = per_coin_spend / px
                try:
                    order = self._place_order_with_min_check(
                        self.exchange, "market", sym,
                        amount=buy_qty, side="buy"
                    )
                    if order is not None:
                        fill_price = self._fetch_last_price(sym)
                        fee = fill_price * buy_qty * Decimal("0.0035")
                        self.portfolio_mgr.record_trade(
                            c, "buy", float(buy_qty),
                            float(fill_price), float(fee),
                            trade_table="trade_history_margin"
                        )
                except Exception as e:
                    margin_logger.warning(f"[PartialRebuy] => {e}")

    def _save_meltdown_anchor(self, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES ('meltdown_anchor', %s, %s)"""
        self.cursor.execute(q, (str(val), now_))
        self.db.commit()

    def _get_meltdown_anchor(self):
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name='meltdown_anchor'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row["param_value"]))
        return None

    def _save_meltdown_stage(self, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES ('meltdown_stage', %s, %s)"""
        self.cursor.execute(q, (str(val), now_))
        self.db.commit()

    def _get_meltdown_stage(self):
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name='meltdown_stage'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            return int(row["param_value"])
        return 0

    def _save_meltdown_low(self, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES ('meltdown_low', %s, %s)"""
        self.cursor.execute(q, (str(val), now_))
        self.db.commit()

    def _get_meltdown_low(self):
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name='meltdown_low'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row["param_value"]))
        return None

    # (D) STOP-LOSS
    def _per_coin_stop_loss_check(self, coin):
        anchor = self._get_coin_stop_anchor(coin)
        if anchor is None:
            sym = self._map_coin(coin)
            if not sym:
                return
            cur_price = Decimal(str(self._fetch_last_price(sym)))
            if cur_price > 0:
                self._save_coin_stop_anchor(coin, cur_price)
            return

        sym = self._map_coin(coin)
        if not sym:
            return
        cur_price = Decimal(str(self._fetch_last_price(sym)))
        if anchor > 0 and cur_price > 0:
            drop_ratio = (anchor - cur_price) / anchor
            if drop_ratio > self.stop_loss_pct:
                margin_logger.warning(f"[StopLoss] {coin} => drop={drop_ratio*100:.2f}% => forced sell.")
                pf = self.portfolio_mgr.get_portfolio()
                amt = pf.get(coin, 0)
                if amt > 0:
                    try:
                        order = self._place_order_with_min_check(
                            self.exchange, "market", sym,
                            amount=amt, side="sell"
                        )
                        if order is not None:
                            fill_price = self._fetch_last_price(sym)
                            fee = fill_price * Decimal(str(amt)) * Decimal("0.0035")
                            self.portfolio_mgr.record_trade(
                                coin, "sell", float(amt), float(fill_price), float(fee),
                                trade_table="trade_history_margin"
                            )
                            margin_logger.warning(f"[StopLoss] {coin} => sold.")
                    except Exception as e:
                        margin_logger.warning(f"[StopLoss error] => {e}")
            else:
                # update anchor if new high
                if cur_price > anchor:
                    self._save_coin_stop_anchor(coin, cur_price)

    def _get_coin_stop_anchor(self, coin):
        param_name = f"stop_anchor_{coin}"
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name=%s"""
        self.cursor.execute(q, (param_name,))
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row["param_value"]))
        return None

    def _save_coin_stop_anchor(self, coin, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        param_name = f"stop_anchor_{coin}"
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)"""
        self.cursor.execute(q, (param_name, str(val), now_))
        self.db.commit()

    # (E) TRAILING STOP
    def _check_trailing_stop(self, coin):
        pf = self.portfolio_mgr.get_portfolio()
        if coin not in pf or pf[coin] <= 0:
            return
        sym = self._map_coin(coin)
        if not sym:
            return
        cur_price = Decimal(str(self._fetch_last_price(sym)))
        if cur_price <= 0:
            return

        anchor_val = self._get_trailing_anchor(coin)
        if anchor_val is None or cur_price > anchor_val:
            self._save_trailing_anchor(coin, cur_price)
            return

        drop_threshold = anchor_val * (Decimal("1") - self.trailing_stop_pct)
        if cur_price < drop_threshold:
            margin_logger.info(f"[TrailingStop] {coin} => triggered => SELL portion.")
            amt_to_sell = pf[coin] * Decimal("0.50")
            if amt_to_sell > 0:
                try:
                    order = self._place_order_with_min_check(
                        self.exchange, "market", sym,
                        amount=amt_to_sell, side="sell"
                    )
                    if order is not None:
                        fill_price = self._fetch_last_price(sym)
                        fee = fill_price * amt_to_sell * Decimal("0.0035")
                        self.portfolio_mgr.record_trade(
                            coin, "sell", float(amt_to_sell),
                            float(fill_price), float(fee),
                            trade_table="trade_history_margin"
                        )
                except Exception as e:
                    margin_logger.warning(f"[TrailingStop] => {e}")
            self._save_trailing_anchor(coin, cur_price)

    def _get_trailing_anchor(self, coin):
        param_name = f"trailing_stop_anchor_{coin}"
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name=%s"""
        self.cursor.execute(q, (param_name,))
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row["param_value"]))
        return None

    def _save_trailing_anchor(self, coin, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        param_name = f"trailing_stop_anchor_{coin}"
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)"""
        self.cursor.execute(q, (param_name, str(val), now_))
        self.db.commit()

    # (F) Net Value => meltdown & position sizing
    def _calc_net_value(self):
        """
        Calculates total net value (mark-to-market) of the entire portfolio, 
        including both long and short positions, in terms of the stable symbol.
        """
        pf = self.portfolio_mgr.get_portfolio()
        stable_bal = Decimal(str(pf.get(self.stable_symbol, 0)))
        total_val = stable_bal  # start with stable coin portion

        for coin, qty in pf.items():
            # Skip stable coin and other fiats you don't margin-trade
            if coin == self.stable_symbol:
                continue
            if coin in ["GBP", "USD"]:
                continue

            symbol = self._map_coin(coin)
            if not symbol:
                continue

            last_price = self._fetch_last_price(symbol)
            # If price fetch failed or is None => skip
            if last_price is None or last_price <= 0:
                margin_logger.warning(f"[_calc_net_value] Skipping {coin} => no valid price.")
                continue

            # Add if long, subtract if short
            if qty > 0:
                total_val += qty * last_price
            else:
                total_val -= abs(qty) * last_price

        return total_val

    # (G) Enforce position size => max_position_fraction
    def _enforce_position_size(self, spend):
        net_val = self._calc_net_value()
        limit_ = net_val * self.max_position_fraction
        if spend > limit_:
            return limit_
        return spend

    # (H) Load dynamic from meta_parameters_margin
    def _load_dynamic_params(self):
        default_params = self.config.get("strategy_defaults", {})
        q = "SELECT param_name, param_value FROM meta_parameters_margin"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        param_map = {r["param_name"]: r["param_value"] for r in rows}
        for k, v in default_params.items():
            if k not in param_map:
                param_map[k] = v
        if "partial_sell_tiers" in param_map:
            try:
                param_map["partial_sell_tiers"] = json.loads(param_map["partial_sell_tiers"])
            except:
                param_map["partial_sell_tiers"] = []
        if "meltdown_tiers" in param_map:
            try:
                param_map["meltdown_tiers"] = json.loads(param_map["meltdown_tiers"])
            except:
                param_map["meltdown_tiers"] = []
        return param_map

    # (I) fetch 50 bars for the AI
    def _fetch_ohlcv_for_ai(self, coin, timeframe='1h', limit=50):
        import pandas as pd
        cursor = self.db.cursor(dictionary=True)
        q = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlc_data
            WHERE coin=%s AND timeframe=%s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        cursor.execute(q, (coin, timeframe, limit))
        rows = cursor.fetchall()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.rename(columns={"timestamp": "time"}, inplace=True)
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def _fetch_ohlcv(self, symbol, timeframe='1h', limit=50):
        import pandas as pd
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not data:
                return None
            df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
            return df
        except Exception as e:
            margin_logger.warning(f"[StrategyMargin] fetch_ohlcv => {symbol} => {e}")
            return None

    # (J) basic fetch last price
    def _fetch_last_price(self, symbol):
        """
        Attempts to fetch a live price via CCXT; if that fails,
        falls back to the last known DB price. Returns a Decimal or None.
        """

        # 1) Try live price
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            live_price = ticker.get("last") or ticker.get("close")
            if live_price is not None:
                price_dec = Decimal(str(live_price))
                # If price_dec is truly 0, that's suspicious; treat it as invalid
                if price_dec > 0:
                    return price_dec
                margin_logger.warning(f"[LivePrice] {symbol} fetched as 0 => ignoring.")
            else:
                margin_logger.warning(f"[LivePrice] No 'last'/'close' in ticker for {symbol}.")
        except Exception as e:
            margin_logger.warning(f"[LivePrice] Exception fetching {symbol} => {e}")

        # 2) Fallback: use the last known DB price
        #    Because our 'symbol' is something like "BTC/EUR", we can parse out the coin part
        base_coin, quote = symbol.split("/")
        db_price = self._fetch_last_db_price(base_coin, timeframe="1h")
        if db_price is not None and db_price > 0:
            margin_logger.warning(f"[FallbackPrice] Using last DB price for {symbol} => {db_price}")
            return db_price

        # 3) If we reached here, no valid price was found
        margin_logger.error(f"[fetch_last_price] No valid price found for {symbol}. Returning None.")
        return None

    def _fetch_last_db_price(self, coin, timeframe='1h'):
        """
        Fallback: Get the last known close price from 'ohlc_data'
        for a given coin + timeframe. Returns Decimal or None.
        """
        try:
            query = """
                SELECT close
                FROM ohlc_data
                WHERE coin=%s AND timeframe=%s
                ORDER BY timestamp DESC
                LIMIT 1
            """
            self.cursor.execute(query, (coin, timeframe))
            row = self.cursor.fetchone()
            if row:
                return Decimal(str(row[0]))
            return None
        except Exception as e:
            margin_logger.warning(f"[FallbackPrice] DB fetch failed for coin={coin}, {e}")
            return None

    # (K) Dynamic _map_coin - returns None if GBP/USD
    def _map_coin(self, coin):
        """
        Convert a base coin (like "BTC","ETH","AAVE") to a Kraken symbol (e.g. "XBT","XETH","AAVE"),
        then append /EUR. If dynamic detection via `find_kraken_symbol_for_coin` fails,
        we fallback to the config-based map.
        """

        # 1) Skip certain fiat coins
        if coin in ["GBP","USD"]:
            return None

        # 2) Attempt dynamic detection (optional)
        dyn_symbol = find_kraken_symbol_for_coin(
            kraken=self.exchange,
            coin=coin,
            preferred_quotes=[self.stable_symbol]
        )
        if dyn_symbol:
            return dyn_symbol  # e.g. "AAVE/EUR"

        # 3) fallback => read 'kraken_map' from config
        kr_map = self.config.get("kraken_map", {})

        # if coin is in the map (like "BTC" -> "XBT", "AAVE" -> "AAVE", etc.)
        base_symbol = kr_map.get(coin, coin)

        # 4) build final pair => e.g. "AAVE/EUR"
        fallback = f"{base_symbol}/{self.stable_symbol}"

        # log a warning if we had to fallback
        margin_logger.warning(
            f"[StrategyMargin] Could not find dynamic symbol => fallback to {fallback}"
        )
        return fallback

    def _place_order_with_min_check(self, exchange, order_type, symbol, amount, price=None, side="buy"):
        exchange.load_markets()
        if symbol not in exchange.markets:
            margin_logger.warning(f"[MarginOrderMinCheck] {symbol} => not in {exchange.id} => skip.")
            return None
        market_info = exchange.markets[symbol]
        min_amt = market_info.get("limits", {}).get("amount", {}).get("min", 0.0)
        if min_amt is None:
            min_amt = 0.0
        if float(amount) < float(min_amt):
            margin_logger.warning(
                f"[MarginOrderMinCheck] => {symbol}, requested {amount} < min {min_amt} => skip."
            )
            return None
        try:
            if order_type == "market":
                if side == "buy":
                    return exchange.create_market_buy_order(
                        symbol, float(amount),
                        params={"leverage": 2}
                    )
                else:
                    return exchange.create_market_sell_order(
                        symbol, float(amount),
                        params={"leverage": 2}
                    )
            else:
                if side == "buy":
                    return exchange.create_limit_buy_order(
                        symbol, float(amount), float(price),
                        params={"leverage": 2}
                    )
                else:
                    return exchange.create_limit_sell_order(
                        symbol, float(amount), float(price),
                        params={"leverage": 2}
                    )
                    
            margin_logger.info(
                f"[MarginOrderMinCheck] => {symbol} => {side} {amount} => ORDER SUCCESS => {order}"
            )
            return order
        
        except Exception as e:
            margin_logger.warning(f"[MarginOrderMinCheck] => {symbol} => {e}")
            return None


    # === ADDED START ===
    # Comprehensive Profit-Taking Strategy (tiered partial exit, time-based, AI synergy)

    def _take_profit_check(self):
        """
        Tiered partial profit-taking + time-based exit + AI synergy.
        However, if after 1 hour we are at a loss but AI is still bullish,
        we skip the forced exit and allow another hour to recover.

        partial_sell_tiers => e.g. [ (0.03,0.30), (0.06,0.30), (0.10,0.40) ]
        max_hold_hours => 1
        """
        partial_sell_tiers = [
            (Decimal("0.010"), Decimal("0.30")), 
            (Decimal("0.020"), Decimal("0.30")),
            (Decimal("0.030"), Decimal("0.40"))
        ]
        max_hold_hours = 12

        pf = self.portfolio_mgr.get_portfolio()
        now_ = datetime.datetime.now()

        for coin, qty in pf.items():
            if coin == self.stable_symbol or abs(qty) < Decimal("0.000001"):
                continue

            cost_basis = self._get_coin_cost_basis(coin)
            if cost_basis is None or cost_basis <= 0:
                continue

            symbol = self._map_coin(coin)
            if not symbol:
                continue

            current_price = self._fetch_last_price(symbol)
            if current_price is None or current_price <= 0:
                continue

            pos_open_time = self._get_position_open_time(coin)
            hours_held = None
            if pos_open_time:
                hours_held = (now_ - pos_open_time).total_seconds() / 3600.0

            # synergy with AI => e.g. "buy","buy_strong","sell","sell_strong","hold"
            ai_signal = self._fetch_ai_signal(coin)

            if qty > 0:
                # net LONG => partial-sell tiers as normal
                self._check_partial_sell_tiers(
                    coin, qty, cost_basis, current_price, 
                    partial_sell_tiers, position_type="long"
                )

                # CHANGED START: time-based exit with loss-check
                if hours_held and hours_held >= max_hold_hours:
                    # Are we in profit or loss?
                    profit_ratio = (current_price - cost_basis) / cost_basis

                    if profit_ratio >= 0:
                        # If in profit, we do time-based forced exit as before
                        margin_logger.info(f"[TimeExit] {coin} => +{profit_ratio*100:.2f}% => SELL remainder.")
                        self._close_position(coin, qty, side="sell")

                    else:
                        # If at a loss, check AI
                        if ai_signal in ["buy","buy_strong"]:
                            # AI says "still bullish," skip forced exit => hold another hour
                            margin_logger.info(
                                f"[TimeExit-Loss] {coin} => at a loss but AI still bullish => hold."
                            )
                        else:
                            # AI not bullish => close
                            margin_logger.info(
                                f"[TimeExit-Loss] {coin} => at a loss, AI not bullish => SELL remainder."
                            )
                            self._close_position(coin, qty, side="sell")
                else:
                    # if AI flips to not bullish => close if in profit
                    if ai_signal not in ["buy","buy_strong"]:
                        profit_ratio = (current_price - cost_basis)/cost_basis
                        if profit_ratio > 0:
                            margin_logger.info(
                                f"[ProfitTake AIFlip] {coin} => AI not bullish => SELL remainder in profit."
                            )
                            self._close_position(coin, qty, side="sell")
                # CHANGED END

            else:
                # net SHORT => partial-sell tiers
                abs_qty = abs(qty)
                self._check_partial_sell_tiers(
                    coin, abs_qty, cost_basis, current_price, 
                    partial_sell_tiers, position_type="short"
                )

                # same logic for short side
                if hours_held and hours_held >= max_hold_hours:
                    profit_ratio = (cost_basis - current_price) / cost_basis
                    if profit_ratio >= 0:
                        # in profit => close
                        margin_logger.info(f"[TimeExit] {coin} => short +{profit_ratio*100:.2f}% => BUY remainder.")
                        self._close_position(coin, abs_qty, side="buy")
                    else:
                        # at a loss => check AI
                        if ai_signal in ["sell","sell_strong"]:
                            margin_logger.info(
                                f"[TimeExit-Loss] {coin} => short in loss but AI still bearish => hold."
                            )
                        else:
                            margin_logger.info(
                                f"[TimeExit-Loss] {coin} => short in loss, AI not bearish => BUY remainder."
                            )
                            self._close_position(coin, abs_qty, side="buy")
                else:
                    # if AI flips to not bearish => close if in profit
                    if ai_signal not in ["sell","sell_strong"]:
                        profit_ratio = (cost_basis - current_price)/cost_basis
                        if profit_ratio > 0:
                            margin_logger.info(
                                f"[ProfitTake AIFlip] {coin} => short => AI not bearish => BUY remainder in profit."
                            )
                            self._close_position(coin, abs_qty, side="buy")

    def _fetch_ai_signal(self, coin):
        """
        A helper to quickly re-fetch the AI's signal if needed.
        Could replicate your code that calls `generate_signal()` with 1h bars, etc.
        Or just return 'hold' if no new data. Adjust as you see fit.
        """
        df_ai = self._fetch_ohlcv_for_ai(coin, '1h', limit=50)
        if df_ai is None or len(df_ai) < 20:
            return "hold"
        return self.ai_predictor.generate_signal(coin, df_ai, seq_len=20, threshold_pct=0.01)


    def _check_partial_sell_tiers(self, coin, qty, cost_basis, current_price, tiers, position_type="long"):
        """
        Check each partial-sell tier. For net LONG: if current_price >= cost_basis*(1+ratio),
        sell that fraction. For net SHORT: if current_price <= cost_basis*(1-ratio), buy that fraction.
        We'll track each tier once so we don't repeatedly sell the same tier.
        """
        sold_tiers = self._get_partial_sell_status(coin)
        # e.g. {"tier1": False, "tier2": True} stored in DB

        for i, (ratio, fraction) in enumerate(tiers, start=1):
            tier_key = f"tier{i}"
            if sold_tiers.get(tier_key, False):
                continue  # already sold that tier

            if position_type == "long":
                trigger_price = cost_basis * (Decimal("1") + ratio)
                if current_price >= trigger_price:
                    # sell fraction of qty
                    to_sell = qty * fraction
                    margin_logger.info(f"[TieredProfit] {coin} => Tier{i} at +{ratio*100:.1f}% => SELL {fraction*100}%")
                    self._close_or_partial_sell(coin, qty, portion=fraction)
                    sold_tiers[tier_key] = True
                    self._save_partial_sell_status(coin, sold_tiers)
                    qty -= to_sell

            else:
                # short => cost_basis*(1 - ratio)
                trigger_price = cost_basis * (Decimal("1") - ratio)
                if current_price <= trigger_price:
                    to_buy = qty * fraction
                    margin_logger.info(f"[TieredProfit] {coin} => Tier{i} short at +{ratio*100:.1f}% => BUY {fraction*100}%")
                    self._close_or_partial_buy(coin, qty, portion=fraction)
                    sold_tiers[tier_key] = True
                    self._save_partial_sell_status(coin, sold_tiers)
                    qty -= to_buy


    def _close_position(self, coin, qty, side=None):
        """
        Fully closes the position for 'coin'. If net long => side='sell',
        if net short => side='buy', unless side is specified.
        """
        if side is None:
            side = "sell" if qty>0 else "buy"

        symbol = self._map_coin(coin)
        if not symbol:
            return
        abs_qty = abs(qty)

        order = self._place_order_with_min_check(self.exchange, "market", symbol, amount=abs_qty, side=side)
        if order:
            fill_price = self._fetch_last_price(symbol)
            fee = fill_price * abs_qty * Decimal("0.0035")
            self.portfolio_mgr.record_trade(
                coin, side, float(abs_qty),
                float(fill_price), float(fee),
                trade_table="trade_history_margin"
            )

    def _close_or_partial_sell(self, coin, qty, portion=Decimal("1.0")):
        """
        Sells a fraction of the net-long position 'coin'.
        """
        if qty <= 0:
            return
        symbol = self._map_coin(coin)
        if not symbol:
            return
        to_sell = qty * portion
        order = self._place_order_with_min_check(self.exchange, "market", symbol, amount=to_sell, side="sell")
        if order:
            fill_price = self._fetch_last_price(symbol)
            fee = fill_price * to_sell * Decimal("0.0035")
            self.portfolio_mgr.record_trade(
                coin, "sell", float(to_sell),
                float(fill_price), float(fee),
                trade_table="trade_history_margin"
            )

    def _close_or_partial_buy(self, coin, qty, portion=Decimal("1.0")):
        """
        Buys a fraction of net-short position 'coin' to close or reduce it.
        """
        if qty <= 0:
            return
        symbol = self._map_coin(coin)
        if not symbol:
            return
        to_buy = qty * portion
        order = self._place_order_with_min_check(self.exchange, "market", symbol, amount=to_buy, side="buy")
        if order:
            fill_price = self._fetch_last_price(symbol)
            fee = fill_price * to_buy * Decimal("0.0035")
            self.portfolio_mgr.record_trade(
                coin, "buy", float(to_buy),
                float(fill_price), float(fee),
                trade_table="trade_history_margin"
            )

    def _get_coin_cost_basis(self, coin):
        # convenience to retrieve cost_basis from portfolio mgr.
        qty, cb = self.portfolio_mgr._get_balance_and_cb(coin)
        return cb

    def _get_position_open_time(self, coin):
        """
        Placeholder: fetch a datetime from DB param => position_open_time_{coin}
        Implement in portfolio manager or here. By default returns None.
        """
        param_name = f"position_open_time_{coin}"
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name=%s"""
        self.cursor.execute(q, (param_name,))
        row = self.cursor.fetchone()
        if not row:
            return None
        try:
            return datetime.datetime.strptime(row["param_value"], '%Y-%m-%d %H:%M:%S')
        except:
            return None

    def _get_partial_sell_status(self, coin):
        """
        Placeholder: returns a dict of sold tiers => {"tier1":True, "tier2":False} etc.
        """
        param_name = f"partial_sell_status_{coin}"
        q = """SELECT param_value FROM meta_parameters_margin WHERE param_name=%s"""
        self.cursor.execute(q, (param_name,))
        row = self.cursor.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["param_value"])
        except:
            return {}

    def _save_partial_sell_status(self, coin, status_dict):
        import json
        param_name = f"partial_sell_status_{coin}"
        val_str = json.dumps(status_dict)
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)"""
        self.cursor.execute(q, (param_name, val_str, now_))
        self.db.commit()

    # === ADDED END ===
