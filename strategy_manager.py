import json
import logging
import pandas as pd
import datetime
from decimal import Decimal

from fundamental_engine import find_kraken_symbol_for_coin
# === AI ADD ===
from ml_engine.ai_signals import AIPricePredictor


class StrategyManager:
    def __init__(self, db, exchange, config, portfolio_mgr):
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config
        self.portfolio_mgr = portfolio_mgr

        self.stable_symbol = self.config["portfolio"].get("stable_symbol", "USD")

        # Load dynamic params from DB (meta_parameters) or config
        self.params = self._load_dynamic_params()

        # meltdown
        self.meltdown_threshold = Decimal(str(self.params.get("meltdown_threshold", 0.30)))
        self.meltdown_anchor_buffer = Decimal(str(self.params.get("meltdown_anchor_buffer", 0.05)))
        self.max_position_fraction = Decimal(str(self.params.get("max_position_fraction", 0.30)))

        # normal strategy
        self.core_ratio = Decimal(str(self.params.get("core_ratio", 0.70)))
        self.dip_buy_pct_drop = Decimal(str(self.params.get("dip_buy_pct_drop", 0.05)))
        self.momentum_rsi_threshold = Decimal(str(self.params.get("momentum_rsi", 60)))
        self.momentum_fraction = Decimal(str(self.params.get("momentum_fraction", 0.05)))

        # NOTE: Trailing stops are removed, so we do not define trailing_stop_pct anymore
        self.stop_loss_pct = Decimal(str(self.params.get("stop_loss_pct", 0.35)))

        # partial-sell tiers
        partial_sell_tiers = self.params.get("partial_sell_tiers", [])
        if not isinstance(partial_sell_tiers, list):
            partial_sell_tiers = []
        self.partial_sell_tiers = sorted(
            partial_sell_tiers,
            key=lambda x: Decimal(str(x["gain"]))
        )

        # meltdown tiers
        meltdown_tiers = self.params.get("meltdown_tiers", [])
        if not isinstance(meltdown_tiers, list):
            meltdown_tiers = []
        self.meltdown_tiers = sorted(
            meltdown_tiers,
            key=lambda x: Decimal(str(x["drawdown"]))
        )
        self.meltdown_reentry_pct = Decimal(str(self.params.get("meltdown_reentry_pct", 0.10)))

        # === AI ADD ===
        self.ai_predictor = AIPricePredictor(db, config)
        # === END AI ADD ===

    def check_short_term_trades(self):
        # Reload params each time in case they changed in DB
        self.params = self._load_dynamic_params()

        meltdown_triggered = self._meltdown_check()
        meltdown_str = "yes" if meltdown_triggered else "no"

        # 1) STOP-LOSS check on each coin
        pf = self.portfolio_mgr.get_portfolio()
        stable = self.stable_symbol

        for c, qty_ in pf.items():
            if c == stable or qty_ <= 0:
                continue
            self._per_coin_stop_loss_check(c)

        # If meltdown triggered, skip other signals
        if meltdown_triggered:
            logging.info("[StrategyManager] meltdown triggered => skip other signals.")
        else:
            # 2) DIP BUY, PARTIAL SELL, MOMENTUM BUY, AI
            dip_buy_triggered = False
            partial_sell_triggered = False
            momentum_buy_triggered = False
            ai_action_triggered = False
            ai_action_type = "no"

            for coin, qty in pf.items():
                if coin == stable:
                    continue

                # OPTIONAL FUNDAMENTAL SCORE CHECK
                # score_ = self._load_coin_score(coin)
                # if score_ is not None and score_ < 1:
                #     continue

                df = self._fetch_ohlcv(self._map_coin(coin), '1h', 50)
                if df is None or len(df) < 2:
                    continue

                # DIP BUY
                dip_result = self._check_dip_buy(coin, df)
                if dip_result:
                    dip_buy_triggered = True

                # PARTIAL SELL
                sell_result = self._check_partial_sell(coin, qty, df)
                if sell_result:
                    partial_sell_triggered = True

                # MOMENTUM BUY
                mom_result = self._check_momentum_buy(coin, df)
                if mom_result:
                    momentum_buy_triggered = True

                # === AI ADD ===
                df_ai = self._fetch_ohlcv_for_ai(coin, '1h', limit=20)
                if df_ai is not None and len(df_ai) >= 20:
                    ai_signal = self.ai_predictor.generate_signal(
                        coin, df_ai, seq_len=20, threshold_pct=0.01
                    )
                    if ai_signal == "buy":
                        stable_amt = pf.get(stable, Decimal("0"))
                        if stable_amt > 5:
                            spend = stable_amt * Decimal("0.02")  # e.g. 2%
                            if spend < 5:
                                spend = Decimal("5")
                            if spend > stable_amt:
                                spend = stable_amt
                            last_close = Decimal(str(df_ai.iloc[-1]["close"]))
                            if last_close > 0:
                                buy_amt = spend / last_close
                                order = self._place_order_with_min_check(
                                    self.exchange, "market", self._map_coin(coin),
                                    amount=buy_amt, side="buy"
                                )
                                if order is not None:
                                    fill_price = self._fetch_last_price(self._map_coin(coin))
                                    fee = 0.2
                                    self.portfolio_mgr.record_trade(
                                        coin, "buy", float(buy_amt), float(fill_price), float(fee)
                                    )
                                    ai_action_triggered = True
                                    ai_action_type = "buy"
                    elif ai_signal == "sell":
                        coin_bal = pf.get(coin, Decimal("0"))
                        if coin_bal > 0:
                            sell_amt = coin_bal * Decimal("0.20")
                            last_close = Decimal(str(df_ai.iloc[-1]["close"]))
                            if sell_amt > 0 and last_close > 0:
                                order = self._place_order_with_min_check(
                                    self.exchange, "market", self._map_coin(coin),
                                    amount=sell_amt, side="sell"
                                )
                                if order is not None:
                                    fill_price = self._fetch_last_price(self._map_coin(coin))
                                    fee = 0.2
                                    self.portfolio_mgr.record_trade(
                                        coin, "sell", float(sell_amt), float(fill_price), float(fee)
                                    )
                                    ai_action_triggered = True
                                    ai_action_type = "sell"
                # === END AI ADD ===

        # Final logging
        dip_str = "yes" if locals().get('dip_buy_triggered', False) else "no"
        sell_str = "yes" if locals().get('partial_sell_triggered', False) else "no"
        mom_str = "yes" if locals().get('momentum_buy_triggered', False) else "no"
        ai_str = locals().get('ai_action_type', "no")

        if sell_str == "yes":
            action = "selling"
        elif dip_str == "yes" or mom_str == "yes":
            action = "buying"
        else:
            action = "holding"

        logging.info(
            f"Avarus is {action} => meltdown={meltdown_str}, dip_buy={dip_str}, "
            f"partial_sell={sell_str}, momentum_buy={mom_str}, ai={ai_str}"
        )

    # =================================================================
    # MELTDOWN
    # =================================================================
    def _meltdown_check(self):
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
        meltdown_low = self._get_meltdown_low()
        if meltdown_low is None:
            meltdown_low = current_val
            self._save_meltdown_low(meltdown_low)

        triggered_meltdown = False

        # partial meltdown tiers
        if meltdown_stage < len(self.meltdown_tiers):
            next_threshold = Decimal(str(self.meltdown_tiers[meltdown_stage]["drawdown"]))
            next_sell_ratio = Decimal(str(self.meltdown_tiers[meltdown_stage]["sell_ratio"]))

            if drawdown >= next_threshold:
                logging.warning(
                    f"[Meltdown] drawdown={drawdown:.2%} >= tier={next_threshold:.2%} => partial meltdown ratio={next_sell_ratio:.2%}"
                )
                self._partial_liquidate_portfolio(next_sell_ratio)
                meltdown_stage += 1
                self._save_meltdown_stage(meltdown_stage)
                meltdown_low = min(meltdown_low, current_val)
                self._save_meltdown_low(meltdown_low)
                triggered_meltdown = True
        else:
            # if meltdown_stage is at max, check final meltdown_threshold
            if drawdown > self.meltdown_threshold:
                logging.warning(
                    f"[Meltdown] => drop {drawdown*100:.2f}% => full meltdown."
                )
                self._liquidate_portfolio()
                triggered_meltdown = True
                meltdown_stage = len(self.meltdown_tiers)
                self._save_meltdown_stage(meltdown_stage)

        # meltdown reentry
        if meltdown_stage > 0:
            if current_val > meltdown_low * (Decimal("1") + self.meltdown_reentry_pct):
                old_stage = meltdown_stage
                meltdown_stage -= 1
                if meltdown_stage < 0:
                    meltdown_stage = 0
                self._save_meltdown_stage(meltdown_stage)

                ratio = Decimal("0")
                tier_index = old_stage - 1
                if tier_index < len(self.meltdown_tiers) and tier_index >= 0:
                    ratio = Decimal(str(self.meltdown_tiers[tier_index]["sell_ratio"]))

                if ratio > 0:
                    logging.warning(
                        f"[Meltdown Reentry] => portfolio recovered => re-buy ratio={ratio:.2%}"
                    )
                    self._partial_rebuy_portfolio(ratio)
                meltdown_low = current_val
                self._save_meltdown_low(meltdown_low)

        # meltdown anchor update
        threshold_new_anchor = meltdown_anchor * (Decimal("1") + self.meltdown_anchor_buffer)
        if current_val > threshold_new_anchor:
            self._save_meltdown_anchor(current_val)

        return triggered_meltdown

    def _calc_net_value(self):
        pf = self.portfolio_mgr.get_portfolio()
        stable_amt = Decimal(str(pf.get(self.stable_symbol, 0)))
        total_val = stable_amt

        for c, amt in pf.items():
            if c == self.stable_symbol or amt <= 0:
                continue
            df = self._fetch_ohlcv(self._map_coin(c), '1h', limit=1)
            if df is not None and len(df) > 0:
                last_close = Decimal(str(df.iloc[-1]["close"]))
                total_val += amt * last_close
        return total_val

    def _liquidate_portfolio(self):
        pf = self.portfolio_mgr.get_portfolio()
        stable = self.stable_symbol
        for c, amt in pf.items():
            if c != stable and amt > 0:
                try:
                    symbol = self._map_coin(c)
                    order = self._place_order_with_min_check(
                        self.exchange,
                        order_type="market",
                        symbol=symbol,
                        amount=amt,
                        side="sell"
                    )
                    if order is not None:
                        fill_price = self._fetch_last_price(symbol)
                        fee = 1.0
                        self.portfolio_mgr.record_trade(c, "sell", float(amt), float(fill_price), fee)
                except Exception as e:
                    logging.warning(f"[MeltdownSell error] => {e}")
        logging.warning("[Meltdown] portfolio fully stable now.")

    def _partial_liquidate_portfolio(self, ratio):
        pf = self.portfolio_mgr.get_portfolio()
        stable = self.stable_symbol
        for c, amt in pf.items():
            if c == stable or amt <= 0:
                continue
            sell_amt = amt * ratio
            if sell_amt > 0:
                try:
                    symbol = self._map_coin(c)
                    order = self._place_order_with_min_check(
                        self.exchange,
                        order_type="market",
                        symbol=symbol,
                        amount=sell_amt,
                        side="sell"
                    )
                    if order is not None:
                        fill_price = self._fetch_last_price(symbol)
                        fee = 1.0
                        self.portfolio_mgr.record_trade(c, "sell", float(sell_amt), float(fill_price), fee)
                except Exception as e:
                    logging.warning(f"[PartialMeltdown error] => {e}")
        logging.warning("[PartialMeltdown] Done partial liquidation.")

    def _partial_rebuy_portfolio(self, ratio):
        pf = self.portfolio_mgr.get_portfolio()
        stable_amt = Decimal(str(pf.get(self.stable_symbol, 0)))
        if stable_amt <= 0:
            logging.warning("[PartialRebuy] No stable funds to rebuy.")
            return

        # Buy across all coins equally
        coins = [c for c in pf.keys() if c != self.stable_symbol]
        if not coins:
            return

        chunk = stable_amt * ratio
        if chunk < Decimal("1"):
            logging.warning(f"[PartialRebuy] ratio is too small => {chunk}")
            return

        share_per_coin = chunk / len(coins)
        for c in coins:
            symbol = self._map_coin(c)
            last_price = Decimal(str(self._fetch_last_price(symbol)))
            if last_price <= 0:
                continue
            amt_coin = share_per_coin / last_price
            if amt_coin < Decimal("0.000001"):
                continue

            try:
                order = self._place_order_with_min_check(
                    self.exchange,
                    order_type="market",
                    symbol=symbol,
                    amount=amt_coin,
                    side="buy"
                )
                if order is not None:
                    fill_price = self._fetch_last_price(symbol)
                    fee = 1.0
                    self.portfolio_mgr.record_trade(c, "buy", float(amt_coin), float(fill_price), fee)
                else:
                    logging.warning(f"[PartialRebuy] skip => below min volume => {c}")
            except Exception as e:
                logging.warning(f"[PartialRebuy error] => {e}")

    def _get_meltdown_anchor(self):
        q = """SELECT param_value FROM meta_parameters WHERE param_name='meltdown_anchor'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row[0]))
        return None

    def _save_meltdown_anchor(self, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
               VALUES ('meltdown_anchor', %s, %s)"""
        self.cursor.execute(q, (str(val), now_))
        self.db.commit()

    def _get_meltdown_stage(self):
        q = """SELECT param_value FROM meta_parameters WHERE param_name='meltdown_stage'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            return int(row[0])
        return 0

    def _save_meltdown_stage(self, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
               VALUES ('meltdown_stage', %s, %s)"""
        self.cursor.execute(q, (str(val), now_))
        self.db.commit()

    def _get_meltdown_low(self):
        q = """SELECT param_value FROM meta_parameters WHERE param_name='meltdown_low'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row[0]))
        return None

    def _save_meltdown_low(self, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
               VALUES ('meltdown_low', %s, %s)"""
        self.cursor.execute(q, (str(val), now_))
        self.db.commit()

    # =================================================================
    # PER-COIN STOP LOSS
    # =================================================================
    def _per_coin_stop_loss_check(self, coin):
        anchor = self._get_coin_stop_anchor(coin)
        if anchor is None:
            cur_price = Decimal(str(self._fetch_last_price(self._map_coin(coin))))
            if cur_price > 0:
                self._save_coin_stop_anchor(coin, cur_price)
            return

        cur_price = Decimal(str(self._fetch_last_price(self._map_coin(coin))))
        if anchor > 0 and cur_price > 0:
            drop_ratio = (anchor - cur_price) / anchor
            if drop_ratio > self.stop_loss_pct:
                logging.warning(
                    f"[StopLoss] {coin} => drop {drop_ratio*100:.2f}% => forced sell."
                )
                pf = self.portfolio_mgr.get_portfolio()
                amt = pf.get(coin, 0)
                if amt > 0:
                    try:
                        symbol = self._map_coin(coin)
                        order = self._place_order_with_min_check(
                            self.exchange,
                            order_type="market",
                            symbol=symbol,
                            amount=amt,
                            side="sell"
                        )
                        if order is not None:
                            fill_price = self._fetch_last_price(symbol)
                            fee = 0.5
                            self.portfolio_mgr.record_trade(
                                coin, "sell", float(amt), float(fill_price), float(fee)
                            )
                            logging.warning(f"[StopLoss] {coin} fully sold.")
                    except Exception as e:
                        logging.warning(f"[StopLoss error] => {e}")
            else:
                # if price recovers above anchor, update anchor
                if cur_price > anchor:
                    self._save_coin_stop_anchor(coin, cur_price)

    def _get_coin_stop_anchor(self, coin):
        param_name = f"stop_anchor_{coin}"
        q = """SELECT param_value FROM meta_parameters WHERE param_name=%s"""
        self.cursor.execute(q, (param_name,))
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row[0]))
        return None

    def _save_coin_stop_anchor(self, coin, val):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        param_name = f"stop_anchor_{coin}"
        q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)"""
        self.cursor.execute(q, (param_name, str(val), now_))
        self.db.commit()

    # =================================================================
    # DIP BUY
    # =================================================================
    def _check_dip_buy(self, coin, df):
        last_close = Decimal(str(df.iloc[-1]["close"]))
        prev_close = Decimal(str(df.iloc[-2]["close"]))
        if prev_close <= 0:
            return False

        pct = (last_close - prev_close) / prev_close
        if pct < -self.dip_buy_pct_drop:
            logging.info(f"[{coin}] DIP => {pct*100:.2f}% => place buy.")
            pf = self.portfolio_mgr.get_portfolio()
            stable_amt = pf.get(self.stable_symbol, 0)
            spend = stable_amt * Decimal("0.02")
            if spend < 5:
                spend = Decimal("5")

            spend = self._enforce_position_size(spend)
            if spend > stable_amt:
                spend = stable_amt

            if spend > 0:
                amt_coin = spend / last_close
                try:
                    symbol = self._map_coin(coin)
                    order = self._place_order_with_min_check(
                        self.exchange, "limit", symbol, amt_coin, price=last_close, side="buy"
                    )
                    if order is not None:
                        logging.info(f"[RealTrade] DIP Buy placed => {order}")
                    else:
                        logging.warning(f"[DipBuy] skip => below min volume => {coin}")
                except Exception as e:
                    logging.warning(f"[DipBuy error] => {e}")
            return True
        return False

    # =================================================================
    # PARTIAL SELL
    # =================================================================
    def _check_partial_sell(self, coin, qty, df):
        cost_basis = self.portfolio_mgr.get_cost_basis(coin)
        if cost_basis <= 0:
            return False

        last_close = Decimal(str(df.iloc[-1]["close"]))
        gain_ratio = (last_close - Decimal(str(cost_basis))) / Decimal(str(cost_basis))
        if gain_ratio < 0:
            return False

        if len(df) < 20:
            return False

        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands

        rsi_period = 14
        rsi_obj = RSIIndicator(df["close"], window=rsi_period)
        current_rsi = Decimal(str(rsi_obj.rsi().iloc[-1]))
        rsi_overbought = Decimal(str(self.params.get("rsi_overbought", 70)))

        boll_period = int(self.params.get("bollinger_period", 20))
        boll_stddev = float(self.params.get("bollinger_stddev", 2.0))
        if len(df) < boll_period:
            return False

        bb = BollingerBands(df["close"], window=boll_period, window_dev=boll_stddev)
        upper_band = bb.bollinger_hband().iloc[-1]

        overbought = (
            current_rsi > rsi_overbought
            or (upper_band and last_close > Decimal(str(upper_band)))
        )
        if not overbought:
            return False

        pf = self.portfolio_mgr.get_portfolio()
        total_qty = pf.get(coin, Decimal("0"))
        core_qty = total_qty * self.core_ratio
        trading_qty = total_qty - core_qty
        if trading_qty <= 0:
            return False

        partial_sell_tiers = self.partial_sell_tiers
        if not partial_sell_tiers:
            return False

        triggered_any = False
        current_trading_qty = trading_qty

        current_max_sold = self._get_max_gain_sold_for_coin(coin)
        for tier in partial_sell_tiers:
            tier_gain = Decimal(str(tier["gain"]))
            tier_ratio = Decimal(str(tier["ratio"]))

            if gain_ratio >= tier_gain and tier_gain > current_max_sold:
                sell_amt = current_trading_qty * tier_ratio
                if sell_amt < Decimal("0.000001"):
                    continue

                symbol = self._map_coin(coin)
                try:
                    market_info = self.exchange.markets.get(symbol, {})
                    min_amt = market_info.get("limits", {}).get("amount", {}).get("min", 0.0)
                    if float(sell_amt) < float(min_amt):
                        if float(current_trading_qty) >= float(min_amt):
                            logging.warning(
                                f"[PartialSell] {coin} => rounding up sell_amt from {sell_amt} to min_lot={min_amt}"
                            )
                            sell_amt = Decimal(str(min_amt))
                        else:
                            logging.warning(
                                f"[PartialSell] skip => leftover={current_trading_qty} < min_lot={min_amt}"
                            )
                            continue

                    order = self._place_order_with_min_check(
                        self.exchange, "market", symbol, sell_amt, side="sell"
                    )
                    if order is not None:
                        fill_price = self._fetch_last_price(symbol)
                        self.portfolio_mgr.record_trade(
                            coin, "sell", float(sell_amt), float(fill_price), fee=0.0
                        )
                        logging.info(
                            f"[PartialSell] {coin} => sold {sell_amt} @ {fill_price}, "
                            f"gain={(gain_ratio * 100):.2f}%, tier={tier_gain}, RSI={current_rsi}"
                        )

                        current_trading_qty -= sell_amt
                        triggered_any = True

                        # update the max tier sold
                        self._save_max_gain_sold_for_coin(coin, float(tier_gain))

                        if current_trading_qty <= Decimal("0.000001"):
                            break
                    else:
                        logging.warning(f"[PartialSell] skip => below min volume => {coin}")
                except Exception as e:
                    logging.warning(f"[PartialSell error] => {e}")

        return triggered_any

    def _get_max_gain_sold_for_coin(self, coin):
        param_name = f"max_gain_sold_{coin}"
        q = """SELECT param_value FROM meta_parameters WHERE param_name=%s"""
        self.cursor.execute(q, (param_name,))
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row[0]))
        return Decimal("0")

    def _save_max_gain_sold_for_coin(self, coin, val):
        param_name = f"max_gain_sold_{coin}"
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)"""
        self.cursor.execute(q, (param_name, str(val), now_))
        self.db.commit()

    # =================================================================
    # MOMENTUM BUY
    # =================================================================
    def _check_momentum_buy(self, coin, df):
        try:
            from ta.momentum import RSIIndicator
        except ImportError:
            return False

        rsi_obj = RSIIndicator(df["close"], window=14)
        last_rsi = rsi_obj.rsi().iloc[-1]

        if last_rsi > float(self.momentum_rsi_threshold):
            pf = self.portfolio_mgr.get_portfolio()
            stable_amt = pf.get(self.stable_symbol, 0)
            spend = Decimal(str(stable_amt)) * self.momentum_fraction
            if spend < 10:
                spend = Decimal("10")

            spend = self._enforce_position_size(spend)
            if spend > stable_amt:
                spend = stable_amt

            if spend > 0:
                last_close = Decimal(str(df.iloc[-1]["close"]))
                amt_coin = spend / last_close
                try:
                    symbol = self._map_coin(coin)
                    order = self._place_order_with_min_check(
                        self.exchange, "limit", symbol, amt_coin, price=last_close, side="buy"
                    )
                    if order is not None:
                        logging.info(f"[RealTrade] Momentum Buy placed => {order}")
                    else:
                        logging.warning(f"[MomentumBuy] skip => below min volume => {coin}")
                except Exception as e:
                    logging.warning(f"[MomentumBuy] => {e}")
            return True
        return False

    # =================================================================
    # HELPER
    # =================================================================
    def _enforce_position_size(self, spend):
        net_val = self._calc_net_value()
        limit_ = net_val * self.max_position_fraction
        if spend > limit_:
            return limit_
        return spend

    def _map_coin(self, coin):
        stable = self.stable_symbol
        dyn_symbol = find_kraken_symbol_for_coin(
            kraken=self.exchange,
            coin=coin,
            preferred_quotes=[stable]
        )
        if dyn_symbol:
            return dyn_symbol

        kr_map = self.config.get("kraken_map", {})
        base = kr_map.get(coin, coin)
        fallback_pair = f"{base}/{stable}"
        logging.warning(f"[Strategy] Could not find dynamic symbol => fallback to {fallback_pair}")
        return fallback_pair

    def _fetch_ohlcv_for_ai(self, coin, timeframe='1h', limit=50):
        import pandas as pd
        cursor = self.db.cursor(dictionary=True)
        q = """
            SELECT timestamp, close, volume
            FROM ohlc_data
            WHERE coin=%s
              AND timeframe=%s
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
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not data:
                return None
            df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
            return df
        except Exception as e:
            logging.warning(f"[Strategy] fetch_ohlcv => {symbol} => {e}")
            return None

    def _fetch_last_price(self, symbol):
        try:
            t = self.exchange.fetch_ticker(symbol)
            return t.get('last', t.get('close', 0.0))
        except:
            return 0.0

    def _load_coin_score(self, coin):
        q = """
            SELECT metric_value
            FROM fundamentals
            WHERE coin=%s
              AND metric_name='score'
            ORDER BY date DESC
            LIMIT 1
        """
        self.cursor.execute(q, (coin,))
        row = self.cursor.fetchone()
        if row:
            return float(row[0])
        return None

    def _load_dynamic_params(self):
        """Loads defaults from config plus overrides from meta_parameters."""
        default_params = self.config.get("strategy_defaults", {})
        q = "SELECT param_name, param_value FROM meta_parameters"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        param_map = {r[0]: r[1] for r in rows}

        for k, v in default_params.items():
            if k not in param_map:
                param_map[k] = v

        # partial_sell_tiers => parse JSON if present
        if "partial_sell_tiers" in param_map:
            try:
                param_map["partial_sell_tiers"] = json.loads(param_map["partial_sell_tiers"])
            except:
                param_map["partial_sell_tiers"] = []

        # meltdown_tiers => parse JSON if present
        if "meltdown_tiers" in param_map:
            try:
                param_map["meltdown_tiers"] = json.loads(param_map["meltdown_tiers"])
            except:
                param_map["meltdown_tiers"] = []

        return param_map

    def _place_order_with_min_check(self, exchange, order_type, symbol, amount, price=None, side="buy"):
        exchange.load_markets()
        if symbol not in exchange.markets:
            logging.warning(f"[OrderMinCheck] {symbol} not in {exchange.id} markets => skip.")
            return None

        market_info = exchange.markets[symbol]
        min_amt = market_info.get("limits", {}).get("amount", {}).get("min", 0.0)
        if min_amt is None:
            min_amt = 0.0

        if float(amount) < float(min_amt):
            logging.warning(
                f"[OrderMinCheck] {symbol}: requested {amount} < min {min_amt} => skip."
            )
            return None

        try:
            if order_type == "market":
                if side == "buy":
                    return exchange.create_market_buy_order(symbol, float(amount))
                else:
                    return exchange.create_market_sell_order(symbol, float(amount))
            else:
                if side == "buy":
                    return exchange.create_limit_buy_order(symbol, float(amount), float(price))
                else:
                    return exchange.create_limit_sell_order(symbol, float(amount), float(price))
        except Exception as e:
            logging.warning(f"[OrderMinCheck] {symbol} => {e}")
            return None
