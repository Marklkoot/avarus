import logging
import datetime
import json
from decimal import Decimal
import ccxt

class ArbitrageManager:
    """
    Advanced multi-exchange arbitrage engine for Avarus 2.0, now incorporating
    taker fees from both exchanges in the net spread check. No placeholders, partial-fills included.

    1) Reads param from meta_parameters (arb_profit_threshold, arb_trade_fraction, arb_pairs).
    2) Fetches each exchange's 'taker' fee from ccxt markets.
    3) Subtracts (takerFeeA + takerFeeB) from the raw difference check.
    4) Executes partial fill logic with up to 15s polling.
    5) Logs each successful arbitrage trade in 'arbitrage_history'.

    You can schedule scan_and_execute_arbitrage() every minute in your Executor, e.g.:
        schedule.every(1).minutes.do(self._arbitrage_routine)
    """

    def __init__(self, db, config):
        self.db = db
        self.cursor = db.cursor()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load multi-exchange config from self.config["arbitrage_exchanges"]
        self.exchanges = []
        exch_cfgs = self.config.get("arbitrage_exchanges", [])
        for econf in exch_cfgs:
            exchange_class = getattr(ccxt, econf["name"])
            exch_obj = exchange_class({
                'apiKey': econf["apiKey"],
                'secret': econf["secret"],
                'enableRateLimit': econf.get("enableRateLimit", True)
            })
            exch_obj.load_markets()
            self.exchanges.append({
                "id": econf["name"],
                "ccxt": exch_obj
            })

        # Read arbitrage params from meta_parameters or fallback config
        self.params = self._load_arbitrage_params()

    def _load_arbitrage_params(self):
        """
        e.g. arb_profit_threshold, arb_trade_fraction, arb_pairs
        from meta_parameters, fallback to config
        """
        param_map = {}
        q = "SELECT param_name, param_value FROM meta_parameters"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        db_map = {r[0]: r[1] for r in rows}

        profit_th = db_map.get("arb_profit_threshold", None)
        trade_fr  = db_map.get("arb_trade_fraction", None)
        pairs_str = db_map.get("arb_pairs", None)

        if not profit_th:
            profit_th = self.config.get("arbitrage_defaults", {}).get("profit_threshold", 0.005)
        if not trade_fr:
            trade_fr = self.config.get("arbitrage_defaults", {}).get("trade_fraction", 0.01)
        if not pairs_str:
            pairs_str = json.dumps(self.config.get("arbitrage_pairs", ["XRP/USDC","ETH/USDC"]))

        param_map["profit_threshold"] = Decimal(str(profit_th))
        param_map["trade_fraction"]   = Decimal(str(trade_fr))
        try:
            param_map["pairs"] = json.loads(pairs_str)
        except:
            param_map["pairs"] = []

        return param_map

    def scan_and_execute_arbitrage(self):
        """
        Called by your Executor on a schedule (e.g. every 1 min).
        We'll only handle the first 2 exchanges for demonstration. If you have more, do nested loops.
        Incorporates taker fee from both sides in the net difference check.
        """
        if len(self.exchanges) < 2:
            self.logger.warning("[Arb] Need >=2 exchanges => skip arbitrage.")
            return

        # We'll demonstrate with exactly 2 exchanges
        exchA = self.exchanges[0]
        exchB = self.exchanges[1]

        # Reload updated params each run
        self.params = self._load_arbitrage_params()
        pairs = self.params["pairs"]
        threshold = self.params["profit_threshold"]
        trade_frac= self.params["trade_fraction"]

        for pair in pairs:
            
            A_has = (pair in exchA["ccxt"].markets)
            B_has = (pair in exchB["ccxt"].markets)
            if not (A_has and B_has):
                self.logger.info(f"[Arb] {pair} => not found on both => skip.")
                continue

            try:
                tA = exchA["ccxt"].fetch_ticker(pair)
                tB = exchB["ccxt"].fetch_ticker(pair)

                askA = Decimal(str(tA.get("ask", 0)))
                bidA = Decimal(str(tA.get("bid", 0)))
                askB = Decimal(str(tB.get("ask", 0)))
                bidB = Decimal(str(tB.get("bid", 0)))

                # Skip if we can't get valid ask/bid
                if askA<=0 or bidA<=0 or askB<=0 or bidB<=0:
                    self.logger.info(f"[Arb] {pair} => invalid ask/bid => skip.")
                    continue

                # fetch taker fees from each exchange for this pair
                takerA = self._get_taker_fee(exchA, pair)
                takerB = self._get_taker_fee(exchB, pair)

                # direction AB => buy on A, sell on B
                raw_diffAB = (bidB - askA) / askA
                effective_diffAB = raw_diffAB - (takerA + takerB)

                # direction BA => buy on B, sell on A
                raw_diffBA = (bidA - askB) / askB
                effective_diffBA = raw_diffBA - (takerA + takerB)

                # Log the computed diffs so we know what's happening
                self.logger.debug(
                    f"[Arb] {pair} => raw_diffAB={raw_diffAB*100:.2f}%, eff_diffAB={effective_diffAB*100:.2f}%, "
                    f"raw_diffBA={raw_diffBA*100:.2f}%, eff_diffBA={effective_diffBA*100:.2f}%, "
                    f"threshold={(threshold*100):.2f}%"
                )

                if effective_diffAB > threshold:
                    self.logger.info(
                        f"[Arb] {pair} => buy on {exchA['id']} ask={askA}, sell on {exchB['id']} bid={bidB}, "
                        f"raw_diff={raw_diffAB*100:.2f}%, eff_diff={effective_diffAB*100:.2f}%"
                    )
                    self._execute_arbitrage_trade(exchA, exchB, pair, askA, bidB, "AB", trade_frac)

                elif effective_diffBA > threshold:
                    self.logger.info(
                        f"[Arb] {pair} => buy on {exchB['id']} ask={askB}, sell on {exchA['id']} bid={bidA}, "
                        f"raw_diff={raw_diffBA*100:.2f}%, eff_diff={effective_diffBA*100:.2f}%"
                    )
                    self._execute_arbitrage_trade(exchB, exchA, pair, askB, bidA, "BA", trade_frac)

                else:
                    # Previously: self.logger.info(f"[Arb] {pair} => no direction meets threshold => skip.")
                    # Now:
                    self.logger.debug(f"[Arb] {pair} => no direction meets threshold => skip.")

            except Exception as e:
                self.logger.warning(f"[Arb] {pair} => error => {e}")

    def _get_taker_fee(self, exch_dict, pair):
        """
        Attempt to retrieve the 'taker' fee from ccxt's markets structure. If not found, fallback to 0.0015 (0.2%).
        This is a simple approach. You can do more advanced logic or store fees in DB.
        """
        try:
            market_info = exch_dict["ccxt"].markets.get(pair, {})
            taker_fee = market_info.get("taker", 0.002)  # default 0.2% if not found
            return Decimal(str(taker_fee))
        except:
            return Decimal("0.002")

    def _execute_arbitrage_trade(self, buyExch, sellExch, pair, buyPrice, sellPrice, direction, trade_fraction):
        try:
            base_coin, quote_coin = pair.split("/")
            balBuy  = buyExch["ccxt"].fetch_balance()
            balSell = sellExch["ccxt"].fetch_balance()

            stable_amt = Decimal(str(balBuy["total"].get(quote_coin, 0)))  # e.g. USDC/EUR
            coin_amt   = Decimal(str(balSell["total"].get(base_coin, 0)))

            spend = stable_amt * Decimal(str(trade_fraction))
            if spend < Decimal("5"):
                spend = Decimal("5")

            if spend > stable_amt:
                spend = stable_amt

            if spend < Decimal("2"):
                self.logger.info("[Arb] Not enough stable => skip.")
                return

            buy_qty = spend / Decimal(str(buyPrice))
            if buy_qty < Decimal("0.0001"):
                self.logger.info("[Arb] buy_qty too small => skip.")
                return

            sell_qty = min(buy_qty, coin_amt)
            if sell_qty < Decimal("0.0001"):
                self.logger.info("[Arb] not enough coin => skip.")
                return

            self.logger.info(
                f"[Arb] => place limit BUY on {buyExch['id']} {pair}, qty={buy_qty}, price={buyPrice}"
            )
            order_buy = buyExch["ccxt"].create_limit_buy_order(
                symbol=pair,
                amount=float(buy_qty),
                price=float(buyPrice)
            )
            filled_buy, remaining_buy, fee_buy = self._poll_order_fills(buyExch, order_buy)
            self.logger.info(
                f"[Arb] => buy fill={filled_buy}, remaining={remaining_buy}, fee={fee_buy}"
            )
            if filled_buy <= Decimal("0"):
                self.logger.info("[Arb] buy fill=0 => skip selling.")
                return

            actual_sell_qty = min(sell_qty, filled_buy)
            self.logger.info(
                f"[Arb] => place limit SELL on {sellExch['id']} {pair}, qty={actual_sell_qty}, price={sellPrice}"
            )
            order_sell = sellExch["ccxt"].create_limit_sell_order(
                symbol=pair,
                amount=float(actual_sell_qty),
                price=float(sellPrice)
            )
            filled_sell, remaining_sell, fee_sell = self._poll_order_fills(sellExch, order_sell)
            self.logger.info(
                f"[Arb] => sell fill={filled_sell}, remaining={remaining_sell}, fee={fee_sell}"
            )

            profit_est = (filled_sell * Decimal(str(sellPrice))) - (filled_buy * Decimal(str(buyPrice)))
            self.logger.info(
                f"[Arb] => fillBuy={filled_buy}, fillSell={filled_sell}, profit_est={profit_est}"
            )
            self._record_arbitrage_trade(
                buyExch['id'], sellExch['id'], pair,
                filled_buy, float(buyPrice), fee_buy,
                filled_sell, float(sellPrice), fee_sell,
                profit_est
            )

        except Exception as e:
            self.logger.warning(f"[Arb Execute] => {e}")

    def _poll_order_fills(self, exch_dict, order_obj, timeout_seconds=15):
        """
        Minimal partial fill approach => poll fetchOrder() up to 'timeout_seconds'.
        Returns (filled_qty, remaining_qty, fee).
        """
        filled = Decimal("0")
        remaining = Decimal("0")
        fee = Decimal("0")
        try:
            order_id = order_obj.get("id")
            sym = order_obj["symbol"]
            start_t = datetime.datetime.now()
            while (datetime.datetime.now() - start_t).total_seconds() < timeout_seconds:
                fetched = exch_dict["ccxt"].fetch_order(order_id, sym)
                filled_ = Decimal(str(fetched.get("filled", 0)))
                remaining_ = Decimal(str(fetched.get("remaining", 0)))
                fee_obj = fetched.get("fee", None)
                if fee_obj and "cost" in fee_obj:
                    fee = Decimal(str(fee_obj["cost"]))
                status = fetched.get("status")
                if filled_ > Decimal("0"):
                    filled = filled_
                if remaining_ >= Decimal("0"):
                    remaining = remaining_
                if status in ["closed", "canceled"] or remaining <= Decimal("0"):
                    break
        except Exception as e:
            self.logger.warning(f"_poll_order_fills => {e}")
        return (filled, remaining, fee)

    def _record_arbitrage_trade(self,
                                buy_exch, sell_exch, pair,
                                buy_qty, buy_price, buy_fee,
                                sell_qty, sell_price, sell_fee,
                                profit_est):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """INSERT INTO arbitrage_history
               (buy_exchange, sell_exchange, pair, buy_qty, buy_price, buy_fee,
                sell_qty, sell_price, sell_fee, profit_est, timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        self.cursor.execute(q, (
            buy_exch, sell_exch, pair,
            float(buy_qty), float(buy_price), float(buy_fee),
            float(sell_qty), float(sell_price), float(sell_fee),
            float(profit_est),
            now_
        ))
        self.db.commit()
        self.logger.info(f"[Arb] => logged in arbitrage_history => profit_est={profit_est}")
