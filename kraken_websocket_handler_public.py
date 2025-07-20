# kraken_websocket_handler_public.py

import logging
import json
import threading
import websocket
import time
import datetime
from decimal import Decimal

class KrakenWebsocketHandlerPublic:
    """
    Connects to Kraken's public WS => wss://ws.kraken.com
    Subscribes to 'ticker' for real-time price updates.
    Supports dynamic subscribe/unsubscribe for various pairs.

    Now includes real-time partial-sell logic in _on_message:
    whenever a coin's price crosses a partial-sell threshold, we immediately
    place a partial sell order (using your margin approach).

    If the connection is lost, we automatically reconnect and re-subscribe
    to the pairs in self.current_pairs.
    """

    def __init__(self, db, config, portfolio_mgr, exchange):
        self.db = db
        self.config = config
        self.portfolio_mgr = portfolio_mgr
        self.exchange = exchange

        self.ws_url = "wss://ws.kraken.com"  # PUBLIC endpoint

        self.ws_app = None
        self.ws_thread = None
        self.running = False

        # Keep track of which pairs are currently subscribed
        self.current_pairs = set()

        # Define your partial-sell tiers for real-time approach:
        # At +1%, sell 30%, +2% => sell 30%, +5% => sell 40%.
        # Adjust to your liking.
        self.partial_sell_tiers = [
            (Decimal("0.01"), Decimal("0.30")),  # e.g. at +1%, sell 30%
            (Decimal("0.02"), Decimal("0.30")),  # +2%
            (Decimal("0.05"), Decimal("0.40"))   # +5%
        ]

    def start(self):
        """
        We run a loop that connects via ws_app.run_forever().
        If the connection is lost, we sleep 5s and try again,
        until self.running is set to False by stop().
        """
        logging.info("[WS-PUBLIC] Starting public WebSocket => wss://ws.kraken.com")
        self.running = True

        def _run():
            while self.running:
                self.ws_app = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.ws_app.run_forever()

                if self.running:
                    logging.warning("[WS-PUBLIC] Disconnected => Reconnect in 5s")
                    time.sleep(5)

        self.ws_thread = threading.Thread(target=_run, daemon=True)
        self.ws_thread.start()

    def _on_open(self, ws):
        logging.info("[WS-PUBLIC] Connected to public endpoint.")
        # If we already had pairs in self.current_pairs, let's re-subscribe them:
        if self.current_pairs:
            logging.info("[WS-PUBLIC] Re-subscribing to existing pairs after reconnect.")
            for pair_str in list(self.current_pairs):
                self.subscribe_pair(pair_str)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            # For ticker => data is: [channelID, { TICKER_INFO }, "ticker", "XBT/USD"]

            if isinstance(data, list):
                if len(data) >= 4 and data[-2] == "ticker":
                    pair_str = data[-1]  # e.g. "XBT/USD"
                    ticker_info = data[1]
                    if "c" in ticker_info:
                        last_price_str = ticker_info["c"][0]
                        last_price = Decimal(last_price_str)
                        # Convert pair => coin
                        coin = self._map_pair_to_coin(pair_str)
                        if coin:
                            # >>> REAL-TIME PARTIAL-TAKE <<<
                            self._check_realtime_partial_take(coin, last_price)
                else:
                    logging.debug(f"[WS-PUBLIC] unhandled array => {data}")
            elif isinstance(data, dict):
                event = data.get("event")
                if event == "heartbeat":
                    return
                elif event == "systemStatus":
                    logging.info(f"[WS-PUBLIC] systemStatus => {data}")
                elif event == "subscriptionStatus":
                    status = data.get("status")
                    pair_ = data.get("pair","?")
                    logging.info(f"[WS-PUBLIC] subscriptionStatus => {status}, pair={pair_}, msg={data}")
                else:
                    logging.debug(f"[WS-PUBLIC] dict => {data}")
            else:
                logging.debug(f"[WS-PUBLIC] unhandled => {data}")

        except Exception as e:
            logging.error(f"[WS-PUBLIC] on_message => {e}", exc_info=True)

    def _on_error(self, ws, error):
        logging.error(f"[WS-PUBLIC] error => {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"[WS-PUBLIC] Closed => code={close_status_code}, msg={close_msg}")

    def stop(self):
        """
        Gracefully stops the auto-reconnect loop and closes any existing ws_app.
        """
        logging.info("[WS-PUBLIC] Stopping public WebSocket auto-reconnect.")
        self.running = False
        if self.ws_app:
            self.ws_app.close()
        if self.ws_thread:
            self.ws_thread.join()
            self.ws_thread = None

    # ====================================
    # REAL-TIME PARTIAL-SELL IMPLEMENTATION
    # ====================================
    def _check_realtime_partial_take(self, coin, current_price):
        """
        For real-time partial profit taking:
        1) Get net position (qty) and cost_basis from portfolio manager
        2) If net LONG and current_price > cost_basis*(1+pct), place partial sells
        3) Mark tiers as 'sold' in DB to avoid re-selling the same tier repeatedly.
        """
        # 1) net position
        pf = self.portfolio_mgr.get_portfolio()  # coin->qty
        qty = pf.get(coin, Decimal("0"))
        if qty <= 0:
            return  # not net long => skip partial-sell logic (if short or 0)

        # 2) cost basis
        old_qty, cost_basis = self.portfolio_mgr._get_balance_and_cb(coin)
        if cost_basis <= 0:
            return  # can't compute partial sell if cost_basis=0 or negative

        # 3) Check partial-sell tiers
        sold_tiers = self._load_partial_sell_status(coin)  # dict => e.g. {"tier1":True}
        for i, (pct, fraction) in enumerate(self.partial_sell_tiers, start=1):
            tier_key = f"tier{i}"
            if sold_tiers.get(tier_key, False):
                continue  # already sold that tier

            trigger_price = cost_basis * (Decimal("1") + pct)
            if current_price >= trigger_price:
                # partial-sell fraction of net position
                to_sell = qty * fraction
                logging.info(f"[RT-PartialTake] => {coin} at +{pct*100:.1f}% => SELL {fraction*100:.1f}% of {qty} (={float(to_sell):.4f})")
                self._place_partial_sell(coin, to_sell)

                # mark tier as sold
                sold_tiers[tier_key] = True
                self._save_partial_sell_status(coin, sold_tiers)

                # Optionally update local qty so multiple tiers can trigger in same price update
                qty -= to_sell
                if qty <= 0:
                    break

    def _place_partial_sell(self, coin, amount_to_sell):
        """
        Actually place a market SELL order for 'amount_to_sell' of 'coin' on margin
        using your existing approach. Then record the trade in DB.
        """
        symbol = self._map_coin_to_pair(coin)  # e.g. "XBT/EUR"
        if not symbol:
            logging.warning(f"[RT-PartialTake] => can't map coin={coin}")
            return

        # place a market SELL with leverage=3 (example)
        try:
            order = self.exchange.create_market_sell_order(symbol, float(amount_to_sell), params={"leverage": 3})
            # fetch fill price for approximate
            fill_price = self._fetch_last_price(symbol)
            fee = fill_price * amount_to_sell * Decimal("0.0035")  # approximate fee
            self.portfolio_mgr.record_trade(
                coin=coin,
                side="sell",
                amount=float(amount_to_sell),
                price=float(fill_price),
                fee=float(fee),
                order_obj=order
            )
            logging.info(f"[RT-PartialTake] => SELL {amount_to_sell:.4f} {coin} @ ~{fill_price} => partial exit.")
        except Exception as e:
            logging.error(f"[RT-PartialTake] => Failed to SELL {amount_to_sell:.4f} {coin} => {e}")

    def _fetch_last_price(self, pair_str):
        """
        Quick helper to fetch ticker for 'XBT/EUR' from ccxt.
        """
        try:
            t = self.exchange.fetch_ticker(pair_str)
            return Decimal(str(t.get('last', t.get('close', 0.0))))
        except:
            return Decimal("0")

    def _load_partial_sell_status(self, coin):
        """
        Loads sold tiers from DB (meta_parameters_margin or equivalent).
        e.g. param_name='partial_sell_status_{coin}' => JSON dict => {"tier1":True,"tier2":False}
        """
        param_name = f"partial_sell_status_{coin}"
        query = "SELECT param_value FROM meta_parameters_margin WHERE param_name=%s"
        cur = self.db.cursor(dictionary=True)
        cur.execute(query, (param_name,))
        row = cur.fetchone()
        if not row:
            return {}
        import json
        try:
            return json.loads(row["param_value"])
        except:
            return {}

    def _save_partial_sell_status(self, coin, status_dict):
        """
        Saves sold-tier dictionary back to DB.
        """
        import json
        param_name = f"partial_sell_status_{coin}"
        val_str = json.dumps(status_dict)
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
                   VALUES (%s, %s, %s)
                """
        cur = self.db.cursor()
        cur.execute(query, (param_name, val_str, now_))
        self.db.commit()

    # ============================
    # SUBSCRIBE/UNSUBSCRIBE
    # ============================
    def subscribe_pair(self, pair_str):
        if pair_str not in self.current_pairs:
            self.current_pairs.add(pair_str)
            if self.ws_app:
                msg = {
                    "event": "subscribe",
                    "pair": [pair_str],
                    "subscription": {"name":"ticker"}
                }
                try:
                    self.ws_app.send(json.dumps(msg))
                    logging.info(f"[WS-PUBLIC] Subscribing to pair={pair_str}")
                except Exception as e:
                    logging.error(f"[WS-PUBLIC] subscribe_pair => {e}")
            else:
                logging.info(f"[WS-PUBLIC] subscribe_pair => No ws_app yet, stored pair={pair_str}")

    def unsubscribe_pair(self, pair_str):
        if pair_str in self.current_pairs:
            self.current_pairs.remove(pair_str)
            if self.ws_app:
                msg = {
                    "event":"unsubscribe",
                    "pair":[pair_str],
                    "subscription":{"name":"ticker"}
                }
                try:
                    self.ws_app.send(json.dumps(msg))
                    logging.info(f"[WS-PUBLIC] Unsubscribing pair={pair_str}")
                except Exception as e:
                    logging.error(f"[WS-PUBLIC] unsubscribe_pair => {e}")

    def subscribe_new_coin(self, coin):
        pair_str = self._map_coin_to_pair(coin)
        if pair_str:
            self.subscribe_pair(pair_str)

    def unsubscribe_coin(self, coin):
        pair_str = self._map_coin_to_pair(coin)
        if pair_str:
            self.unsubscribe_pair(pair_str)

    def _map_coin_to_pair(self, coin):
        kr_map = self.config.get("kraken_map", {})
        base = kr_map.get(coin, coin)
        stable = self.config.get("margin_portfolio",{}).get("stable_symbol","USD")
        return f"{base}/{stable}"

    def _map_pair_to_coin(self, pair_str):
        splitted = pair_str.split("/")
        if len(splitted)<2:
            return None
        base = splitted[0]
        inv_map = {}
        for ckey, cval in self.config.get("kraken_map", {}).items():
            inv_map[cval] = ckey
        if base in inv_map:
            return inv_map[base]
        if base == "XBT":
            return "BTC"
        return base
