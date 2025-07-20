# kraken_websocket_handler_margin.py

import logging
import json
import threading
import websocket
import time
import requests
import hashlib
import hmac
import base64
import urllib.parse
import datetime
from decimal import Decimal

class KrakenWebsocketHandlerMargin:
    def __init__(self, exchange, db, config, portfolio_mgr):
        """
        exchange: ccxt exchange instance (your margin exchange)
        db: Database connection
        config: the margin config (from configmargin.yaml)
        portfolio_mgr: instance of PortfolioManagerMargin
        """
        self.exchange = exchange
        self.db = db
        self.config = config
        self.portfolio_mgr = portfolio_mgr

        # This is the PRIVATE endpoint for "ownTrades", "openOrders", etc.
        self.kraken_ws_url = "wss://ws-auth.kraken.com/"
        self._kraken_ws_token = None
        self.ws_thread = None
        self.ws_app = None
        self.running = False

    def start(self):
        """
        Fetch WS token, then connect and run in a separate thread.
        """
        self._kraken_ws_token = self._fetch_kraken_token()  # must be a string
        logging.debug(f"[WS-MARGIN] token type => {type(self._kraken_ws_token)}")

        self.ws_app = websocket.WebSocketApp(
            self.kraken_ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.running = True
        self.ws_thread.start()

    def _fetch_kraken_token(self):
        """
        Calls GET https://api.kraken.com/0/private/GetWebSocketsToken
        Returns a string token or raises an exception if not successful.
        """
        logging.info("[WS-MARGIN] Fetching Kraken WS token for margin account...")

        api_key    = self.config["exchange"]["apiKey"]
        api_secret = self.config["exchange"]["secret"]

        nonce = str(int(time.time() * 1000))
        path = "/0/private/GetWebSocketsToken"
        url  = "https://api.kraken.com" + path
        postdata_str = urllib.parse.urlencode({"nonce": nonce})
        encoded = (nonce + postdata_str).encode("utf-8")
        sha256  = hashlib.sha256(encoded).digest()
        hmac_key  = base64.b64decode(api_secret)
        to_sign   = path.encode('utf-8') + sha256
        signature = hmac.new(hmac_key, to_sign, hashlib.sha512)
        sigdigest = base64.b64encode(signature.digest()).decode()

        headers = {
            "API-Key": api_key,
            "API-Sign": sigdigest,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        resp = requests.post(url, data={"nonce": nonce}, headers=headers, timeout=10)

        if resp.status_code == 200:
            j = resp.json()
            err_list = j.get("error", [])
            if err_list:
                raise Exception(f"Kraken returned error => {err_list}")
            # Must be a string:
            token = j["result"]["token"]
            logging.info(f"[WS-MARGIN] Got Kraken WS token = {token[:8]}... (redacted)")
            return token  # e.g. "T-vQ7..."

        else:
            raise Exception(f"[WS-MARGIN] Token fetch fail => {resp.status_code} => {resp.text}")

    def _on_open(self, ws):
        """
        Once connected, subscribe to 'ownTrades' using the string token.
        """
        logging.info("[WS-MARGIN] Connected => Subscribing to ownTrades with token")

        sub_msg = {
            "event": "subscribe",
            "subscription": {
                "name": "ownTrades",
                "token": self._kraken_ws_token  # must be str
            }
        }
        ws.send(json.dumps(sub_msg))

    def _on_message(self, ws, message):
        """
        Handler for incoming messages.
        Distinguish between dict (events) and list (ownTrades messages).
        """
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                event = data.get("event")
                if event == "heartbeat":
                    return
                elif event == "subscriptionStatus":
                    status = data.get("status")
                    logging.info(f"[WS-MARGIN] subscriptionStatus => {status}, msg={data}")
                else:
                    logging.debug(f"[WS-MARGIN] dict => {data}")

            elif isinstance(data, list):
                # If it's ownTrades => parse fill data
                if len(data) >= 3 and data[-1] == "ownTrades":
                    own_trades_arr = data[0]
                    for tx_info in own_trades_arr:
                        for txid, fill_info in tx_info.items():
                            self._handle_own_trade(txid, fill_info)
                else:
                    logging.debug(f"[WS-MARGIN] array => {data}")
            else:
                logging.debug(f"[WS-MARGIN] unhandled => {data}")

        except Exception as e:
            logging.error(f"[WS-MARGIN] on_message => {e}", exc_info=True)

    def _handle_own_trade(self, txid, fill_data):
        """
        E.g. fill_data might look like:
        {
          "ordertxid":"O5MDTG-3P6CV-HVNVQJ",
          "pair":"XBT/USD",
          "time":1687251577.456,
          "type":"sell",
          "ordertype":"market",
          "price":"26849.9",
          "cost":"110.96093",
          "fee":"0.44384",
          "vol":"0.0011457",
          "margin":"0.00000",
          ...
        }
        We'll parse it and pass to portfolio_mgr.record_trade(...) so your local DB updates.
        """
        side = fill_data.get("type", "")  # "buy" or "sell"
        vol_str  = fill_data.get("vol","0")
        vol      = float(vol_str)
        cost_str = fill_data.get("cost","0")
        cost     = float(cost_str)
        fee_str  = fill_data.get("fee","0")
        fee      = float(fee_str)
        pair     = fill_data.get("pair","XBT/USD")

        fill_time = fill_data.get("time", None)
        fill_dt = None
        if fill_time:
            fill_dt = datetime.datetime.utcfromtimestamp(float(fill_time))

        # If volume > 0, price can be cost / vol
        if vol > 0:
            price = cost / vol if vol else 0.0
        else:
            # fallback if vol=0 => parse "price"
            price = float(fill_data.get("price", 0.0))

        coin = self._map_pair_to_coin(pair)

        order_id = fill_data.get("ordertxid", txid)
        logging.info(
            f"[WS-MARGIN] ownTrade => txid={txid}, ordertxid={order_id}, "
            f"side={side}, vol={vol} {coin} @ {price}, fee={fee}"
        )

        # Build order_obj for portfolio_mgr
        order_obj = {
            "id": order_id,
            "timestamp": None,
            "fee": {"cost": fee}
        }
        if fill_dt:
            ms_ = int(fill_dt.timestamp() * 1000)
            order_obj["timestamp"] = ms_

        # Record trade in DB
        self.portfolio_mgr.record_trade(
            coin=coin,
            side=side,
            amount=vol,
            price=price,
            fee=fee,
            order_obj=order_obj
        )

    def _on_error(self, ws, error):
        logging.error(f"[WS-MARGIN] error => {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"[WS-MARGIN] Closed => code={close_status_code}, msg={close_msg}")
        self.running = False

    def stop(self):
        if self.ws_app:
            logging.info("[WS-MARGIN] Stopping WebSocket.")
            self.ws_app.close()
        if self.ws_thread:
            self.ws_thread.join()
            self.ws_thread = None
        self.running = False

    # ----------------------------------------------------------------------
    # Additional helper method to map "XBT/USD" => "BTC" from your config
    # If your config has e.g. "kraken_map": {"BTC":"XBT","ETH":"XETH","SOL":"SOL"}
    # or similar.
    # ----------------------------------------------------------------------
    def _map_pair_to_coin(self, pair_str):
        """
        Convert something like "XBT/USD" => "BTC" if "XBT" is in your config's kraken_map.
        """
        splitted = pair_str.split("/")
        base = splitted[0] if splitted else "XBT"

        inv_map = {}
        if "kraken_map" in self.config:
            # invert the map => e.g. {"XBT":"BTC","XETH":"ETH", ...}
            for c_key, c_val in self.config["kraken_map"].items():
                # c_key is e.g. "BTC", c_val is e.g. "XBT"
                inv_map[c_val] = c_key

        if base in inv_map:
            return inv_map[base]  # e.g. "BTC"
        if base == "XBT":
            return "BTC"
        return base
