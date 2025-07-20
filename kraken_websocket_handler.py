import logging
import json
import threading
import websocket
import time
import ccxt
import requests
import hashlib
import hmac
import base64
import urllib.parse
import datetime

class KrakenWebsocketHandler:
    def __init__(self, exchange, db, config, portfolio_mgr):
        """
        exchange: ccxt exchange instance
        db: MySQL or your ledger
        config:
        portfolio_mgr: so we can call record_trade(...) on partial fills
        """
        self.exchange = exchange
        self.db = db
        self.config = config
        self.portfolio_mgr = portfolio_mgr

        self.kraken_ws_url = "wss://ws-auth.kraken.com/"
        self._kraken_ws_token = None
        self.ws_thread = None
        self.ws_app = None
        self.running = False

    def start(self):
        # 1) fetch WS token
        self._kraken_ws_token = self._fetch_kraken_token()

        # 2) connect
        self.ws_app = websocket.WebSocketApp(
            self.kraken_ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever)
        self.running = True
        self.ws_thread.start()

    def _fetch_kraken_token(self):
        logging.info("[WS] Fetching a REAL Kraken WS token...")

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
            token = j["result"]["token"]
            logging.info(f"[WS] Got real Kraken WS token = {token[:8]}... (redacted)")
            return token
        else:
            raise Exception(f"[WS] Token fetch fail => {resp.status_code} => {resp.text}")

    def _on_open(self, ws):
        logging.info("[WS] Connected => Subscribing to ownTrades with token")
        sub_msg = {
            "event": "subscribe",
            "subscription": {
                "name": "ownTrades",
                "token": self._kraken_ws_token
            }
        }
        ws.send(json.dumps(sub_msg))

        # Optionally, also subscribe to openOrders if you want more detail:
        # sub_msg_orders = {
        #     "event": "subscribe",
        #     "subscription": {
        #         "name": "openOrders",
        #         "token": self._kraken_ws_token
        #     }
        # }
        # ws.send(json.dumps(sub_msg_orders))

    def _on_message(self, ws, message):
        # For debugging all raw messages
        try:
            data = json.loads(message)
            if isinstance(data, dict):
                event = data.get("event")
                if event == "heartbeat":
                    return
                elif event == "subscriptionStatus":
                    status = data.get("status")
                    logging.info(f"[WS] subscriptionStatus => {status}, msg={data}")
                else:
                    logging.debug(f"[WS] dict => {data}")

            elif isinstance(data, list):
                # Usually => [ [ { txid: {...}} ], chanID, "ownTrades" ]
                if len(data) >= 3 and data[-1] == "ownTrades":
                    own_trades_arr = data[0]
                    for tx_info in own_trades_arr:
                        for txid, fill_data in tx_info.items():
                            self._handle_own_trade(txid, fill_data)
                else:
                    logging.debug(f"[WS] array => {data}")
            else:
                logging.debug(f"[WS] unhandled => {data}")
        except Exception as e:
            logging.error(f"[WS] on_message => {e}", exc_info=True)

    def _handle_own_trade(self, txid, fill_data):
        """
        Example fill_data might look like:
        {
          "ordertxid":"O5MDTG-3P6CV-HVNVQJ",
          "pair":"XBT/USD",
          "time":1687251577.456,
          "type":"sell",
          "ordertype":"market",
          "price":"96849.9",
          "cost":"110.96093",
          "fee":"0.44384",
          "vol":"0.0011457",
          "margin":"0.00000",
          "misc":"",
          "posstatus":"",
          ...
        }
        We parse the relevant fields and pass them to record_trade.
        """
        side = fill_data.get("type","")  # "buy" or "sell"
        vol_str  = fill_data.get("vol","0")
        vol = float(vol_str)
        cost_str = fill_data.get("cost","0")
        cost = float(cost_str)
        fee_str  = fill_data.get("fee","0")
        fee  = float(fee_str)
        pair = fill_data.get("pair","XBT/USD")

        # If we have a "time" field, parse it as UTC seconds
        fill_time = fill_data.get("time", None)
        fill_dt = None
        if fill_time:
            # "time" is float with Unix epoch
            fill_dt = datetime.datetime.utcfromtimestamp(float(fill_time))

        # For price
        if vol > 0:
            price = cost / vol
        else:
            price = float(fill_data.get("price", 0.0))

        # Convert pair to a coin symbol
        base_quote = pair.split("/")
        base = base_quote[0] if len(base_quote) > 0 else "XBT"

        # Attempt to invert from kraken_map
        kr_map = self.config.get("kraken_map", {})
        inv_map = {v: k for k, v in kr_map.items()}
        coin = inv_map.get(base, base)  # fallback to base if not found

        # Instead of a random ID, let's parse "ordertxid" as the real order ID
        # That helps us avoid "unknown" or random UUID collisions
        order_id = fill_data.get("ordertxid", txid)

        logging.info(f"[WS] ownTrade => txid={txid}, ordertxid={order_id}, side={side}, vol={vol} {coin} @ {price}, fee={fee}")

        # Build a minimal 'order_obj' so portfolio_manager can store the real order_id + timestamp
        # If we do this, it won't fallback to random uuid
        order_obj = {
            "id": order_id,
            "timestamp": None,  # We'll set actual fill time in ms if we want
            "fee": {
                "cost": fee
            }
        }
        if fill_dt:
            # fill_dt is a datetime => convert to ms
            ms_ = int(fill_dt.timestamp() * 1000)
            order_obj["timestamp"] = ms_

        # Now record the trade in both 'trades' and 'trade_history'
        self.portfolio_mgr.record_trade(
            coin=coin,
            side=side,
            amount=vol,
            price=price,
            fee=fee,
            order_obj=order_obj
        )

    def _on_error(self, ws, error):
        logging.error(f"[WS] error => {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"[WS] Closed => code={close_status_code}, msg={close_msg}")
        self.running = False

    def stop(self):
        if self.ws_app:
            logging.info("[WS] Stopping WebSocket.")
            self.ws_app.close()
        if self.ws_thread:
            self.ws_thread.join()
            self.ws_thread = None
