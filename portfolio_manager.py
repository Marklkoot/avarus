import logging
from decimal import Decimal
import datetime
import os
import uuid  # for generating unique fallback IDs

class PortfolioManager:
    def __init__(self, db, exchange, config, stable_override=None):
        """
        :param db: MySQL connection
        :param exchange: ccxt exchange object
        :param config: config dict
        :param stable_override: if set, forces e.g. "EUR" for margin trades
        """
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config

        if stable_override is not None:
            self.stable_symbol = stable_override
        else:
            self.stable_symbol = self.config["portfolio"].get("stable_symbol", "USD")

        self.default_cost_basis = Decimal("0.0")
        self._recent_trades = []

    def initialize_starting_capital(self):
        init_inv = Decimal(str(self.config.get("initial_investment", 5500)))
        q = """SELECT coin, quantity FROM portfolio_positions WHERE coin=%s"""
        self.cursor.execute(q, (self.stable_symbol,))
        row = self.cursor.fetchone()

        if not row:
            logging.info(f"[Portfolio] No '{self.stable_symbol}' => depositing {init_inv}.")
            now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ins = """INSERT INTO portfolio_positions (coin, quantity, cost_basis, last_updated)
                     VALUES (%s, %s, %s, %s)"""
            self.cursor.execute(ins, (
                self.stable_symbol,
                float(init_inv),
                float(self.default_cost_basis),
                now_
            ))
            self.db.commit()
        else:
            qty_found = Decimal(str(row[1]))
            logging.info(f"[Portfolio] stable found => skip deposit. qty={qty_found}")

    def get_portfolio(self):
        pf = {}
        q = """SELECT coin, quantity FROM portfolio_positions"""
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        for r in rows:
            c, q_ = r
            pf[c] = Decimal(str(q_))
        return pf

    def record_trade(self, coin, side, amount, price, fee=0.0,
                     order_id=None, order_obj=None,
                     trade_table="trade_history"):
        """
        Called whenever a real trade is executed. We:
          - Insert into 'trades' (legacy) and also into the specified 'trade_table'
          - Adjust local DB portfolio positions
          - Append the trade to _recent_trades for replication
        """
        if order_id is None:
            order_id = str(uuid.uuid4())

        fill_dt = datetime.datetime.now()
        actual_fee = Decimal(str(fee))

        # If caller provided a CCXT order_obj with more details, parse them
        if order_obj:
            try:
                if "id" in order_obj and order_obj["id"]:
                    order_id = str(order_obj["id"])
                if "timestamp" in order_obj and order_obj["timestamp"]:
                    fill_ms = order_obj["timestamp"]
                    fill_dt = datetime.datetime.utcfromtimestamp(fill_ms / 1000.0)

                # parse fee if present
                if "fee" in order_obj and order_obj["fee"]:
                    fee_data = order_obj["fee"]
                    if fee_data and "cost" in fee_data:
                        actual_fee = Decimal(str(fee_data["cost"]))
                elif "fees" in order_obj and order_obj["fees"]:
                    total_fees = Decimal("0")
                    for fobj in order_obj["fees"]:
                        if "cost" in fobj:
                            total_fees += Decimal(str(fobj["cost"]))
                    actual_fee = total_fees

            except Exception as fe:
                logging.warning(f"[Portfolio] Could not parse precise fees => {fe}")

        logging.info(
            f"[Portfolio] record_trade => {side} {amount} {coin} @ {price}, "
            f"fee={actual_fee}, order_id={order_id}, table={trade_table}"
        )

        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 1) Insert into trades (a "legacy" table)
        ins_legacy = """INSERT INTO trades (coin, side, amount, price, fee, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)"""
        self.cursor.execute(ins_legacy, (coin, side, float(amount), float(price),
                                        float(actual_fee), now_str))
        self.db.commit()

        # 2) Insert into trade_table (which defaults to 'trade_history')
        fill_ts_str = fill_dt.strftime('%Y-%m-%d %H:%M:%S')
        ins_new = f"""INSERT INTO {trade_table}
                      (order_id, coin, side, quantity, price, fee, fill_timestamp, created_at)
                      VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())"""
        self.cursor.execute(ins_new, (
            order_id,
            coin,
            side.lower(),
            float(amount),
            float(price),
            float(actual_fee),
            fill_ts_str
        ))
        self.db.commit()

        # 3) Adjust local DB portfolio
        cost = Decimal(str(price)) * Decimal(str(amount))
        if side.lower() == "buy":
            self._adjust_balance(self.stable_symbol, -(cost + actual_fee))
            self._adjust_balance(coin, Decimal(str(amount)))
            self._update_cost_basis_on_buy(coin, Decimal(str(amount)), Decimal(str(price)))
        elif side.lower() == "sell":
            self._adjust_balance(coin, -Decimal(str(amount)))
            net_gain = cost - actual_fee
            self._adjust_balance(self.stable_symbol, net_gain)

        # 4) Add to _recent_trades for replication or logging
        trade_info = {
            "coin": coin,
            "side": side,
            "amount": str(amount),
            "price": str(price),
            "fee": str(actual_fee),
            "order_id": order_id,
            "timestamp": fill_ts_str
        }
        self._recent_trades.append(trade_info)

    def get_recent_trades(self):
        trades_copy = list(self._recent_trades)
        self._recent_trades = []
        return trades_copy

    def _adjust_balance(self, coin, delta):
        old_qty, old_cb = self._get_balance_and_cb(coin)
        new_qty = old_qty + delta
        if new_qty < Decimal("0"):
            new_qty = Decimal("0")

        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """INSERT INTO portfolio_positions (coin, quantity, cost_basis, last_updated)
               VALUES (%s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE quantity=%s, last_updated=%s
            """
        self.cursor.execute(q, (
            coin,
            float(new_qty),
            float(old_cb),
            now_,
            float(new_qty),
            now_
        ))
        self.db.commit()

    def _get_balance(self, coin):
        q = """SELECT quantity FROM portfolio_positions WHERE coin=%s"""
        self.cursor.execute(q, (coin,))
        row = self.cursor.fetchone()
        if row:
            return Decimal(str(row[0]))
        return Decimal("0")

    def _get_balance_and_cb(self, coin):
        q = """SELECT quantity, cost_basis FROM portfolio_positions WHERE coin=%s"""
        self.cursor.execute(q, (coin,))
        row = self.cursor.fetchone()
        if row:
            qty_ = Decimal(str(row[0]))
            cb_  = Decimal(str(row[1]))
            return qty_, cb_
        return Decimal("0"), self.default_cost_basis

    def _update_cost_basis_on_buy(self, coin, buy_amount, buy_price):
        old_qty, old_cb = self._get_balance_and_cb(coin)
        if old_qty <= Decimal("0"):
            new_cb = buy_price
        else:
            total_cost_before = old_cb * old_qty
            total_new_cost    = buy_price * buy_amount
            new_cb = (total_cost_before + total_new_cost) / old_qty

        current_qty, _ = self._get_balance_and_cb(coin)
        old_qty_before_buy = current_qty - buy_amount
        if old_qty_before_buy < Decimal("0"):
            old_qty_before_buy = Decimal("0")

        if current_qty > Decimal("0"):
            new_cb = (old_cb * old_qty_before_buy + buy_price * buy_amount) / current_qty
        else:
            new_cb = buy_price

        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """UPDATE portfolio_positions 
               SET cost_basis=%s, last_updated=%s 
               WHERE coin=%s
            """
        self.cursor.execute(q, (float(new_cb), now_, coin))
        self.db.commit()

    def sync_with_real_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            kr_map = self.config.get("kraken_map", {})

            coins_found = set()
            for ex_sym, amt in bal['total'].items():
                if amt is None:
                    continue
                coin = self._reverse_map_symbol(ex_sym, kr_map)
                coins_found.add(coin)

                if amt <= 0:
                    q = """INSERT INTO portfolio_positions (coin, quantity, cost_basis, last_updated)
                           VALUES (%s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE quantity=%s, last_updated=%s
                        """
                    self.cursor.execute(q, (
                        coin, 0.0, float(self.default_cost_basis),
                        now_, 0.0, now_
                    ))
                else:
                    q = """INSERT INTO portfolio_positions (coin, quantity, cost_basis, last_updated)
                           VALUES (%s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE quantity=%s, last_updated=%s
                        """
                    self.cursor.execute(q, (
                        coin, float(amt), float(self.default_cost_basis),
                        now_, float(amt), now_
                    ))
            self.db.commit()

            logging.info("[Portfolio] synced real kraken balances => done.")

        except Exception as e:
            logging.warning(f"[Portfolio] sync_with_real_balance => {e}")

    def _reverse_map_symbol(self, ex_sym, kr_map):
        inv = {v: k for k, v in kr_map.items()}
        return inv.get(ex_sym, ex_sym)

    def get_cost_basis(self, coin):
        qty_, cb_ = self._get_balance_and_cb(coin)
        return cb_