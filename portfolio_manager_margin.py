import logging
import datetime
import uuid
from decimal import Decimal

class PortfolioManagerMargin:
    def __init__(self, db, exchange, config):
        """
        A margin-specific portfolio manager
        that uses 'portfolio_positions_margin'
        and writes trades to 'trade_history_margin'.
        """
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config

        # Possibly fix stable_symbol to 'EUR' or read from config:
        self.stable_symbol = self.config.get("margin_portfolio", {}).get("stable_symbol", "EUR")

        self.default_cost_basis = Decimal("0.0")
        self._recent_trades = []

    def initialize_starting_capital(self):
        """
        Example: ensures a stable_symbol row with initial deposit if none is found.
        If you have REAL margin, you might skip this or set it to 0.
        """
        init_inv = Decimal(str(self.config.get("initial_investment_margin", 700)))
        q = """SELECT coin, quantity FROM portfolio_positions_margin WHERE coin=%s"""
        self.cursor.execute(q, (self.stable_symbol,))
        row = self.cursor.fetchone()

        if not row:
            logging.info(f"[MarginPM] No '{self.stable_symbol}' => depositing {init_inv}.")
            now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ins = """INSERT INTO portfolio_positions_margin 
                     (coin, quantity, cost_basis, last_updated)
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
            logging.info(f"[MarginPM] stable found => skip deposit. qty={qty_found}")

    def get_portfolio(self):
        """
        Returns a dict of coin -> quantity from portfolio_positions_margin.
        (coin is the base symbol, e.g. "LINK", "AAVE", "BTC", etc.)
        """
        pf = {}
        q = """SELECT coin, quantity FROM portfolio_positions_margin"""
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        for r in rows:
            c, q_ = r
            pf[c] = Decimal(str(q_))
        return pf

    def record_trade(
        self,
        coin,
        side,
        amount,
        price,
        fee=0.0,
        order_id=None,
        order_obj=None,
        trade_table="trade_history_margin"
    ):
        """
        Record a margin trade in DB, adjusting 'portfolio_positions_margin'
        and storing in 'trade_history_margin'.

        If the 'order_obj' has partial fills or a list of trades, we handle them.
        Otherwise, single fill.

        We call this only after a *successful* exchange order, so no forced local resets on failure.
        """
        if order_id is None:
            order_id = str(uuid.uuid4())

        partial_fills = []
        fill_dt = datetime.datetime.now()
        actual_fee = Decimal(str(fee))

        if order_obj:
            try:
                if "id" in order_obj and order_obj["id"]:
                    order_id = str(order_obj["id"])
                if "timestamp" in order_obj and order_obj["timestamp"]:
                    fill_ms = order_obj["timestamp"]
                    fill_dt = datetime.datetime.utcfromtimestamp(fill_ms / 1000.0)

                if "trades" in order_obj and isinstance(order_obj["trades"], list):
                    for t in order_obj["trades"]:
                        pf_side   = t.get("side", side).lower()
                        pf_amount = Decimal(str(t.get("amount", amount)))
                        pf_price  = Decimal(str(t.get("price", price)))
                        pf_fee    = Decimal(str(t.get("fee", fee))) if t.get("fee") else Decimal("0")
                        pf_ts     = fill_dt
                        if "timestamp" in t and t["timestamp"]:
                            pf_ts = datetime.datetime.utcfromtimestamp(t["timestamp"]/1000.0)

                        partial_fills.append({
                            "side": pf_side,
                            "amount": pf_amount,
                            "price": pf_price,
                            "fee": pf_fee,
                            "timestamp": pf_ts
                        })
                else:
                    partial_fills.append({
                        "side": side.lower(),
                        "amount": Decimal(str(amount)),
                        "price": Decimal(str(price)),
                        "fee": Decimal(str(fee)),
                        "timestamp": fill_dt
                    })
            except Exception as e:
                logging.warning(f"[MarginPM] partial fill parse error => {e}")
                partial_fills = [{
                    "side": side.lower(),
                    "amount": Decimal(str(amount)),
                    "price": Decimal(str(price)),
                    "fee": Decimal(str(fee)),
                    "timestamp": fill_dt
                }]
        else:
            partial_fills = [{
                "side": side.lower(),
                "amount": Decimal(str(amount)),
                "price": Decimal(str(price)),
                "fee": Decimal(str(fee)),
                "timestamp": fill_dt
            }]

        for pf in partial_fills:
            self._record_single_fill(
                coin=coin,
                side=pf["side"],
                amount=pf["amount"],
                price=pf["price"],
                fee=pf["fee"],
                fill_dt=pf["timestamp"],
                order_id=order_id,
                trade_table=trade_table
            )

    def _record_single_fill(
        self,
        coin,
        side,
        amount,
        price,
        fee,
        fill_dt,
        order_id,
        trade_table
    ):
        fill_ts_str = fill_dt.strftime('%Y-%m-%d %H:%M:%S')
        actual_fee = Decimal(str(fee))

        logging.info(
            f"[Portfolio] record_trade => {side} {float(amount):.6f} {coin} "
            f"@ {float(price):.5f}, fee={actual_fee}, "
            f"order_id={order_id}, table={trade_table}"
        )

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

        cost = Decimal(str(price)) * Decimal(str(amount))
        if side.lower() == "buy":
            # buy => reduce stable by cost+fee, increase coin by amount
            self._adjust_balance(self.stable_symbol, -(cost + actual_fee))
            self._adjust_balance(coin, amount)
            self._update_cost_basis_on_buy(coin, amount, Decimal(str(price)))

        elif side.lower() == "sell":
            # sell => reduce coin by amount (can go negative => short),
            # stable gets net_gain= cost-fee
            self._adjust_balance(coin, -amount)
            net_gain = cost - actual_fee
            self._adjust_balance(self.stable_symbol, net_gain)
            self._update_cost_basis_on_short(coin, amount, Decimal(str(price)))

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

        # === ADDED START ===
        # If flipping from 0 to a new position => store position open time
        old_qty, _ = self._get_balance_and_cb(coin)
        new_qty = old_qty  # after we've updated in _adjust_balance
        # Actually re-fetch from DB to confirm final new_qty
        final_qty, _ = self._get_balance_and_cb(coin)
        if abs(old_qty) < Decimal("1e-8") and abs(final_qty) > Decimal("1e-8"):
            # we just opened a brand-new position
            self._save_position_open_time(coin, fill_dt)
        # === ADDED END ===


    def get_recent_trades(self):
        """
        Returns a list of recent trades (since the last get).
        Then clears the local buffer.
        """
        trades_copy = list(self._recent_trades)
        self._recent_trades = []
        return trades_copy

    def _adjust_balance(self, coin, delta):
        """
        Adjust local DB balance by 'delta'.
        Negative => short positions.
        """
        old_qty, old_cb = self._get_balance_and_cb(coin)
        new_qty = old_qty + delta

        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """INSERT INTO portfolio_positions_margin 
               (coin, quantity, cost_basis, last_updated)
               VALUES (%s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE 
               quantity=%s, last_updated=%s
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

    def _get_balance_and_cb(self, coin):
        """
        Returns (quantity, cost_basis).
        If not found, returns (0, default_cost_basis).
        """
        q = """SELECT quantity, cost_basis 
               FROM portfolio_positions_margin 
               WHERE coin=%s"""
        self.cursor.execute(q, (coin,))
        row = self.cursor.fetchone()
        if row:
            qty_ = Decimal(str(row[0]))
            cb_  = Decimal(str(row[1]))
            return qty_, cb_
        return Decimal("0"), self.default_cost_basis

    def _update_cost_basis_on_buy(self, coin, buy_amount, buy_price):
        """
        Weighted average cost basis logic for going more long.
        If old_qty <= 0 => treat it as new long => cost basis = buy_price.
        """
        old_qty, old_cb = self._get_balance_and_cb(coin)
        if old_qty <= Decimal("0"):
            new_cb = buy_price
        else:
            total_cost_before = old_cb * old_qty
            total_new_cost    = buy_price * buy_amount
            new_cb = (total_cost_before + total_new_cost) / (old_qty + buy_amount)

        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """UPDATE portfolio_positions_margin
               SET cost_basis=%s, last_updated=%s
               WHERE coin=%s
            """
        self.cursor.execute(q, (float(new_cb), now_, coin))
        self.db.commit()

    def _update_cost_basis_on_short(self, coin, sell_amount, sell_price):
        """
        Weighted average cost basis logic for short positions.
        If old_qty >= 0 => new short => cost_basis = sell_price.
        Else average with existing short basis.
        """
        qty_now, old_cb = self._get_balance_and_cb(coin)
        if qty_now >= Decimal("0"):
            # flipping from long => new short basis
            return

        old_qty, old_cb = self._get_balance_and_cb(coin)
        if old_qty >= Decimal("0"):
            new_cb = sell_price
        else:
            total_short_before = old_cb * (-old_qty)
            total_new_short    = sell_price * sell_amount
            new_cb = (total_short_before + total_new_short) / ((-old_qty) + sell_amount)

        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """UPDATE portfolio_positions_margin
               SET cost_basis=%s, last_updated=%s
               WHERE coin=%s
            """
        self.cursor.execute(q, (float(new_cb), now_, coin))
        self.db.commit()

    def sync_with_real_balance(self):
        """
        Updated approach:
         1) fetch open positions from Kraken => privatePostOpenPositions()
         2) parse raw_pair dynamically => (coin_str, stable)
         3) store coin_str in portfolio_positions_margin
         4) forcibly zero out any coin not returned by Kraken
         5) log PnL
        """
        try:
            now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            raw_positions = self.exchange.privatePostOpenPositions()
            open_positions = raw_positions.get("result", {})

            if not open_positions:
                logging.info("[PortfolioMargin] No open margin positions => all zero or none.")
                # (A) forcibly set quantity=0 for all non-stable coins
                self._force_zero_all_but_stable()
            else:
                kraken_coins = set()
                total_unrealized_pnl = Decimal("0")
                total_equity = Decimal("0")

                for txid, posdata in open_positions.items():
                    raw_pair = posdata["pair"]     # e.g. "XXRPZEUR"
                    side = posdata["type"]         # "buy" or "sell"
                    vol  = Decimal(str(posdata["vol"]))
                    cost = Decimal(str(posdata["cost"]))
                    margin_used = Decimal(str(posdata.get("margin","0")))

                    if vol == 0:
                        continue

                    # Use the NEW dynamic parser
                    base_coin, stable = self._parse_kraken_pair_dynamic(raw_pair)
                    if not base_coin or not stable:
                        logging.warning(f"[PortfolioMargin] Cannot parse raw_pair={raw_pair}")
                        continue

                    kraken_coins.add(base_coin)

                    entry_price = Decimal("0")
                    if vol != 0:
                        entry_price = cost / vol

                    qty_ = vol if side=="buy" else -vol

                    # If you want a debug log:
                    logging.debug(f"[PortfolioMargin] => raw_pair={raw_pair}, base={base_coin}, stable={stable}, qty={qty_}")

                    # fetch current price => e.g. "XRP/EUR"
                    symbol = f"{base_coin}/{stable}"
                    current_price = self._fetch_last_price(symbol)
                    if current_price <= 0:
                        continue

                    # compute unrealPnL
                    unreal_pnl = (current_price - entry_price) * qty_
                    total_unrealized_pnl += unreal_pnl
                    position_equity = margin_used + unreal_pnl
                    total_equity += position_equity

                    logging.debug(
                        f"[MarginPnL] raw={raw_pair}, finalSym={symbol}, coin={base_coin}, "
                        f"qty={qty_}, entry={entry_price:.2f}, now={current_price:.2f}, "
                        f"unrealPnL={unreal_pnl:.2f}, margin={margin_used:.2f}"
                    )

                    # Upsert coin=base_coin
                    q = """INSERT INTO portfolio_positions_margin
                           (coin, quantity, cost_basis, last_updated)
                           VALUES (%s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE
                           quantity=%s, cost_basis=%s, last_updated=%s
                        """
                    self.cursor.execute(q, (
                        base_coin,
                        float(qty_), float(entry_price), now_,
                        float(qty_), float(entry_price), now_
                    ))
                self.db.commit()

                logging.info(f"[MarginPnL] TOTAL UnrlPnl => {total_unrealized_pnl:.2f} {self.stable_symbol}, eq={total_equity:.2f}")

                # (B) forcibly zero out any coin not in kraken_coins
                self._force_zero_unlisted(kraken_coins)

            logging.info("[PortfolioMargin] sync real margin balances => done.")

        except Exception as e:
            logging.warning(f"[PortfolioMargin] sync_with_real_balance => {e}")

    def _force_zero_all_but_stable(self):
        """
        Sets quantity=0 for every coin except the stable_symbol.
        Handy when Kraken returns no open positions at all.
        """
        q = """UPDATE portfolio_positions_margin
               SET quantity=0
               WHERE coin <> %s"""
        self.cursor.execute(q, (self.stable_symbol,))
        self.db.commit()
        logging.info("[PortfolioMargin] => forcibly zeroed all positions except stable.")


    def _force_zero_unlisted(self, kraken_coins):
        """
        Forcibly zero out any coin that Kraken did NOT report as an open position.
        That prevents leftover 'ghost' positions in the local DB.
        """
        q = """SELECT coin, quantity FROM portfolio_positions_margin"""
        self.cursor.execute(q)
        rows = self.cursor.fetchall()

        for c, qty in rows:
            if c == self.stable_symbol:
                continue
            if c not in kraken_coins and abs(qty) > Decimal("0"):
                logging.info(f"[PortfolioMargin] => forcibly zeroing coin={c}, was qty={qty}")
                uq = """UPDATE portfolio_positions_margin
                        SET quantity=0
                        WHERE coin=%s"""
                self.cursor.execute(uq, (c,))
        self.db.commit()
        logging.info("[PortfolioMargin] => forcibly zeroed unlisted coins.")


    def _parse_kraken_pair_dynamic(self, raw_pair):
        """
        Dynamically parse raw Kraken pairs like 'XXRPZEUR' => ('XRP','EUR'),
        'XETHZEUR' => ('ETH','EUR'), 'LINKEUR' => ('LINK','EUR'), etc.
        Then confirm the result is recognized in CCXT's loaded markets.
        Returns (base, quote) or (None, None) if we can't parse or find a match.
        """

        if "/" in raw_pair:
            # If raw_pair already has '/', we assume it's 'BASE/QUOTE'.
            # e.g. "BTC/EUR"
            base_, quote_ = raw_pair.split("/")
            return base_, quote_

        rp_up = raw_pair.upper()
        possible_quotes = ["EUR","USD","GBP","USDT","USDC"]
        quote = None
        base_part = None

        # 1) Identify the last 3-4 chars as the quote if it matches
        for q in possible_quotes:
            if rp_up.endswith(q):
                quote = q
                base_part = rp_up[:-len(q)]  # everything before that
                break

        if not quote:
            # fallback: we can't parse => skip
            return (None, None)

        # 2) Remove leading/trailing 'X' or 'Z' if present
        #    e.g. 'XXRPZ' => 'XRP'
        while base_part.startswith("X"):
            base_part = base_part[1:]
        while base_part.endswith("Z"):
            base_part = base_part[:-1]

        base_ = base_part
        if not base_:
            return (None, None)

        # 3) Build a CCXT symbol => 'XRP/EUR'
        possible_symbol = f"{base_}/{quote}"

        # 4) Check if that symbol is recognized by CCXT
        self.exchange.load_markets()
        if possible_symbol in self.exchange.markets:
            return (base_, quote)
        else:
            # fallback if no recognized market
            return (None, None)


    def _fetch_last_price(self, symbol):
        """
        Simplistic approach to get last price from exchange.
        e.g. symbol => "LINK/EUR"
        """
        try:
            t = self.exchange.fetch_ticker(symbol)
            last_ = t.get('last', t.get('close', 0.0))
            return Decimal(str(last_))
        except:
            return Decimal("0")


    # === ADDED START ===
    def _save_position_open_time(self, coin, dt_obj):
        """
        Store a 'position_open_time_{coin}' param in meta_parameters_margin. 
        """
        param_name = f"position_open_time_{coin}"
        dt_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
               VALUES (%s, %s, %s)
        """
        self.cursor.execute(q, (param_name, dt_str, now_))
        self.db.commit()
    # === ADDED END ===
