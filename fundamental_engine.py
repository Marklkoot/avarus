import logging
import requests
import datetime
from decimal import Decimal

class FundamentalEngine:
    def __init__(self, db, exchange, config):
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config

        self.cmc_api_key = self.config.get("fundamental", {}).get("coinmarketcap_api", "")
        self.stable_symbol = self.config["portfolio"].get("stable_symbol", "USD")

    def scan_new_coins(self):
        logging.info("[FundEngine] scan_new_coins => start")

        top_coins = self._fetch_top_coins_cmc(limit=50)
        if not top_coins:
            logging.info("[FundEngine] no top coins => skip scanning.")
            return

        self.exchange.load_markets()

        # Define the set of stable/base coins to skip
        skip_stables = {"USDT", "USDC", "DAI", "XMR"}

        for cinfo in top_coins:
            symbol = cinfo["symbol"].upper()
            mc     = cinfo["market_cap"]

            # --- Skip if stable coin ---
            if symbol in skip_stables:
                logging.info(f"[FundEngine] skip stable => {symbol}")
                continue
            # ---------------------------

            found_symbol = find_kraken_symbol_for_coin(self.exchange, symbol, [self.stable_symbol])
            if found_symbol is not None:
                logging.info(f"[FundEngine] [DYNAMIC] {symbol} => found => {found_symbol}")
            else:
                logging.info(f"[FundEngine] [DYNAMIC] {symbol} => not found => skip (dynamic)")
                continue

            if mc > 250_000_000:
                if self._kraken_has_pair(symbol):
                    self._maybe_add_coin(symbol)
                else:
                    logging.info(f"[FundEngine] {symbol} => not found on Kraken => skip.")
            else:
                logging.info(f"[FundEngine] {symbol} => mc={mc} < threshold => skip.")

        logging.info("[FundEngine] scan_new_coins => done.")

    def _kraken_has_pair(self, coin):
        preferred_quotes = [self.stable_symbol]
        symbol_str = find_kraken_symbol_for_coin(
            kraken=self.exchange,
            coin=coin,
            preferred_quotes=preferred_quotes
        )
        if symbol_str is not None:
            logging.info(f"[FundEngine] {coin} => found => {symbol_str}")
            return True
        else:
            logging.info(f"[FundEngine] {coin} => not found on Kraken => skip.")
            return False

    def _maybe_add_coin(self, symbol):
        q_check = """SELECT coin FROM portfolio_positions WHERE coin=%s"""
        self.cursor.execute(q_check, (symbol,))
        row = self.cursor.fetchone()
        if row:
            return

        logging.info(f"[FundEngine] Adding coin {symbol} with 0 quantity.")
        q_ins = """INSERT INTO portfolio_positions (coin, quantity, cost_basis, last_updated)
                   VALUES (%s, 0, 1.0, NOW())"""
        self.cursor.execute(q_ins, (symbol,))
        self.db.commit()

    def _fetch_top_coins_cmc(self, limit=40):
        if not self.cmc_api_key:
            logging.info("[scan_new_coins] no cmc_api => skip.")
            return []

        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": self.cmc_api_key
        }
        params = {
            "limit": limit,
            "convert": "USD",
            "sort": "market_cap",
            "sort_dir": "desc"
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                out_list = []
                for c in data:
                    sym = c["symbol"].upper()
                    mc  = c["quote"]["USD"]["market_cap"]
                    vol = c["quote"]["USD"]["volume_24h"]
                    out_list.append({
                        "symbol": sym,
                        "market_cap": mc,
                        "volume_24h": vol
                    })
                return out_list
            else:
                logging.warning(f"[scan_new_coins] listings => {resp.status_code} => {resp.text}")
                return []
        except Exception as e:
            logging.warning(f"[scan_new_coins] => {e}")
            return []

    def update_fundamentals(self):
        if self._already_updated_today():
            logging.info("[FundEngine] Already updated fundamentals => skip.")
            return

        coins = self._get_portfolio_coins()
        if not coins:
            logging.info("[FundEngine] No coins => skip fundamentals.")
            return

        cmc_data = self._batch_fetch_coinmarketcap(coins)
        for coin in coins:
            data_for_coin = cmc_data.get(coin.upper(), None)
            if data_for_coin:
                self._store_cmc_fundamentals(coin, data_for_coin)
            else:
                self._fallback_kraken(coin)

        self._compute_scores(coins)
        self._mark_updated_today()
        logging.info("[FundEngine] update_fundamentals => done.")

    def _already_updated_today(self):
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        q = """SELECT param_value FROM meta_parameters WHERE param_name='fundamentals_updated_on'"""
        self.cursor.execute(q)
        row = self.cursor.fetchone()
        if not row:
            return False
        return (row[0]== now_date)

    def _mark_updated_today(self):
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
               VALUES ('fundamentals_updated_on', %s, %s)"""
        self.cursor.execute(q, (now_date, now_))
        self.db.commit()

    def _get_portfolio_coins(self):
        q = """SELECT coin FROM portfolio_positions"""
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        stable = self.stable_symbol
        coins = []
        for r in rows:
            c= r[0]
            if c != stable:
                coins.append(c)
        return coins

    # PATCH #1: chunk requests to avoid 413 errors
    def _batch_fetch_coinmarketcap(self, coins):
        if not self.cmc_api_key:
            logging.info("[FundEngine] no coinmarketcap_api => skip cmc fetch.")
            return {}

        # remove stable or base coins we don't want from the query (like USDC, USDT, EUR, etc.)
        # if you want to skip them completely
        skip_bases = {"USD","USDC","USDT","EUR"}
        filtered_coins = [c for c in coins if c.upper() not in skip_bases]

        # chunk them
        chunk_size = 20
        out_map = {}
        coin_list = list(set(filtered_coins))  # unique
        for i in range(0, len(coin_list), chunk_size):
            subset = coin_list[i:i+chunk_size]
            subset_data = self._fetch_cmc_chunk(subset)
            out_map.update(subset_data)
        return out_map

    def _fetch_cmc_chunk(self, coins_subset):
        symbol_param = ",".join([c.upper() for c in coins_subset])
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": self.cmc_api_key
        }
        params = {
            "symbol": symbol_param,
            "convert": "USD"
        }
        out_map = {}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code==200:
                data= resp.json().get("data", {})
                for sym, val in data.items():
                    quote= val["quote"]["USD"]
                    mc= quote.get("market_cap", 0)
                    vol= quote.get("volume_24h", 0)
                    sup= val.get("circulating_supply", 0)
                    out_map[sym]= {
                        "market_cap": Decimal(str(mc)),
                        "volume_24h": Decimal(str(vol)),
                        "circulating_supply": Decimal(str(sup))
                    }
            else:
                logging.warning(f"[FundEngine] CMC fetch fail => {resp.status_code} => {resp.text}")
        except Exception as e:
            logging.warning(f"[FundEngine] batch_fetch_coinmarketcap => {e}")
        return out_map
    # end of chunk patch

    def _store_cmc_fundamentals(self, coin, data):
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        def _ins(metric, val):
            q = """INSERT INTO fundamentals (coin, metric_name, metric_value, date)
                   VALUES (%s, %s, %s, %s)"""
            self.cursor.execute(q, (coin, metric, float(val), now_date))

        mc= data["market_cap"]
        vol= data["volume_24h"]
        sup= data["circulating_supply"]

        _ins("cmc_market_cap", mc)
        _ins("cmc_volume_24h", vol)
        _ins("cmc_circ_supply", sup)
        self.db.commit()

    def _fallback_kraken(self, coin):
        # PATCH #2: skip fallback for stable/base
        if coin.upper() in ["USD","USDC","USDT","EUR"]:
            logging.debug(f"[FundEngine] skip fallback => stable or base coin: {coin}")
            return

        logging.info(f"[FundEngine] fallback => kraken => {coin}")
        pair= self._map_coin_to_kraken(coin)
        try:
            ticker= self.exchange.fetch_ticker(pair)
            base_vol = ticker.get("baseVolume", 0)
            quote_vol= ticker.get("quoteVolume", 0)
            now_date = datetime.datetime.now().strftime('%Y-%m-%d')

            ins = """INSERT INTO fundamentals (coin, metric_name, metric_value, date)
                     VALUES (%s, %s, %s, %s)"""
            self.cursor.execute(ins, (coin, 'kraken_base_volume_24h', float(base_vol), now_date))
            self.cursor.execute(ins, (coin, 'kraken_quote_volume_24h', float(quote_vol), now_date))
            self.db.commit()
        except Exception as e:
            logging.warning(f"[FundEngine] fallback kraken => {e}")

    def _map_coin_to_kraken(self, coin):
        kr_map= self.config.get("kraken_map", {})
        base= kr_map.get(coin, coin)
        return f"{base}/{self.stable_symbol}"

    def _compute_scores(self, coins):
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        for coin in coins:
            q = """
            SELECT metric_name, metric_value
            FROM fundamentals
            WHERE coin=%s
            ORDER BY date DESC
            LIMIT 50
            """
            self.cursor.execute(q, (coin,))
            rows = self.cursor.fetchall()
            mc_ = None
            vol_ = None

            for r in rows:
                metric = r[0]
                val = Decimal(str(r[1]))
                if metric == 'cmc_market_cap':
                    mc_ = val
                elif metric == 'cmc_volume_24h':
                    vol_ = val

            if mc_ is None:
                mc_ = Decimal("0")
            if vol_ is None:
                vol_ = Decimal("0")

            raw_score = (mc_ / Decimal("1e9")) + (vol_ / Decimal("1e8"))
            score = min(raw_score, Decimal("100"))

            ins = """INSERT INTO fundamentals (coin, metric_name, metric_value, date)
                     VALUES (%s, 'score', %s, %s)"""
            self.cursor.execute(ins, (coin, float(score), now_date))

        self.db.commit()

def find_kraken_symbol_for_coin(kraken, coin, preferred_quotes=None):
    if preferred_quotes is None:
        preferred_quotes = ["USD","USDT","EUR"]

    kraken.load_markets()
    all_markets = kraken.markets
    coin_upper = coin.upper()

    for symbol_str, market_info in all_markets.items():
        base = market_info.get('base')
        quote = market_info.get('quote')
        if base and base.upper() == coin_upper and quote.upper() in [q.upper() for q in preferred_quotes]:
            return symbol_str
    return None
