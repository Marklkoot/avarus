# executor_margin.py

import schedule
import time
import logging
import yaml
import ccxt
import datetime
import os
import sys
import signal
from tabulate import tabulate
from decimal import Decimal

from db.connection import get_db_connection

# MARGIN Portfolio Manager + Strategy
from portfolio_manager_margin import PortfolioManagerMargin
from strategy_margin import StrategyMargin

# Optional if you still want to keep them around (but not used by meltdown only):
# from fundamental_engine import FundamentalEngine
# from ml_engine.daily_trainer import DailyTrainer
# from ml_engine.weekly_trainer import WeeklyTrainer

# We skip OHLC fetch for margin
# from ohlc_fetcher import OhlcFetcher

from kraken_websocket_handler_public import KrakenWebsocketHandlerPublic
from kraken_websocket_handler_margin import KrakenWebsocketHandlerMargin

# Set up margin-specific logger
margin_logger = logging.getLogger("margin_trading")
margin_logger.setLevel(logging.INFO)

LOG_PATH_MARGIN = r"C:\Users\markl\Avarus2\logs\margin_trading.log"
os.makedirs(os.path.dirname(LOG_PATH_MARGIN), exist_ok=True)

file_handler = logging.FileHandler(LOG_PATH_MARGIN)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
margin_logger.addHandler(file_handler)
margin_logger.propagate = False

class ExecutorMargin:
    def __init__(self, config_path="configmargin.yaml"):
        # Load the margin config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        log_level = self.config.get("logging_level","INFO").upper()
        logging.getLogger().setLevel(log_level)

        self.db = get_db_connection()
        self.cursor = self.db.cursor(dictionary=True)

        # Initialize ccxt exchange with MARGIN account credentials
        self.exchange = self._init_exchange(self.config["exchange"])

        # Create a Margin PortfolioManager
        self.margin_mgr = PortfolioManagerMargin(
            db=self.db,
            exchange=self.exchange,
            config=self.config
        )

        # Margin Strategy
        self.strategy_mgr_margin = StrategyMargin(
            db=self.db,
            exchange=self.exchange,
            config=self.config,
            portfolio_mgr=self.margin_mgr
        )

         #If you'd like an initial deposit of margin capital:
        #self.margin_mgr.initialize_starting_capital(500)

        # If you wanted to run fundamentals or ML:
        # self.fund_engine = FundamentalEngine(self.db, self.exchange, self.config)
        # self.daily_trainer = DailyTrainer(self.db, self.exchange, self.config)
        # self.weekly_trainer = WeeklyTrainer(self.db, self.exchange, self.config)

        self.ws_handler_public = KrakenWebsocketHandlerPublic(
            db=self.db,
            config=self.config,
            portfolio_mgr=self.margin_mgr,
            exchange=self.exchange
        )
        self.ws_handler_public.start()

        # WebSocket for Margin
        self.ws_handler = KrakenWebsocketHandlerMargin(
            exchange=self.exchange,
            db=self.db,
            config=self.config,
            portfolio_mgr=self.margin_mgr
        )
        self.ws_handler.start()

        # Track initial deposit for PnL if you want
        self.initial_margin_deposit = Decimal("700")

        self._schedule_tasks()

    def _init_exchange(self, exch_cfg):
        exchange_class = getattr(ccxt, exch_cfg["name"])
        return exchange_class({
            'apiKey': exch_cfg["apiKey"],
            'secret': exch_cfg["secret"],
            'enableRateLimit': exch_cfg.get("enableRateLimit", True)
        })

    def _schedule_tasks(self):
        # Short-term margin trades
        schedule.every(60).minutes.at(":02").do(self._short_term_trades_margin)
        # Log margin equity
        schedule.every(60).minutes.at(":00").do(self.log_margin_equity)
        schedule.every(10).minutes.do(self.log_open_positions_kraken)

        # Sync real margin balances
        schedule.every(5).minutes.do(self.margin_mgr.sync_with_real_balance)

        # We skip OHLC fetch for margin here

        # Daily routine at 06:00 if you want
        schedule.every().day.at("06:00").do(self._daily_routine)
        # Weekly routine at Monday 06:00
        schedule.every().monday.at("06:00").do(self._weekly_routine)

        # DB backup
        schedule.every().day.at("23:59").do(self._daily_db_backup)

        # === ADDED START ===
        # 4) Dynamic WebSocket subscription updates => every 5 minutes
        schedule.every(5).minutes.do(self._update_ws_subscriptions)
        # === ADDED END ===

    def run(self):
        logging.info("[ExecutorMargin] Main loop started.")

        def _shutdown_signal_handler(signum, frame):
            logging.info("[ExecutorMargin] Caught shutdown signal => stopping everything.")
            if self.ws_handler:
                self.ws_handler.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown_signal_handler)
        signal.signal(signal.SIGTERM, _shutdown_signal_handler)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("[ExecutorMargin] Stopped by user.")

    def _short_term_trades_margin(self):
        logging.info("[ExecutorMargin] => short_term_trades_margin")
        try:
            self.strategy_mgr_margin.check_short_term_trades()
        except Exception as e:
            logging.error(f"[ExecutorMargin] _short_term_trades_margin => {e}", exc_info=True)

    def log_margin_equity(self):
        """
        Logs the official margin equity from Kraken's 'TradeBalance' endpoint
        (or 'fetch_balance' with type='trading'), along with daily PnL 
        vs. self.initial_margin_deposit.
        """
        try:
            # 1) Attempt CCXT fetch_balance for 'trading' type
            bal = self.exchange.fetch_balance({'type': 'trading'})
            eb = None
            if 'info' in bal and 'result' in bal['info']:
                eb = bal['info']['result'].get('eb')  # 'eb' => equity balance, per Kraken docs

            # 2) If still no 'eb', do a direct raw call to 'TradeBalance'
            if not eb:
                resp = self.exchange.fetch2(
                    'TradeBalance',
                    'private',
                    'POST',
                    {'asset': 'ZEUR'}  # or 'ZUSD' if your base currency is USD
                )
                # Expect: {'error': [], 'result': {'eb': '...', 'tb': '...', ...}}
                eb = resp['result'].get('eb')

            if not eb:
                # We couldn't find the 'eb' field => log a warning
                margin_logger.warning("[RealMargin] 'eb' field not found in TradeBalance or fetch_balance response.")
                return

            # Convert to Decimal, compute daily PnL
            kraken_eq = Decimal(eb)
            daily_pnl = kraken_eq - self.initial_margin_deposit

            margin_logger.info(
                f"[RealMargin] Current equity => {kraken_eq:.2f} EUR, "
                f"Daily PnL => {daily_pnl:+.2f} EUR"
            )

        except Exception as e:
            margin_logger.warning(f"[RealMargin] Could not fetch official margin equity => {e}")

    def _daily_routine(self):
        logging.info("[ExecutorMargin] => daily_routine")
        try:
            # If you need meltdown anchor resets or other daily tasks, do them here
            # If you wanted fundamentals => self.fund_engine.update_fundamentals() etc.
            pass
        except Exception as e:
            logging.error(f"[ExecutorMargin] daily_routine => {e}", exc_info=True)

    def _weekly_routine(self):
        logging.info("[ExecutorMargin] => weekly_routine")
        try:
            # If you want margin rebalancing or meltdown resets weekly
            pass
        except Exception as e:
            logging.error(f"[ExecutorMargin] weekly_routine => {e}", exc_info=True)

    def _daily_db_backup(self):
        logging.info("[ExecutorMargin] => daily_db_backup")
        try:
            now_ = datetime.datetime.now().strftime('%Y%m%d')
            backup_file = f"db_backup_{now_}.sql"
            cmd = (
                f"mysqldump -h{os.getenv('DB_HOST','localhost')} "
                f"-u{os.getenv('DB_USER','avarus_user')} "
                f"-p{os.getenv('DB_PASS','someStrongPassword')} "
                f"{os.getenv('DB_NAME','avarus2')} > {backup_file}"
            )
            os.system(cmd)
            logging.info(f"[ExecutorMargin] => {backup_file} created.")
        except Exception as e:
            logging.error(f"[ExecutorMargin] daily_db_backup => {e}", exc_info=True)

    def _fetch_last_price(self, symbol):
        try:
            t = self.exchange.fetch_ticker(symbol)
            return Decimal(str(t.get('last', t.get('close', 0.0))))
        except:
            return Decimal("0")
            
        # === ADDED START ===
    def _update_ws_subscriptions(self):
        """
        Called every 5 minutes by schedule.
        Dynamically checks which coins we actively trade
        and subscribes/unsubscribes them on the **public** WS
        (wss://ws.kraken.com) for real-time ticker data.
        """
        logging.info("[ExecutorMargin] => _update_ws_subscriptions")

        # 1) Get your margin portfolio
        pf = self.margin_mgr.get_portfolio()  # coin -> qty
        all_coins_in_pf = set()
        for c, amt in pf.items():
            # skip stable or fiat
            if c in ["USD", "GBP", self.config["margin_portfolio"].get("stable_symbol", "EUR")]:
                continue
            if abs(amt) > 0:
                all_coins_in_pf.add(c)

        # 2) Subscribe any new coins on the PUBLIC WS (NOT the margin one)
        for coin in all_coins_in_pf:
            # => calls "subscribe_new_coin" on the public ws
            self.ws_handler_public.subscribe_new_coin(coin)

        # 3) Optionally unsubscribe coins no longer in portfolio
        #    if you want to stop receiving ticker for them.

        # We'll figure out which coins the public WS is currently subscribed to:
        # Convert the pair strings (like "XBT/USD") back to coins
        subscribed_coins = self._get_coins_from_pairs(self.ws_handler_public.current_pairs)
        removed_coins = subscribed_coins - all_coins_in_pf
        for rc in removed_coins:
            self.ws_handler_public.unsubscribe_coin(rc)

        # done

    def _get_coins_from_pairs(self, pair_set):
        """
        Helper: given a set of pairs like {"XBT/USD","ETH/USD"},
        return the corresponding coin names => {"BTC","ETH"} etc.
        Uses the same logic that your public WS uses (the _map_pair_to_coin method).
        """
        coins = set()
        for pair_str in pair_set:
            splitted = pair_str.split("/")
            if len(splitted)<2:
                continue
            base = splitted[0]
            # invert your config's kraken_map
            inv_map = {}
            for ckey, cval in self.config.get("kraken_map", {}).items():
                inv_map[cval] = ckey
            if base in inv_map:
                coins.add(inv_map[base])
            elif base == "XBT":
                coins.add("BTC")
            else:
                coins.add(base)
        return coins
  
    def log_open_positions_kraken(self):
        """
        Fetches all open margin positions from Kraken (not open orders)
        and logs them in a neat table.
        """
        try:
            resp = self.exchange.privatePostOpenPositions({'docalcs': True})
            positions = resp.get("result", {})

            if not positions:
                logging.info("No open positions on Kraken margin.")
                return
            
            # Build rows for a table
            rows = []
            for txid, pos_data in positions.items():
                pair      = pos_data.get("pair", "")
                side      = pos_data.get("type", "")
                vol       = pos_data.get("vol", "0")
                cost      = pos_data.get("cost", "0")
                margin    = pos_data.get("margin", "0")
                # 'net' might be unrealized PnL if docalcs=True; see docs
                net       = pos_data.get("net", "0")  
                # or parse your own PnL from cost/cprice if needed

                rows.append([
                    txid,
                    pair,
                    side,
                    vol,
                    cost,
                    margin,
                    net
                ])

            # Print or tabulate
            from tabulate import tabulate
            table = tabulate(
                rows,
                headers=["TXID", "Pair", "Side", "Volume", "Cost", "MarginUsed", "Net"],
                tablefmt="pretty"
            )
            logging.info(f"\n===== OPEN POSITIONS =====\n{table}\n")
        except Exception as e:
            logging.warning(f"[log_open_positions_kraken] failed => {e}")
