import schedule
import time
import logging
import yaml
import ccxt
import datetime
import os
import sys
import signal
from decimal import Decimal

# DB & Portfolio
from db.connection import get_db_connection
from portfolio_manager import PortfolioManager  # For Spot

# Strategy for Spot
from strategy_manager import StrategyManager

# Fundamentals & Trainers
from fundamental_engine import FundamentalEngine
from ml_engine.daily_trainer import DailyTrainer
from ml_engine.weekly_trainer import WeeklyTrainer

from ohlc_fetcher import OhlcFetcher
from tabulate import tabulate
import statistics

from kraken_websocket_handler import KrakenWebsocketHandler


class Executor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set logging level
        log_level = self.config.get("logging_level","INFO").upper()
        logging.getLogger().setLevel(log_level)

        # DB connection
        self.db = get_db_connection()
        self.cursor = self.db.cursor(dictionary=True)

        # Initialize ccxt exchange (Spot account)
        self.exchange = self._init_exchange(self.config["exchange"])

        # === Create a Spot PortfolioManager (USD) ===
        self.spot_mgr = PortfolioManager(
            db=self.db,
            exchange=self.exchange,
            config=self.config,
            stable_override=None  # uses config["portfolio"]["stable_symbol"]
        )

        # Fundamentals & ML
        self.fund_engine = FundamentalEngine(self.db, self.exchange, self.config)
        self.daily_trainer = DailyTrainer(self.db, self.exchange, self.config)
        self.weekly_trainer = WeeklyTrainer(self.db, self.exchange, self.config)

        # Spot strategy
        self.strategy_mgr = StrategyManager(
            db=self.db,
            exchange=self.exchange,
            config=self.config,
            portfolio_mgr=self.spot_mgr
        )

        # Make sure we have some stable coin for Spot
        self.spot_mgr.initialize_starting_capital()

        # Websocket for Spot (Kraken)
        self.ws_handler = KrakenWebsocketHandler(
            exchange=self.exchange,
            db=self.db,
            config=self.config,
            portfolio_mgr=self.spot_mgr
        )
        self.ws_handler.start()

        # Schedule tasks
        self._schedule_tasks()

    def _init_exchange(self, exch_cfg):
        exchange_class = getattr(ccxt, exch_cfg["name"])
        return exchange_class({
            'apiKey': exch_cfg["apiKey"],
            'secret': exch_cfg["secret"],
            'enableRateLimit': exch_cfg.get("enableRateLimit", True)
        })

    def _schedule_tasks(self):
        # short-term spot trades every 15 minutes
        schedule.every(60).minutes.at(":05").do(self._short_term_trades)

        # daily & weekly routines
        schedule.every().day.at("06:00").do(self._daily_routine)
        schedule.every().monday.at("06:00").do(self._weekly_routine)

        # poll open orders
        schedule.every(1).minutes.do(self._poll_open_orders)

        # sync real spot balances
        schedule.every(5).minutes.do(self.spot_mgr.sync_with_real_balance)

        # daily DB backup at 23:59
        schedule.every().day.at("23:59").do(self._daily_db_backup)

        # fetch OHLC each hour
        schedule.every().hour.at(":00").do(self._hourly_ohlc_fetch)

    def run(self):
        logging.info("[Executor] Main loop started.")

        def _shutdown_signal_handler(signum, frame):
            logging.info("[Executor] Caught shutdown signal => stopping everything.")
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
            logging.info("[Executor] Stopped by user.")

    def _short_term_trades(self):
        logging.info("[Executor] => short_term_trades (Spot)")
        try:
            self.strategy_mgr.check_short_term_trades()
        except Exception as e:
            logging.error(f"[Executor] short_term_trades => {e}", exc_info=True)

    def _daily_routine(self):
        logging.info("[Executor] => daily_routine")
        try:
            self.fund_engine.scan_new_coins()
            self.fund_engine.update_fundamentals()
            coin_list = self._get_active_coins()
            self._compute_and_store_30d_volatility(coin_list)
            self._store_performance_snapshot()
        except Exception as e:
            logging.error(f"[Executor] daily_routine => {e}", exc_info=True)

    def _weekly_routine(self):
        logging.info("[Executor] => weekly_routine")
        try:
            self.fund_engine.scan_new_coins()
            self.strategy_mgr.rebalance_portfolio()  # Spot weekly rebalance
        except Exception as e:
            logging.error(f"[Executor] weekly_routine => {e}", exc_info=True)

    def _poll_open_orders(self):
        try:
            open_os = self.exchange.fetch_open_orders()
            now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            table = []
            for o in open_os:
                oid = o.get('id','UNKNOWN')
                symbol = o.get('symbol','?')
                side = o.get('side','?')
                amount = o.get('amount',0.0)
                filled = o.get('filled',0.0)
                status = o.get('status','open')
                table.append([oid, symbol, side, amount, filled, status])
            if table:
                print("\n===== PARTIAL FILL DASHBOARD =====")
                print(f"Time: {now_}")
                print(tabulate(table, headers=["OrderID","Symbol","Side","Amt","Filled","Status"]))
        except Exception as e:
            logging.warning(f"[Executor] _poll_open_orders => {e}")

    def _daily_db_backup(self):
        logging.info("[Executor] => daily_db_backup")
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
            logging.info(f"[Backup] => {backup_file} created.")
        except Exception as e:
            logging.error(f"[Backup] error => {e}", exc_info=True)

    def _hourly_ohlc_fetch(self):
        coins = self._get_active_coins()
        ofetch = OhlcFetcher(
            db=self.db,
            exchange_cfg=self.config["exchange"],
            coin_list=coins
        )
        ofetch.fetch_and_store_ohlc(timeframe="1h", limit=100)
        logging.info("[Executor] Hourly OHLC fetch complete.")

    def _store_performance_snapshot(self):
        total_val = self._calc_portfolio_value()
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        note = "Daily Snapshot"
        q = """INSERT INTO performance_snapshots (snapshot_date, total_value, note)
               VALUES (%s, %s, %s)"""
        self.cursor.execute(q, (now_date, float(total_val), note))
        self.db.commit()
        logging.info(f"[Performance] snapshot => date={now_date}, total={total_val}")

    def _calc_portfolio_value(self):
        pf = self.spot_mgr.get_portfolio()
        stable_amt = pf.get(self.spot_mgr.stable_symbol, 0)
        total_val = stable_amt
        for c, amt in pf.items():
            if c == self.spot_mgr.stable_symbol or amt <= 0:
                continue
            pair = self.config.get("kraken_map",{}).get(c, c) + f"/{self.spot_mgr.stable_symbol}"
            ohlc = self._try_fetch_ohlcv(pair, '1h', limit=1)
            if ohlc and len(ohlc) > 0:
                last_close = ohlc[-1][4]
                last_close_dec = Decimal(str(last_close))
                total_val += amt * last_close_dec
            else:
                logging.warning(f"[PerformanceVal] can't fetch {pair}")
        return total_val

    def _try_fetch_ohlcv(self, symbol, timeframe='1h', limit=1):
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except:
            return None

    def _get_active_coins(self):
        coin_list = []
        q = "SELECT coin FROM portfolio_positions"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        for r in rows:
            c = r["coin"]
            if c and c != self.config["portfolio"]["stable_symbol"]:
                coin_list.append(c)
        return list(set(coin_list))
