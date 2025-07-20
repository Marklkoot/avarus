import logging
import datetime
import os
import yaml
import ccxt
import pandas as pd
from decimal import Decimal

from db.connection import get_db_connection
# Import your advanced margin AI (the updated code with Optuna)
from ml_engine.ai_signals_margin import AIPricePredictorMargin

LOG_PATH = r"C:\Users\markl\Avarus2\logs\margin_ai_trainer.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

class MarginAITrainer:
    """
    Runs the margin AI training step.
    - For top coins => use Optuna for hyperparam tuning (n_trials=10, e.g.).
    - For other coins => standard training.
    Typically scheduled daily or every few hours.
    """

    def __init__(self, db, exchange, config):
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # The advanced margin AI predictor
        self.ai_predictor_margin = AIPricePredictorMargin(db, config)

    def run_ai_training(self):
        """
        Main entry: Train margin AI on new data for each relevant coin.
        Top coins => Optuna search
        Others => standard training
        """
        self.logger.info("[MarginAITrainer] => start margin AI training (12-feature + time-split + possible Optuna)")

        coin_list = self._pick_daily_coins_for_ai()
        # Example: first 10 coins => do Optuna, rest => fallback
        # Adjust to your preference
        top_coins = set(coin_list[:20])  # top 20

        for coin in coin_list:
            df_raw = self._fetch_ohlc_for_ai(coin, timeframe="1h", days=60)
            if df_raw is None or len(df_raw) < 100:
                self.logger.info(
                    f"[MarginAITrainer] => skip AI => not enough data => {coin}"
                )
                continue

            try:
                if coin in top_coins:
                    # For top 20 coins => do an Optuna search
                    self.logger.info(f"[MarginAITrainer] => Optuna search => {coin}")
                    best_params, best_loss = self.ai_predictor_margin.hyperparam_optuna_search(
                        coin=coin,
                        df_raw=df_raw,
                        seq_len=20,         # or 30, if you prefer
                        epochs=30,          # you can raise or lower
                        val_fraction=0.2,
                        patience=3,
                        n_trials=30         # you can raise or lower
                    )
                    self.logger.info(f"[MarginAITrainer] => {coin} => best params={best_params}, best loss={best_loss}")
                else:
                    # For the rest => do a standard training
                    self.logger.info(f"[MarginAITrainer] => standard train => {coin}")
                    self.ai_predictor_margin.train_model(
                        coin,
                        df_raw,
                        seq_len=20,
                        epochs=30,
                        val_fraction=0.2,
                        patience=3
                    )
            except Exception as e:
                self.logger.warning(
                    f"[MarginAITrainer] => AI train error => {coin}, {e}"
                )

        self.logger.info("[MarginAITrainer] => done margin AI training")

    def _pick_daily_coins_for_ai(self):
        # 1) gather coins from portfolio_positions_margin
        q = """SELECT coin FROM portfolio_positions_margin"""
        self.cursor.execute(q)
        pf_rows = self.cursor.fetchall()
        pf_coins = {r[0] for r in pf_rows}

        # 2) gather distinct coins from ohlc_data
        q2 = """SELECT DISTINCT coin FROM ohlc_data WHERE timeframe='1h'"""
        self.cursor.execute(q2)
        db_rows = self.cursor.fetchall()
        db_coins = {r[0] for r in db_rows}

        # union them
        all_coins = pf_coins.union(db_coins)

        # optional filter out stable coins, etc.
        skip_coins = {"EUR","USD","GBP"}
        final_coins = [c for c in all_coins if c not in skip_coins]

        # if it's huge, you may want to limit to the top N by volume or something else
        return list(final_coins)  # or sorted, or whatever

    def _fetch_ohlc_for_ai(self, coin, timeframe="1h", days=60):
        """
        Return a DataFrame with columns: [time, open, high, low, close, volume].
        Sorted ascending by time, typed as datetime.
        """
        end_dt = datetime.datetime.utcnow()
        start_dt = end_dt - datetime.timedelta(days=days)

        q = """
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlc_data
            WHERE coin=%s
              AND timeframe=%s
              AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
        """
        self.cursor.execute(q, (coin, timeframe, start_dt, end_dt))
        rows = self.cursor.fetchall()
        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
        df.rename(columns={"timestamp":"time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def main():
    """
    Entry point if run as a script.
    Typically scheduled to run every 6 or 24 hours.
    """
    logging.info("[MarginAITrainer] => script start")

    # Load config
    config_path = os.getenv("CONFIG_PATH","config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # DB + exchange
    import ccxt
    db = get_db_connection()
    exch_cfg = config["exchange"]
    exchange_class = getattr(ccxt, exch_cfg["name"])
    exchange = exchange_class({
        'apiKey': exch_cfg["apiKey"],
        'secret': exch_cfg["secret"],
        'enableRateLimit': exch_cfg.get("enableRateLimit", True)
    })

    # Create trainer + run
    trainer = MarginAITrainer(db, exchange, config)
    trainer.run_ai_training()

    logging.info("[MarginAITrainer] => script end => AI training done.")


if __name__=="__main__":
    main()
