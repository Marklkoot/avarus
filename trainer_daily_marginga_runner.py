# trainer_daily_margin_runner.py

import os
import logging
import yaml
import ccxt
from db.connection import get_db_connection
# IMPORTANT: Import the margin trainer class name from daily_trainer_margin
from ml_engine.daily_trainer_margin import DailyTrainerMargin

LOG_PATH = r"C:\Users\markl\Avarus2\logs\daily_margin_trainer.log"
logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def main():
    # Example: read config from "config/config.yaml"
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # get DB connection
    db = get_db_connection()

    # create exchange
    exch_cfg = config["exchange"]
    exchange_class = getattr(ccxt, exch_cfg["name"])
    exchange = exchange_class({
        'apiKey': exch_cfg["apiKey"],
        'secret': exch_cfg["secret"],
        'enableRateLimit': exch_cfg.get("enableRateLimit", True)
    })

    margin_trainer = DailyTrainerMargin(db, exchange, config)
    margin_trainer.run_daily_training()

    logging.info("[trainer_daily_margin_runner] => margin training done => new params stored in meta_parameters_margin.")

if __name__ == "__main__":
    main()

