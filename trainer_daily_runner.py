# trainer_daily_runner.py

import os
import logging
import yaml
import ccxt
from db.connection import get_db_connection
from ml_engine.daily_trainer import DailyTrainer

LOG_PATH = r"C:\Users\markl\Avarus2\logs\daily_trainer.log"
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

    # instantiate daily trainer
    daily_trainer = DailyTrainer(db, exchange, config)

    # run the training
    daily_trainer.run_daily_training()

    logging.info("[trainer_daily_runner] => daily training done => new params stored in meta_parameters.")

if __name__ == "__main__":
    main()
