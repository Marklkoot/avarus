# trainer_weekly_runner.py

import os
import logging
import yaml
import ccxt
from db.connection import get_db_connection
from ml_engine.weekly_trainer import WeeklyTrainer

LOG_PATH = r"C:\Users\markl\Avarus2\logs\weekly_trainer.log"
logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def main():
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    db = get_db_connection()
    exch_cfg = config["exchange"]
    exchange_class = getattr(ccxt, exch_cfg["name"])
    exchange = exchange_class({
        'apiKey': exch_cfg["apiKey"],
        'secret': exch_cfg["secret"],
        'enableRateLimit': exch_cfg.get("enableRateLimit", True)
    })

    weekly_trainer = WeeklyTrainer(db, exchange, config)
    weekly_trainer.run_weekly_training()

    logging.info("[trainer_weekly_runner] => weekly training done => new macro params stored.")

if __name__ == "__main__":
    main()
