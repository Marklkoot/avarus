# trainer_daily_margin_ai_runner.py

import os
import logging
import yaml
import ccxt
from db.connection import get_db_connection

# Instead of importing DailyTrainerMargin from daily_trainer_margin,
# we now import the AI trainer class from daily_trainer_margin_ai
from ml_engine.daily_trainer_margin_ai import MarginAITrainer

LOG_PATH = r"C:\Users\markl\Avarus2\logs\margin_ai_trainer.log"
logging.basicConfig(
    filename=LOG_PATH,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def main():
    # 1) Read config
    config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2) DB connection
    db = get_db_connection()

    # 3) Create exchange
    exch_cfg = config["exchange"]
    exchange_class = getattr(ccxt, exch_cfg["name"])
    exchange = exchange_class({
        'apiKey': exch_cfg["apiKey"],
        'secret': exch_cfg["secret"],
        'enableRateLimit': exch_cfg.get("enableRateLimit", True)
    })

    # 4) Instantiate the new margin AI trainer
    margin_ai_trainer = MarginAITrainer(db, exchange, config)

    # 5) Run margin AI training
    margin_ai_trainer.run_ai_training()

    logging.info("[trainer_daily_margin_ai_runner] => margin AI training done => new models stored in ai_models_margin")

if __name__ == "__main__":
    main()
