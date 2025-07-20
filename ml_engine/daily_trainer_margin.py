import logging
import datetime
import os
import yaml
import ccxt
import random
import json
from decimal import Decimal

from db.connection import get_db_connection
# The margin backtester for meltdown param evaluation:
from ml_engine.backtester_margin import BacktesterMargin

LOG_PATH = r"C:\Users\markl\Avarus2\logs\margin_ga_runner.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

class DailyTrainerMargin:
    """
    Runs meltdown GA param search for margin trades.
    No AI training here.
    """

    def __init__(self, db, exchange, config):
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.backtester_margin = BacktesterMargin(
            db=db,
            config=config,
            stable_symbol=config.get("margin_portfolio",{}).get("stable_symbol","EUR"),
            initial_investment=500.0
        )

    def run_daily_training(self):
        """
        Main meltdown GA: pick random candidates, evaluate with backtester, 
        evolve best meltdown param sets.
        """
        self.logger.info("[MarginGARunner] => meltdown GA start")

        coin_list = self._pick_coins_for_ga()

        end_date = datetime.datetime.utcnow()
        full_start_date = end_date - datetime.timedelta(days=60)
        in_sample_end = full_start_date + datetime.timedelta(days=40)
        oos_start_date = in_sample_end
        oos_end_date = end_date

        population_size = 30
        generations = 8
        survival_rate = 0.3
        mutation_prob = 0.2

        population = [self._random_candidate() for _ in range(population_size)]
        best_score = Decimal("-999999")
        best_params = {}
        acc_alpha = Decimal("2.0")

        for gen in range(generations):
            self.logger.info(f"[GA][Margin] gen={gen+1}/{generations}, pop={len(population)}")

            scored_pop = []
            for params in population:
                result_in_sample = self.backtester_margin.run_backtest(
                    param_map=params,
                    coin_list=coin_list,
                    start_date=full_start_date,
                    end_date=in_sample_end,
                    timeframe="1h"
                )
                fv_insample = Decimal(str(result_in_sample["final_value"]))
                total_coins = Decimal(str(result_in_sample["total_coins"]))

                combined_score = fv_insample + acc_alpha* total_coins
                scored_pop.append((params, combined_score, result_in_sample))

            scored_pop.sort(key=lambda x: x[1], reverse=True)

            if scored_pop[0][1] > best_score:
                best_score = scored_pop[0][1]
                best_params = scored_pop[0][0]
                self.logger.info(f"[GA][Margin] new best => score={best_score}, params={best_params}")

            survivors_count = int(len(scored_pop)* survival_rate)
            survivors = scored_pop[:survivors_count]
            next_pop = []
            for (sp, _, _) in survivors:
                next_pop.append(sp)

            while len(next_pop)< population_size:
                p1 = random.choice(survivors)[0]
                p2 = random.choice(survivors)[0]
                child = self._crossover(p1, p2)
                self._mutate(child, mutation_prob)
                next_pop.append(child)

            population = next_pop

        # Final best => check OOS
        oos_result = self.backtester_margin.run_backtest(
            param_map=best_params,
            coin_list=coin_list,
            start_date=oos_start_date,
            end_date=oos_end_date,
            timeframe="1h"
        )
        self.logger.info(f"[GA][Margin] OOS => final_val={oos_result['final_value']}, "
                         f"DD={oos_result['max_dd']}, coins={oos_result['total_coins']}")

        # Compare vs old
        old_params = self._load_current_params()
        old_insample = self.backtester_margin.run_backtest(
            param_map=old_params,
            coin_list=coin_list,
            start_date=full_start_date,
            end_date=in_sample_end,
            timeframe="1h"
        )
        fv_old = Decimal(str(old_insample["final_value"]))
        old_coins = Decimal(str(old_insample["total_coins"]))
        combined_old = fv_old + acc_alpha* old_coins

        if best_score> combined_old:
            self._store_params(best_params)
            self.logger.info("[GA][Margin] => stored new meltdown params.")
        else:
            self.logger.info("[GA][Margin] => no improvement => keep old meltdown params.")

        self.logger.info("[MarginGARunner] meltdown GA end => best_params concluded")

    #-----------------------------------------------
    # GA helper stuff
    #-----------------------------------------------
    def _random_candidate(self):
        dip_buy_opts = [0.003, 0.005, 0.01, 0.02]
        meltdown_thresh_opts = [0.25, 0.30, 0.35, 0.40]
        meltdown_anchor_buf_opts = [0.02, 0.05]
        meltdown_reentry_opts = [0.10, 0.15]

        partial_sell_tiers_list = [
            [{"gain": 0.05, "ratio": 0.03}, {"gain": 0.10, "ratio": 0.05}],
            [{"gain": 0.08, "ratio": 0.04}, {"gain": 0.15, "ratio": 0.07}],
            [{"gain": 0.10, "ratio": 0.05}, {"gain": 0.20, "ratio": 0.10}],
        ]
        meltdown_tiers_list = [
            [{"drawdown":0.20, "sell_ratio":0.25}, {"drawdown":0.30, "sell_ratio":0.30}],
            [{"drawdown":0.25, "sell_ratio":0.25}, {"drawdown":0.35, "sell_ratio":0.50}],
            [{"drawdown":0.15, "sell_ratio":0.20}, {"drawdown":0.25, "sell_ratio":0.25}, {"drawdown":0.40, "sell_ratio":0.50}],
        ]
        cand = {
            "dip_buy_pct_drop": random.choice(dip_buy_opts),
            "meltdown_threshold": random.choice(meltdown_thresh_opts),
            "meltdown_anchor_buffer": random.choice(meltdown_anchor_buf_opts),
            "meltdown_reentry_pct": random.choice(meltdown_reentry_opts),
            "partial_sell_tiers": random.choice(partial_sell_tiers_list),
            "meltdown_tiers": random.choice(meltdown_tiers_list),
            # for AI fraction if desired
            "ai_buy_fraction": 0.02,
            "ai_sell_fraction": 0.20,
            "core_ratio": 0.70,
            "trailing_stop_pct":0.08
        }
        return cand

    def _crossover(self, p1, p2):
        child = {}
        for k in p1.keys():
            child[k] = random.choice([p1[k], p2[k]])
        return child

    def _mutate(self, param_map, prob=0.2):
        if random.random() < prob:
            param_to_mutate = random.choice(list(param_map.keys()))
            if param_to_mutate=="dip_buy_pct_drop":
                param_map[param_to_mutate] = random.choice([0.003, 0.005, 0.01, 0.02, 0.03])

    #-----------------------------------------------
    # DB param fetch/store
    #-----------------------------------------------
    def _pick_coins_for_ga(self):
        q = """
          SELECT coin, quantity
          FROM portfolio_positions_margin
          ORDER BY quantity DESC
          LIMIT 30
        """
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        coin_list = [r[0] for r in rows]
        if not coin_list:
            coin_list = ["BTC","ETH"]
        return coin_list

    def _load_current_params(self):
        q = "SELECT param_name, param_value FROM meta_parameters_margin"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        pm = {r[0]: r[1] for r in rows}
        # parse JSON
        if "partial_sell_tiers" in pm:
            try:
                pm["partial_sell_tiers"] = json.loads(pm["partial_sell_tiers"])
            except:
                pm["partial_sell_tiers"] = []
        if "meltdown_tiers" in pm:
            try:
                pm["meltdown_tiers"] = json.loads(pm["meltdown_tiers"])
            except:
                pm["meltdown_tiers"] = []
        return pm

    def _store_params(self, param_dict):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for k,v in param_dict.items():
            if isinstance(v, (list,dict)):
                v_str = json.dumps(v)
            else:
                v_str = str(v)
            q = """REPLACE INTO meta_parameters_margin (param_name, param_value, last_updated)
                   VALUES (%s, %s, %s)"""
            self.cursor.execute(q, (k, v_str, now_))
        self.db.commit()

def main():
    """
    Typically run every 48 hours or so.
    """
    logging.info("[MarginGARunner] => meltdown GA script start")

    config_path = os.getenv("CONFIG_PATH","config/config.yaml")
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

    runner = MarginGARunner(db, exchange, config)
    runner.run_meltdown_ga()

    logging.info("[MarginGARunner] => meltdown GA script end")


if __name__=="__main__":
    main()
