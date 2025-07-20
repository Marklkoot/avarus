# ml_engine/daily_trainer.py

import logging
import datetime
import json
import random
from decimal import Decimal
import pandas as pd

from ml_engine.backtester import Backtester
from ml_engine.ai_signals import AIPricePredictor


class DailyTrainer:
    """
    Now uses a Genetic Algorithm to search for best short-term params,
    while preserving the AI training step and meltdown param optimization.
    """

    def __init__(self, db, exchange, config):
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)

        # AI predictor for coin-level training
        self.ai_predictor = AIPricePredictor(db, config)

    def _fetch_ohlc_for_ai(self, coin, timeframe="1h", days=20):
        # (unchanged from your code)
        cursor = self.db.cursor(dictionary=True)
        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=days)

        q = """SELECT timestamp,
                      close,
                      volume,
                      boll_up,
                      boll_down,
                      macd,
                      macd_signal,
                      macd_diff,
                      ema_10,
                      ema_50
               FROM ohlc_data
               WHERE coin=%s
                 AND timeframe=%s
                 AND timestamp BETWEEN %s AND %s
               ORDER BY timestamp ASC
        """
        cursor.execute(q, (coin, timeframe, start_dt, end_dt))
        rows = cursor.fetchall()
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.rename(columns={"timestamp": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def run_daily_training(self):
        self.logger.info("[DailyTrainer] run_daily_training => start")

        # === 1) AI TRAINING STEP ===
        coin_list_for_ai = self._pick_daily_coins()
        for c in coin_list_for_ai:
            df = self._fetch_ohlc_for_ai(c, timeframe="1h", days=20)
            if df is not None and len(df) > 40:  # need at least 2*seq_len
                self.logger.info(f"[DailyTrainer] AI training => {c}")
                self.ai_predictor.train_model(coin=c, df=df, seq_len=20, epochs=10)
            else:
                self.logger.info(f"[DailyTrainer] Not enough data for AI => {c}")

        # === 2) GA CONFIG AND BACKTEST SETUP ===
        end_date = datetime.datetime.now()
        full_start_date = end_date - datetime.timedelta(days=90)
        in_sample_end = full_start_date + datetime.timedelta(days=60)
        oos_start_date = in_sample_end
        oos_end_date = end_date

        coin_list = self._pick_daily_coins()
        backtester = Backtester(
            db=self.db,
            config=self.config,
            stable_symbol=self.config["portfolio"]["stable_symbol"],
            initial_investment=5500.0
        )

        # The weighting factor for total_coins
        acc_alpha = 10.0

        # GA parameters
        population_size = 50
        generations = 10
        survival_rate = 0.3
        mutation_prob = 0.2

        # Initialize random population
        population = [self._random_candidate() for _ in range(population_size)]
        best_overall_score = Decimal("-999999")
        best_overall_params = {}
        best_stats_in_sample = {}

        for gen in range(generations):
            self.logger.info(
                f"[DailyTrainer][GA] Generation {gen+1}/{generations} => "
                f"Population={len(population)}"
            )

            # Evaluate each param set in-sample
            scored_pop = []
            for params in population:
                result_in_sample = backtester.run_backtest(
                    param_map=params,
                    coin_list=coin_list,
                    start_date=full_start_date,
                    end_date=in_sample_end,
                    timeframe="1h"
                )
                fv_insample = result_in_sample["final_value"]
                total_coins_float = result_in_sample["total_coins"]

                # single-objective: final_value + acc_alpha * total_coins
                combined_score_f = fv_insample + (acc_alpha * total_coins_float)
                scored_pop.append((params, combined_score_f, result_in_sample))

            # Sort descending by combined_score_f
            scored_pop.sort(key=lambda x: x[1], reverse=True)

            # Check if best in this generation is better than all-time best
            if scored_pop[0][1] > float(best_overall_score):
                best_overall_score = Decimal(str(scored_pop[0][1]))
                best_overall_params = scored_pop[0][0]
                best_stats_in_sample = scored_pop[0][2]
                self.logger.info(f"[GA] New best overall => score={best_overall_score}, {best_overall_params}")

            # Survive top fraction
            survivors_count = int(len(scored_pop) * survival_rate)
            survivors = scored_pop[:survivors_count]

            self.logger.info(
                f"[GA] best in gen => {survivors[0][1]} => {survivors[0][0]}  /  "
                f"worst in gen => {survivors[-1][1]} => {survivors[-1][0]}"
            )

            # Reproduce => build next population
            next_population = []
            # keep survivors as-is
            for (survivor_params, _, _) in survivors:
                next_population.append(survivor_params)

            # fill up rest of population
            while len(next_population) < population_size:
                p1, _, _ = random.choice(survivors)
                p2, _, _ = random.choice(survivors)
                child = self._crossover(p1, p2)
                self._mutate(child, prob=mutation_prob)
                next_population.append(child)

            population = next_population

        # best_overall_params now found
        self.logger.info(
            f"[DailyTrainer][GA] best overall in-sample => score={best_overall_score}, params={best_overall_params}"
        )

        # === 3) OOS check for best_overall_params ===
        result_oos = backtester.run_backtest(
            param_map=best_overall_params,
            coin_list=coin_list,
            start_date=oos_start_date,
            end_date=oos_end_date,
            timeframe="1h"
        )
        oos_value = result_oos["final_value"]
        oos_total_coins = result_oos["total_coins"]
        oos_dd = result_oos["max_dd"]
        oos_win = result_oos["win_count"]
        oos_loss= result_oos["loss_count"]
        self.logger.info(
            f"[DailyTrainer] OOS => final_val={oos_value:.2f}, maxDD={oos_dd:.2f}, "
            f"wins={oos_win}, losses={oos_loss}, total_coins={oos_total_coins:.2f}"
        )

        # === 4) Compare vs old params ===
        old_params = self._load_current_params()
        result_old = backtester.run_backtest(
            param_map=old_params,
            coin_list=coin_list,
            start_date=full_start_date,
            end_date=in_sample_end,
            timeframe="1h"
        )
        fv_old = result_old["final_value"]
        old_total_coins = result_old["total_coins"]
        combined_old = fv_old + (acc_alpha * old_total_coins)

        live_trades_24h = self._get_live_trade_count(hours=24)
        self.logger.info(f"[DailyTrainer] Live trades in last 24h = {live_trades_24h}")

        if float(best_overall_score) > combined_old:
            self._store_params(best_overall_params)
            self.logger.info("[DailyTrainer] new short-term params stored => live manager picks them up.")
        else:
            if live_trades_24h == 0:
                self.logger.info("[DailyTrainer] no improvement in backtest, but zero live trades => forcing param update.")
                self._store_params(best_overall_params)
                self.logger.info("[DailyTrainer] forced short-term params => live manager picks them up.")
            else:
                self.logger.info("[DailyTrainer] no improvement => keep current short-term params.")

        self.logger.info("[DailyTrainer] done.")


    # ----------------------------------------------------------------
    # GA HELPER METHODS
    # ----------------------------------------------------------------
    def _random_candidate(self):
        """
        Create a random param set from known lists.
        Added meltdown-related parameters so GA can optimize meltdown too.
        """
        dip_buy_opts = [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]
        partial_sell_tiers_list = [
            [
                {"gain": 0.05, "ratio": 0.03},
                {"gain": 0.10, "ratio": 0.05}
            ],
            [
                {"gain": 0.08, "ratio": 0.04},
                {"gain": 0.15, "ratio": 0.07}
            ],
            [
                {"gain": 0.10, "ratio": 0.05},
                {"gain": 0.20, "ratio": 0.10}
            ],
            [
                {"gain": 0.05, "ratio": 0.10},
                {"gain": 0.10, "ratio": 0.20}
            ],
            [
                {"gain": 0.15, "ratio": 0.05},
                {"gain": 0.30, "ratio": 0.10}
            ],
            [
                {"gain": 0.20, "ratio": 0.10},
                {"gain": 0.40, "ratio": 0.15}
            ],
            [
                {"gain": 0.25, "ratio": 0.10},
                {"gain": 0.50, "ratio": 0.20}
            ]
        ]
        rsi_overbought_opts = [60, 65, 70, 75, 80]
        momentum_rsi_opts   = [45, 50, 55, 60, 65]
        momentum_fraction_opts = [0.03, 0.05, 0.08, 0.10]
        trailing_stop_opts = [0.10, 0.15, 0.20]
        core_ratio_opts = [0.65, 0.70, 0.80, 0.85, 0.90]

        ai_buy_fraction_opts = [0.02, 0.05, 0.10]
        ai_sell_fraction_opts= [0.10, 0.20, 0.50]

        # meltdown param options (example ranges)
        meltdown_threshold_opts = [0.50, 0.55, 0.60]
        meltdown_anchor_buf_opts = [0.02, 0.05, 0.08]
        meltdown_reentry_opts = [0.10, 0.15]

        cand = {
            "dip_buy_pct_drop": random.choice(dip_buy_opts),
            "partial_sell_tiers": random.choice(partial_sell_tiers_list),
            "rsi_overbought": random.choice(rsi_overbought_opts),
            "momentum_rsi": random.choice(momentum_rsi_opts),
            "momentum_fraction": random.choice(momentum_fraction_opts),
            "trailing_stop_pct": random.choice(trailing_stop_opts),
            "core_ratio": random.choice(core_ratio_opts),
            "ai_buy_fraction": random.choice(ai_buy_fraction_opts),
            "ai_sell_fraction": random.choice(ai_sell_fraction_opts),

            # meltdown params
            "meltdown_threshold": random.choice(meltdown_threshold_opts),
            "meltdown_anchor_buffer": random.choice(meltdown_anchor_buf_opts),
            "meltdown_reentry_pct": random.choice(meltdown_reentry_opts),

            # fixed
            "bollinger_period": 20,
            "bollinger_stddev": 2.0
        }
        return cand

    def _crossover(self, p1, p2):
        """
        Combine two parents by picking each param from p1 or p2 randomly.
        """
        child = {}
        for k in p1.keys():
            child[k] = random.choice([p1[k], p2[k]])
        return child

    def _mutate(self, params, prob=0.2):
        """
        With probability prob, pick 1 param key and randomize it.
        """
        if random.random() < prob:
            param_to_mutate = random.choice(list(params.keys()))
            if param_to_mutate == "dip_buy_pct_drop":
                params[param_to_mutate] = random.choice([0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05])
            elif param_to_mutate == "rsi_overbought":
                params[param_to_mutate] = random.choice([55, 60, 65, 70])
            elif param_to_mutate == "meltdown_threshold":
                params[param_to_mutate] = random.choice([0.20, 0.25, 0.30, 0.35])
            elif param_to_mutate == "meltdown_anchor_buffer":
                params[param_to_mutate] = random.choice([0.02, 0.05, 0.08])
            elif param_to_mutate == "meltdown_reentry_pct":
                params[param_to_mutate] = random.choice([0.05, 0.10, 0.15])
            # ... you can add more for other params
        return params

    # ----------------------------------------------------------------
    # OLD generate_short_term_candidates() => Not used now (GA replaced)
    # ----------------------------------------------------------------
    # def _generate_short_term_candidates(self):
    #     # (Commented out or removed. We now do GA, no brute force.)
    #     pass

    def _pick_daily_coins(self):
        q = """SELECT coin, quantity FROM portfolio_positions
               WHERE coin != %s
               ORDER BY quantity DESC
               LIMIT 30"""
        self.cursor.execute(q, (self.config["portfolio"]["stable_symbol"],))
        rows = self.cursor.fetchall()
        coin_list = [r[0] for r in rows]
        if not coin_list:
            coin_list = ["BTC","ETH"]
        return coin_list

    def _load_current_params(self):
        q = "SELECT param_name, param_value FROM meta_parameters"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        pm = {r[0]: r[1] for r in rows}

        # 1) Parse meltdown_tiers if present
        if "meltdown_tiers" in pm:
            try:
                pm["meltdown_tiers"] = json.loads(pm["meltdown_tiers"])
            except Exception as e:
                self.logger.warning(f"[Params] meltdown_tiers not valid JSON => {e}")
                pm["meltdown_tiers"] = []

        # 2) Parse partial_sell_tiers if present (already there)
        if "partial_sell_tiers" in pm:
            try:
                pm["partial_sell_tiers"] = json.loads(pm["partial_sell_tiers"])
            except:
                pm["partial_sell_tiers"] = []

        return pm

    def _store_params(self, param_dict):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for k, v in param_dict.items():
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v)
            else:
                v_str = str(v)
            q = """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
                   VALUES (%s, %s, %s)"""
            self.cursor.execute(q, (k, v_str, now_))
        self.db.commit()

    def _get_live_trade_count(self, hours=24):
        try:
            q = """
            SELECT COUNT(*) 
            FROM trade_history
            WHERE fill_timestamp >= (NOW() - INTERVAL %s HOUR)
            """
            self.cursor.execute(q, (hours,))
            row = self.cursor.fetchone()
            if row and row[0]:
                return int(row[0])
        except Exception as e:
            self.logger.warning(f"[DailyTrainer] _get_live_trade_count error => {e}")
        return 0
