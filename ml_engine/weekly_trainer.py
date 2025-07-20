# weekly_trainer.py

import logging
import datetime
import json
from decimal import Decimal

from ml_engine.backtester import Backtester

class WeeklyTrainer:
    """
    Weekly trainer => focuses on 'macro' parameters like core_ratio, fundamental_weight, 
    and updates the rebalancing allocations, etc.
    
    We remove meltdown threshold combos since the daily trainer's GA already tunes meltdown 
    parameters. That avoids overwriting meltdown daily vs. weekly.
    """

    def __init__(self, db, exchange, config):
        self.db = db
        self.cursor = db.cursor()
        self.exchange = exchange
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run_weekly_training(self):
        self.logger.info("[WeeklyTrainer] run_weekly_training => start")

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=60)

        # We'll pick top coins by marketcap for a 'macro' approach
        coin_list = self._pick_top_coins_by_marketcap(limit=5)

        # Step 1) Build combos for core_ratio, fundamental_weight, etc.
        combos = self._generate_macro_candidates()
        self.logger.info("[WeeklyTrainer] => searching combos for core_ratio, fundamental_weight, etc.")

        backtester = Backtester(
            db=self.db,
            config=self.config,
            stable_symbol=self.config["portfolio"]["stable_symbol"],
            initial_investment=5500.0
        )

        acc_alpha = 10.0
        best_score = Decimal("-999999")
        best_params = {}

        # 2) Evaluate each combo
        for combo in combos:
            merged_params = self._merge_with_shortterm_defaults(combo)
            result = backtester.run_backtest(
                param_map=merged_params,
                coin_list=coin_list,
                start_date=start_date,
                end_date=end_date,
                timeframe="1h"
            )
            fv_weekly = result["final_value"]
            total_coins_float = result["total_coins"]

            combined_score_f = fv_weekly + (acc_alpha * total_coins_float)
            if combined_score_f > float(best_score):
                best_score = Decimal(str(combined_score_f))
                best_params = merged_params

        # We found best "macro" params (in this code: core_ratio, fundamental_weight, etc.)
        core_rat    = best_params.get('core_ratio')
        fund_weight = best_params.get('fundamental_weight')

        self.logger.info(
            f"[WeeklyTrainer] best macro combined_score={best_score}, "
            f"core_ratio={core_rat}, fundamental_weight={fund_weight}"
        )

        # Compare vs old
        old_params = self._load_current_params()
        merged_old = self._merge_with_shortterm_defaults(old_params)
        result_old = backtester.run_backtest(
            param_map=merged_old,
            coin_list=coin_list,
            start_date=start_date,
            end_date=end_date,
            timeframe="1h"
        )
        fv_old = result_old["final_value"]
        old_coins_float = result_old["total_coins"]
        combined_old = fv_old + (acc_alpha * old_coins_float)

        if float(best_score) > combined_old:
            # 3) Store new macro-level params
            to_store = {
                "core_ratio":          core_rat,
                "fundamental_weight":  fund_weight,
            }
            self._store_params(to_store)
            self.logger.info("[WeeklyTrainer] new macro params => stored.")
        else:
            self.logger.info("[WeeklyTrainer] no macro improvement => keep old macro params.")

        # --- 4) REBALANCING ---
        rebalance_map = self._pick_coins_for_rebalance()
        self._store_params({"rebalance_target_allocations": rebalance_map})
        self.logger.info("[WeeklyTrainer] updated 'rebalance_target_allocations' => done.")

        self.logger.info("[WeeklyTrainer] done.")

    # ------------------------------------------------------------------
    # Example: pick top coins by marketcap (unchanged)
    # ------------------------------------------------------------------
    def _pick_top_coins_by_marketcap(self, limit=5):
        q = """
        SELECT f.coin, f.metric_value as market_cap
        FROM fundamentals f
        WHERE f.metric_name='cmc_market_cap'
        ORDER BY f.metric_value DESC
        LIMIT %s
        """
        self.cursor.execute(q, (limit,))
        rows = self.cursor.fetchall()
        coins = [r[0] for r in rows]
        if not coins:
            coins = ["BTC","ETH","BNB","XRP","ADA"][:limit]
        return coins

    # ------------------------------------------------------------------
    # CHANGED: Only searching combos for macro items 
    # (not meltdown_threshold or meltdown_anchor, etc.)
    # ------------------------------------------------------------------
    def _generate_macro_candidates(self):
        core_ratio_opts = [0.60, 0.70, 0.80]
        fundamental_weight_opts = [0.0, 0.5, 1.0]

        combos = []
        for cr_ in core_ratio_opts:
            for fw_ in fundamental_weight_opts:
                combos.append({
                    "core_ratio": cr_,
                    "fundamental_weight": fw_
                })
        return combos

    # ------------------------------------------------------------------
    # Merge short-term defaults => keep meltdown from daily
    # ------------------------------------------------------------------
    def _merge_with_shortterm_defaults(self, param_map):
        current = self._load_current_params()
        shortterm_keys = [
            "dip_buy_pct_drop","partial_sell_tiers","rsi_overbought",
            "bollinger_period","bollinger_stddev","momentum_rsi","momentum_fraction",
            "trailing_stop_pct",
            # meltdown keys come from daily trainer, so we do NOT override them here
            "meltdown_threshold","meltdown_anchor_buffer","meltdown_reentry_pct","meltdown_tiers"
        ]
        merged = dict(param_map)
        for k in shortterm_keys:
            if k in current:
                # partial_sell_tiers or meltdown_tiers might be JSON
                if k in ["partial_sell_tiers","meltdown_tiers"]:
                    try:
                        merged[k] = json.loads(current[k]) if isinstance(current[k], str) else current[k]
                    except:
                        merged[k] = []
                else:
                    # parse as float
                    merged[k] = float(current[k]) if self._is_numeric(current[k]) else current[k]
            else:
                # fallback if not present
                if k=="partial_sell_tiers":
                    merged[k] = [{"gain":0.10,"ratio":0.05},{"gain":0.20,"ratio":0.10}]
                elif k=="dip_buy_pct_drop":
                    merged[k] = 0.05
                elif k=="rsi_overbought":
                    merged[k] = 70
                elif k=="bollinger_period":
                    merged[k] = 20
                elif k=="bollinger_stddev":
                    merged[k] = 2.0
                elif k=="momentum_rsi":
                    merged[k] = 60
                elif k=="momentum_fraction":
                    merged[k] = 0.05
                elif k=="trailing_stop_pct":
                    merged[k] = 0.10
                elif k=="meltdown_threshold":
                    merged[k] = 0.30
                elif k=="meltdown_anchor_buffer":
                    merged[k] = 0.02
                elif k=="meltdown_reentry_pct":
                    merged[k] = 0.10
                elif k=="meltdown_tiers":
                    merged[k] = [
                        {"drawdown":0.20,"sell_ratio":0.25},
                        {"drawdown":0.30,"sell_ratio":0.25},
                        {"drawdown":0.40,"sell_ratio":0.50}
                    ]
        return merged

    def _is_numeric(self, val):
        try:
            float(val)
            return True
        except:
            return False

    # ------------------------------------------------------------------
    # Load/Store
    # ------------------------------------------------------------------
    def _load_current_params(self):
        q = "SELECT param_name, param_value FROM meta_parameters"
        self.cursor.execute(q)
        rows = self.cursor.fetchall()
        pm = {r[0]: r[1] for r in rows}
        return pm

    def _store_params(self, param_dict):
        now_ = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for k,v in param_dict.items():
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v)
            else:
                v_str = str(v)
            q= """REPLACE INTO meta_parameters (param_name, param_value, last_updated)
                  VALUES (%s, %s, %s)"""
            self.cursor.execute(q, (k, v_str, now_))
        self.db.commit()

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------
    def _pick_coins_for_rebalance(self, limit=10):
        """
        Example of picking coins for rebalancing with a 
        fundamental + volatility weighting approach.
        Returns JSON with coin => weight
        """
        q = """
        SELECT f1.coin,
               f1.metric_value AS score_val,
               f2.metric_value AS vol_val
        FROM fundamentals f1
        JOIN fundamentals f2 ON (f1.coin = f2.coin)
        WHERE f1.metric_name='score'
          AND f2.metric_name='volatility_30d'
        ORDER BY f1.metric_value DESC
        LIMIT %s
        """
        self.cursor.execute(q, (limit,))
        rows = self.cursor.fetchall()

        weight_map = {}
        total_factor = 0.0

        for row in rows:
            coin     = row[0]
            score    = float(row[1])
            volatility = float(row[2])
            if score <= 0.0:
                continue

            eps = 1e-8
            if volatility< eps:
                volatility = eps

            factor = (score / 100.0) * (1.0 / volatility)
            if factor>0:
                weight_map[coin] = factor
                total_factor += factor

        if not weight_map or total_factor<=0:
            return "{}"

        final_allocs = {}
        for c, f in weight_map.items():
            final_allocs[c] = f / total_factor

        import json
        return json.dumps(final_allocs)
