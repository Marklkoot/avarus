"""
ai_signals_margin.py

A refactored version incorporating:
 - A single FEATURE_COLS list of 12 features (including RSI)
 - Force removal of extra columns (prev_close, etc.)
 - Feature scaling (StandardScaler per coin)
 - Time-based (chronological) train/validation split
 - Early stopping
 - Dropout in LSTMs & MLP
 - Weighted multi-task loss
 - 2 LSTMs + 1 MLP ensemble
 - Advanced Hyperparameter Tuning with Optuna
 - Automatic storage of best hyperparams for inference
 - *** Now predicting returns instead of raw next-close! ***
"""

import os
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from decimal import Decimal

# For feature scaling
from sklearn.preprocessing import StandardScaler
import joblib

# Optuna for hyperparameter search
import optuna

########################
# GLOBAL FEATURE LIST
########################
FEATURE_COLS = [
    "close",
    "volume",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "boll_up",
    "boll_down",
    "atr",
    "ema_short",
    "ema_long",
    "volatility"
]
# If you do NOT want RSI, remove it and reduce input_size to 11.


########################
# (A) Feature Computations
########################

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, slow=26, fast=12, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_bollinger(series, period=20, stddev=2.0):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = ma + stddev * std
    lower_band = ma - stddev * std
    return upper_band, lower_band

def compute_atr(df, period=14):
    df['prev_close'] = df['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_pc'] = (df['high'] - df['prev_close']).abs()
    df['low_pc'] = (df['low'] - df['prev_close']).abs()
    df['TR'] = df[['high_low','high_pc','low_pc']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df['ATR']

def compute_ema(series, period=10):
    return series.ewm(span=period, adjust=False).mean()

def compute_volatility(series, period=14):
    return series.pct_change().rolling(window=period).std()


########################
# (B) Dataset Definition
########################

class MultiFeatureDatasetMargin(Dataset):
    """
    Takes a dataframe with scaled features + 'target_return',
    creates sequences of length seq_len (X) plus next value (y).
    """
    def __init__(self, df, seq_len=20):
        self.seq_len = seq_len
        # The last column is now 'target_return'
        # The rest are scaled features
        self.X_data = df.drop(columns=["target_return"]).values
        self.y_data = df["target_return"].values
        self.valid_length = len(self.X_data) - self.seq_len

    def __len__(self):
        return max(self.valid_length, 0)

    def __getitem__(self, idx):
        X_seq = self.X_data[idx : idx + self.seq_len]  # (seq_len, #features)
        y_val = self.y_data[idx + self.seq_len]        # float => the next return
        return (
            torch.tensor(X_seq, dtype=torch.float),
            torch.tensor(y_val, dtype=torch.float)
        )


########################
# (C) Models for Ensemble
########################

class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq_len, hidden_size)
        out = out[:, -1, :]          # last time step
        out = self.fc(out)           # (batch,1)
        return out

class MLPModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, dropout=0.2):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # For MLP, we only use last time step's features
        x = x[:, -1, :]  # (batch, input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc3(x)  # (batch,1)
        return out


########################
# (D) Ensemble: combine 2 LSTMs + 1 MLP
########################

class EnsembleMargin(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, lstm_layers=2, dropout=0.2):
        super(EnsembleMargin, self).__init__()
        self.lstm1 = LSTMModel(input_size, hidden_size, num_layers=lstm_layers, dropout=dropout)
        self.lstm2 = LSTMModel(input_size, hidden_size, num_layers=lstm_layers, dropout=dropout)
        self.mlp   = MLPModel(input_size, hidden_size, dropout=dropout)

        # aggregator: (3 => 2) => predicted_return + confidence
        self.fc_agg = nn.Linear(3, 2)

    def forward(self, x):
        out1 = self.lstm1(x)  # (batch,1) => predicted return #1
        out2 = self.lstm2(x)  # (batch,1) => predicted return #2
        out3 = self.mlp(x)    # (batch,1) => predicted return #3

        combined = torch.cat([out1, out2, out3], dim=1)  # (batch,3)
        final = self.fc_agg(combined)                    # (batch,2)

        # first => predicted_return, second => raw confidence => pass through sigmoid
        pred_return = final[:, 0]               # shape => (batch,)
        confidence  = torch.sigmoid(final[:, 1])# shape => (batch,)
        return pred_return.unsqueeze(1), confidence.unsqueeze(1)


########################
# (E) AIPricePredictorMargin
########################

class AIPricePredictorMargin:
    """
    - Computes features + scaling
    - Creates train/val datasets (time-based split)
    - Trains an ensemble model (2 LSTMs + 1 MLP) with MSE + BCE
    - Saves best model + scaler
    - Predicts next close (converted from predicted return) + confidence
    - Generates trading signal
    - Includes Optuna-based hyperparameter tuning
    - Stores best hyperparams in JSON for consistent inference
    """

    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model_dir = config.get("ai_models_margin_dir", "ai_models_margin")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # We have 12 columns in FEATURE_COLS
        self.input_size = len(FEATURE_COLS)

    ##################################################
    # E1) Utility: Compute + Scale Features
    ##################################################

    def _compute_features(self, df_raw):
        df = df_raw.copy()

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=numeric_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # TA indicators
        df["rsi"] = compute_rsi(df["close"], period=14)
        macd_line, macd_signal, macd_hist = compute_macd(df["close"])
        df["macd_line"]   = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"]   = macd_hist

        upper_b, lower_b  = compute_bollinger(df["close"], period=20, stddev=2.0)
        df["boll_up"]   = upper_b
        df["boll_down"] = lower_b

        df["atr"]       = compute_atr(df, period=14)
        df["ema_short"] = compute_ema(df["close"], period=10)
        df["ema_long"]  = compute_ema(df["close"], period=50)
        df["volatility"]= compute_volatility(df["close"], period=14)

        # ***** CHANGED: We now produce 'target_return' instead of 'target_close' *****
        df["target_return"] = (df["close"].shift(-1) / df["close"]) - 1.0

        final_cols = FEATURE_COLS + ["target_return"]
        df = df[final_cols]

        for c in final_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.dropna(subset=final_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _scale_features(self, coin, df_feat, fit_new_scaler=True):
        scaler_path = os.path.join(self.model_dir, f"scaler_{coin}.pkl")

        for col in FEATURE_COLS:
            df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

        df_feat.dropna(subset=FEATURE_COLS, inplace=True)
        df_feat.reset_index(drop=True, inplace=True)
        df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].astype("float32", copy=False)

        if fit_new_scaler:
            scaler = StandardScaler()
            scaler.fit(df_feat[FEATURE_COLS])
            joblib.dump(scaler, scaler_path)
        else:
            if not os.path.exists(scaler_path):
                scaler = StandardScaler()
                scaler.fit(df_feat[FEATURE_COLS])
            else:
                scaler = joblib.load(scaler_path)

        df_scaled = df_feat.copy()
        df_scaled[FEATURE_COLS] = scaler.transform(df_feat[FEATURE_COLS])
        return df_scaled

    ##################################################
    # E2) Base Training (time-based split, default hparams)
    ##################################################

    def train_model(self, coin, df_raw, seq_len=20, epochs=30, val_fraction=0.2, patience=3):
        """
        Trains with default hyperparams => hidden_size=32, dropout=0.2, bce_weight=0.05, lr=0.001
        """
        return self._train_model_with_params(
            coin=coin,
            df_raw=df_raw,
            seq_len=seq_len,
            epochs=epochs,
            val_fraction=val_fraction,
            patience=patience,
            hidden_size=32,
            lstm_layers=2,
            dropout=0.2,
            bce_weight=0.05,
            lr=0.001
        )

    ##################################################
    # E3) Training with Provided Hyperparams
    ##################################################

    def _train_model_with_params(self, coin, df_raw,
                                 seq_len=20, epochs=30,
                                 val_fraction=0.2, patience=3,
                                 hidden_size=32, lstm_layers=2,
                                 dropout=0.2, bce_weight=0.05, lr=0.001):
        """
        A flexible version of train_model that allows direct hyperparam overrides.
        Returns the best validation loss found, or None if training was skipped.
        Also saves best model + best hyperparams (JSON) to disk.
        """
        # 1) Compute raw features as before
        df_feat = self._compute_features(df_raw)
        if len(df_feat) < seq_len * 2:
            self.logger.warning(f"[MarginAITrain] Not enough data => skip => {coin}")
            return None

        # 2) Time-based split (or proportion-based in chronological order)
        total_len = len(df_feat)
        train_size = int((1.0 - val_fraction) * total_len)
        if train_size < seq_len:
            self.logger.warning(f"[MarginAITrain] Not enough training samples => skip => {coin}")
            return None

        # Take the first `train_size` rows as training, the rest as validation
        df_train = df_feat.iloc[:train_size].copy()
        df_val   = df_feat.iloc[train_size:].copy()

        # 3) Fit scaler on TRAIN portion ONLY
        from sklearn.preprocessing import StandardScaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{coin}.pkl")

        scaler = StandardScaler()
        scaler.fit(df_train[FEATURE_COLS])  # NO LEAKAGE: only fit on train

        # 4) Transform train & val using that same scaler
        df_train_scaled = df_train.copy()
        df_val_scaled   = df_val.copy()

        df_train_scaled[FEATURE_COLS] = scaler.transform(df_train_scaled[FEATURE_COLS])
        df_val_scaled[FEATURE_COLS]   = scaler.transform(df_val_scaled[FEATURE_COLS])

        # Save the fitted scaler to disk
        import joblib
        joblib.dump(scaler, scaler_path)

        # 5) Create separate Datasets for train & val
        train_dataset = MultiFeatureDatasetMargin(df_train_scaled, seq_len=seq_len)
        val_dataset   = MultiFeatureDatasetMargin(df_val_scaled,   seq_len=seq_len)

        # Check sizes
        if len(train_dataset) < 2:
            self.logger.warning(f"[MarginAITrain] Not enough train sequences => skip => {coin}")
            return None
        if len(val_dataset) < 1:
            self.logger.warning(f"[MarginAITrain] Not enough val sequences => skip => {coin}")
            return None

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
        val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnsembleMargin(
            input_size=self.input_size,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout
        ).to(device)

        # We'll do MSE on predicted_return vs actual_return
        criterion_main = nn.MSELoss()
        # We'll do BCE on "is return > 0"
        criterion_conf = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_epoch = 0
        no_improvement_count = 0

        model_path = os.path.join(self.model_dir, f"ai_{coin}_ensemble.pt")
        params_path= os.path.join(self.model_dir, f"best_params_{coin}.json")

        for ep in range(epochs):
            # TRAIN
            model.train()
            total_train_loss = 0.0

            for X_seq, y_val in train_loader:
                X_seq = X_seq.to(device)
                y_val = y_val.to(device)

                # pred_return, pred_conf ~ the model outputs
                pred_return, pred_conf = model(X_seq)

                # label_up => whether the actual return is > 0
                label_up = (y_val > 0).float()

                # main loss => MSE on returns
                loss_main = criterion_main(pred_return.squeeze(-1), y_val)
                # conf loss => BCE
                loss_conf = criterion_conf(pred_conf.squeeze(-1), label_up)
                loss = loss_main + bce_weight * loss_conf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # VALIDATION
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for X_seq, y_val in val_loader:
                    X_seq = X_seq.to(device)
                    y_val = y_val.to(device)

                    pred_return, pred_conf = model(X_seq)
                    label_up = (y_val > 0).float()

                    loss_main = criterion_main(pred_return.squeeze(-1), y_val)
                    loss_conf = criterion_conf(pred_conf.squeeze(-1), label_up)
                    val_loss  = loss_main + bce_weight * loss_conf
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            self.logger.info(
                f"[MarginAITrain] {coin} => ep={ep}, "
                f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, "
                f"hsize={hidden_size}, layers={lstm_layers}, drop={dropout}, bce={bce_weight}, lr={lr}"
            )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = ep
                no_improvement_count = 0

                # Save best model
                torch.save(model.state_dict(), model_path)

                # Also save best hyperparams (JSON)
                best_params_dict = {
                    "hidden_size": hidden_size,
                    "lstm_layers": lstm_layers,
                    "dropout": dropout,
                    "bce_weight": bce_weight,
                    "lr": lr,
                    "seq_len": seq_len
                }
                with open(params_path, 'w') as fp:
                    json.dump(best_params_dict, fp)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    self.logger.info(f"[MarginAITrain] Early stopping at epoch={ep} for {coin}")
                    break

        self.logger.info(
            f"[MarginAITrain] {coin} => best_val_loss={best_val_loss:.6f} at ep={best_epoch}, "
            f"hsize={hidden_size}, layers={lstm_layers}, drop={dropout}, bce={bce_weight}, lr={lr}"
        )
        return best_val_loss


    ##################################################
    # E4) Optuna Hyperparam Tuning
    ##################################################

    def hyperparam_optuna_search(self, coin, df_raw, seq_len=20, epochs=30, val_fraction=0.2, patience=3, n_trials=10):
        """
        Example usage:
            predictor.hyperparam_optuna_search("BTC", df_btc, seq_len=20, epochs=30, n_trials=10)
        This will:
         - Use Optuna to do n_trials
         - Each trial picks hidden_size, dropout, bce_weight, lr, ...
         - Calls _train_model_with_params
         - Minimizes val_loss
        """
        df_feat = self._compute_features(df_raw)
        if len(df_feat) < seq_len * 2:
            self.logger.warning(f"[MarginAITrain:Optuna] Not enough data => skip => {coin}")
            return None, None

        # We'll do one initial scale just to ensure we can scale, but each trial re-scales with fit_new_scaler=True.
        df_scaled = self._scale_features(coin, df_feat, fit_new_scaler=True)
        if len(df_scaled) < seq_len + 1:
            self.logger.warning(f"[MarginAITrain:Optuna] Not enough data after scaling => skip => {coin}")
            return None, None

        dataset = MultiFeatureDatasetMargin(df_scaled, seq_len=seq_len)
        if len(dataset) < 2:
            self.logger.warning(f"[MarginAITrain:Optuna] Not enough sequences => skip => {coin}")
            return None, None

        def objective(trial):
            # We'll pick hyperparams from some search space
            hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
            lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
            dropout     = trial.suggest_float("dropout", 0.0, 0.5)
            bce_weight  = trial.suggest_float("bce_weight", 0.01, 0.1)
            lr          = trial.suggest_loguniform("lr", 1e-4, 1e-2)

            val_loss = self._train_model_with_params(
                coin=coin,
                df_raw=df_raw,           # same raw
                seq_len=seq_len,
                epochs=epochs,
                val_fraction=val_fraction,
                patience=patience,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                dropout=dropout,
                bce_weight=bce_weight,
                lr=lr
            )
            # If train was skipped, return large value
            if val_loss is None:
                return 9999
            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        best_val_loss = best_trial.value
        self.logger.info(f"[OptunaSearch] Best trial => val_loss={best_val_loss}, params={best_trial.params}")

        return best_trial.params, best_val_loss

    ##################################################
    # E5) Inference + Signal
    ##################################################

    def predict_next_close_and_conf(self, coin, df_raw, seq_len=20):
        """
        Loads saved model & best_params from JSON. Predict next RETURN + confidence
        from last seq_len of data, then we convert the predicted return -> next close.
        """
        model_path = os.path.join(self.model_dir, f"ai_{coin}_ensemble.pt")
        params_path= os.path.join(self.model_dir, f"best_params_{coin}.json")
        if not os.path.exists(model_path):
            self.logger.warning(f"[MarginAIPredict] no model => {model_path}")
            return None, None

        df_feat = self._compute_features(df_raw)
        if len(df_feat) < seq_len:
            return None, None

        # Scale with existing scaler
        df_scaled = self._scale_features(coin, df_feat, fit_new_scaler=False)
        sub_df = df_scaled.iloc[-seq_len:].copy()
        X_seq = sub_df[FEATURE_COLS].values
        X_seq_t = torch.tensor(X_seq, dtype=torch.float).unsqueeze(0)  # (1, seq_len, #features)

        # Load best hyperparams from JSON (or fallback to defaults if missing)
        if os.path.exists(params_path):
            with open(params_path, 'r') as fp:
                best_params = json.load(fp)
        else:
            # fallback if no JSON => use defaults
            best_params = {
                "hidden_size": 32,
                "lstm_layers": 2,
                "dropout": 0.2
            }

        hidden_size = best_params.get("hidden_size", 32)
        lstm_layers = best_params.get("lstm_layers", 2)
        dropout     = best_params.get("dropout", 0.2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnsembleMargin(
            input_size=self.input_size,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout
        ).to(device)

        # load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            # Here, pred_return is what's predicted; we convert it to a next-close guess
            pred_return, conf = model(X_seq_t.to(device))

        # Convert the predicted return -> a predicted close
        last_close = float(df_raw["close"].iloc[-1])
        predicted_close = last_close * (1.0 + pred_return.item())

        return predicted_close, conf.item()

    def generate_signal(self, coin, df_raw, seq_len=20, threshold_pct=0.02):
        """
        Simple rule-based signal generation using predicted next_close vs last_close
        plus confidence. If difference ratio > threshold_pct => buy, < -threshold_pct => sell.
        Also checks if confidence > 0.8 => "strong".
        """
        if len(df_raw) < 1:
            return "hold"

        last_close = float(df_raw["close"].iloc[-1])
        pred_close, confidence = self.predict_next_close_and_conf(coin, df_raw, seq_len)
        if pred_close is None:
            return "hold"

        diff_ratio = (pred_close - last_close) / (last_close if last_close != 0 else 1.0)

        if diff_ratio > threshold_pct:
            return "buy_strong" if confidence > 0.8 else "buy"
        elif diff_ratio < -threshold_pct:
            return "sell_strong" if confidence > 0.8 else "sell"
        else:
            return "hold"
