# ml_engine/ai_signals.py

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from decimal import Decimal
from torch.utils.data import Dataset, DataLoader

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta>0, 0.0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta<0, 0.0)).ewm(alpha=1/period).mean()
    rs = gain/(loss+1e-9)
    rsi = 100 - (100/(1+rs))
    return rsi

class MultiFeatureDataset(Dataset):
    """
    Each item => (X,y)
      X => (seq_len, num_features)
      y => next close
    We expect columns: [close, volume, rsi, target_close].
    """
    def __init__(self, df, seq_len=20):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.df = self.df.dropna().reset_index(drop=True)

        self.X_data = self.df.drop(columns=["target_close"]).values
        self.y_data = self.df["target_close"].values

    def __len__(self):
        return len(self.X_data) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X_data[idx : idx + self.seq_len]
        y_val = self.y_data[idx + self.seq_len]
        return (
            torch.tensor(X_seq, dtype=torch.float),
            torch.tensor(y_val, dtype=torch.float)
        )

class MultiFeatureLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=16):
        super(MultiFeatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # x => (batch,seq_len,input_size)
        out = out[:, -1, :]    # last time step => (batch, hidden_size)
        out = self.fc(out)
        return out

class AIPricePredictor:
    """
    Minimal LSTM for [close, volume, rsi].
    By default, train with seq_len=20, epochs=10.
    """

    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_dir = config.get("ai_models_dir","ai_models")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

def compute_rsi(series, period=14):
    # same as your existing RSI calculation
    delta = series.diff()
    gain = (delta.where(delta>0, 0.0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta<0, 0.0)).ewm(alpha=1/period).mean()
    rs = gain/(loss+1e-9)
    rsi = 100 - (100/(1+rs))
    return rsi

class MultiFeatureDataset(torch.utils.data.Dataset):
    """
    Each item => (X, y)
      X => (seq_len, num_features)
      y => next close
    We expect columns: [close, volume, rsi, target_close].
    """
    def __init__(self, df, seq_len=20):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.df = self.df.dropna().reset_index(drop=True)

        self.X_data = self.df.drop(columns=["target_close"]).values
        self.y_data = self.df["target_close"].values

    def __len__(self):
        return len(self.X_data) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X_data[idx : idx + self.seq_len]
        y_val = self.y_data[idx + self.seq_len]
        return (
            torch.tensor(X_seq, dtype=torch.float),
            torch.tensor(y_val, dtype=torch.float)
        )

class MultiFeatureLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=16):
        super(MultiFeatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)     # x => (batch, seq_len, input_size)
        out = out[:, -1, :]       # last time step => (batch, hidden_size)
        out = self.fc(out)        # => (batch, 1)
        return out

class AIPricePredictor:
    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_dir = config.get("ai_models_dir", "ai_models")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train_model(self, coin, df, seq_len=20, epochs=10):
        if len(df) < seq_len * 2:
            self.logger.warning(f"[AI] {coin} => Not enough data => skipping training.")
            return

        # compute RSI
        df["rsi"] = compute_rsi(df["close"], period=14)

        # shift to get target_close
        df["target_close"] = df["close"].shift(-1)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Force float
        for c in ["close","volume","boll_up","boll_down",
                  "macd","macd_signal","macd_diff","ema_10","ema_50","rsi","target_close"]:
            if c in df.columns:
                df[c] = df[c].astype(float)

        # Build dataset
        features = df[["close","volume","rsi"]]
        dataset_df = features.copy()
        dataset_df["target_close"] = df["target_close"]
        dataset = MultiFeatureDataset(dataset_df, seq_len=seq_len)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # 1) DETECT GPU vs. CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) CREATE MODEL ON DEVICE
        model = MultiFeatureLSTM(input_size=3, hidden_size=16).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for ep in range(epochs):
            for X_seq, y_val in loader:
                # 3) MOVE BATCH TO GPU
                X_seq = X_seq.to(device)
                y_val = y_val.to(device)

                pred = model(X_seq).squeeze(-1)
                loss = criterion(pred, y_val)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.info(f"[AITrain] {coin} => ep={ep}, loss={loss.item():.6f}")

        model_path = os.path.join(self.model_dir, f"ai_{coin}.pt")
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"[AITrain] {coin} => saved model => {model_path}")

    def predict_next_close(self, coin, df, seq_len=20):
        """
        Load model from disk, feed the last seq_len rows => 
        [close, volume, rsi].
        Returns float or None.
        """
        import os

        model_path = os.path.join(self.model_dir, f"ai_{coin}.pt")
        if not os.path.exists(model_path):
            self.logger.warning(f"[AIPredict] {coin} => model file not found => {model_path}")
            return None

        if len(df) < seq_len:
            self.logger.warning(f"[AIPredict] {coin} => not enough rows => need seq_len={seq_len}")
            return None

        df["rsi"] = compute_rsi(df["close"], period=14)
        feature_cols = ["close","volume","rsi"]

        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=feature_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)

        if len(df) < seq_len:
            self.logger.warning(
                f"[AIPredict] {coin} => not enough rows after dropna => have {len(df)}, need {seq_len}"
            )
            return None

        sub_df = df.iloc[-seq_len:].copy()
        sub_df.ffill(inplace=True)
        sub_df.bfill(inplace=True)

        X_seq = sub_df[feature_cols].values
        X_seq_t = torch.tensor(X_seq, dtype=torch.float).unsqueeze(0)

        # DETECT DEVICE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CREATE MODEL ON DEVICE
        model = MultiFeatureLSTM(input_size=3, hidden_size=16).to(device)

        # LOAD WEIGHTS ONTO DEVICE
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        X_seq_t = X_seq_t.to(device)

        with torch.no_grad():
            out = model(X_seq_t)
        return out.item()


    def generate_signal(self, coin, df, seq_len=20, threshold_pct=0.01):
        """
        If (pred-last_close)/last_close > threshold => 'buy'
           < -threshold => 'sell'
           else => 'hold'
        """
        pred = self.predict_next_close(coin, df, seq_len=seq_len)
        if pred is None:
            return "hold"
        last_close = df["close"].iloc[-1]
        if last_close<=0: 
            return "hold"

        diff_ratio = (pred - last_close)/ last_close
        if diff_ratio> threshold_pct:
            return "buy"
        elif diff_ratio< -threshold_pct:
            return "sell"
        else:
            return "hold"
