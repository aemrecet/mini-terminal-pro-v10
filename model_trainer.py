import os, joblib, lightgbm as lgb
import pandas as pd, numpy as np
from data_fetcher import fetch_history

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['return'] = df['Close'].pct_change()
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['vol21'] = df['return'].rolling(21).std()
    df['mom5'] = df['Close'].pct_change(5)
    df = df.dropna()
    return df

def load_ensemble():
    if not os.path.isdir(MODEL_DIR):
        return []
    models = []
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            try:
                models.append(joblib.load(os.path.join(MODEL_DIR, fname)))
            except Exception:
                pass
    return models

def predict_proba_ensemble(models, X_row: pd.DataFrame) -> float:
    if not models:
        return None
    preds = [float(m.predict(X_row)[0]) for m in models]
    return float(np.mean(preds))
