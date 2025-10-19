"""Modeling helpers for training classifiers used by the stock simulator."""
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from typing import Tuple, Optional
import os

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["Ticker", "timestamp"]) if "Ticker" in df.columns else df.sort_values("timestamp")
    df["Return"] = df.groupby("Ticker")["Close"].pct_change()
    df["MA5"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(5).mean())
    df["MA10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(10).mean())
    df["Volatility"] = df.groupby("Ticker")["Return"].transform(lambda x: x.rolling(5).std())
    df["Future_Return"] = df.groupby("Ticker")["Close"].shift(-1) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] > 0).astype(int)
    return df.dropna()

def train_random_forest(X: pd.DataFrame, y: pd.Series, n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model


def train_model(data: pd.DataFrame, features: List[str], model_type: str = "random_forest", output_path: Optional[str] = None, test_frac: float = 0.2, random_state: int = 42) -> Tuple[object, Optional[float], str]:
    """Train a model on the provided DataFrame and pickle the trained model.

    - data: DataFrame containing features and 'Target' column
    - features: list of column names to use as X
    - model_type: 'random_forest' or 'logistic'
    - output_path: where to save the pickled model (default: stock_bot/model.pkl)

    Returns (model, test_accuracy_or_None, path_to_pickle)
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    X = data[features]
    y = data["Target"]
    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == "random_forest":
        model = train_random_forest(X_train, y_train)
    elif model_type == "logistic":
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    acc = (model.predict(X_test) == y_test).mean() if len(X_test) else None

    # Persist the model
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    return model, acc, output_path
