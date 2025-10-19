"""Trading agent: loads a pickled model, scores incoming data, and updates a CSV-backed portfolio."""
from typing import List, Optional
import os
import pickle
import pandas as pd
from datetime import datetime

from .data_loading import get_polygon_data, fetch_all


class TradingAgent:
    def __init__(self, model_path: Optional[str] = None, portfolio_csv: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        self.model_path = model_path
        self.model = self._load_model()

        if portfolio_csv is None:
            portfolio_csv = os.path.join(os.path.dirname(__file__), "portfolio.csv")
        self.portfolio_csv = portfolio_csv
        self.portfolio = self._load_portfolio()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def _load_portfolio(self) -> pd.DataFrame:
        if os.path.exists(self.portfolio_csv):
            df = pd.read_csv(self.portfolio_csv, parse_dates=["date"]) 
        else:
            df = pd.DataFrame(columns=["date", "Ticker", "Shares", "Cash"]) 
            df.to_csv(self.portfolio_csv, index=False)
        return df

    def _save_portfolio(self):
        self.portfolio.to_csv(self.portfolio_csv, index=False)

    def score_and_update(self, tickers: List[str], date: str, features_df: pd.DataFrame, feature_cols: List[str], buy_threshold: float = 0.55, sell_threshold: float = 0.45):
        """Score today's features and update the portfolio CSV.

        - tickers: list of tickers to consider
        - date: date string for the row (YYYY-MM-DD or datetime)
        - features_df: DataFrame with rows corresponding to tickers and columns feature_cols
        """
        X = features_df[feature_cols]
        probs = self.model.predict_proba(X)[:, 1]
        decisions = dict(zip(features_df["Ticker"], probs))

        # Simple portfolio bookkeeping: one row per action
        for t in tickers:
            prob = float(decisions.get(t, 0))
            # Current holdings
            holdings = self.portfolio[(self.portfolio["Ticker"] == t) & (self.portfolio["date"] <= pd.to_datetime(date))]
            current_shares = int(holdings["Shares"].iloc[-1]) if not holdings.empty else 0
            cash = float(holdings["Cash"].iloc[-1]) if not holdings.empty else 0.0

            # For simplicity: buy 1 share if prob > buy_threshold, sell all if prob < sell_threshold
            if prob > buy_threshold:
                current_shares += 1
                # cash impact is not known here (price not supplied); user should provide later integration
            elif prob < sell_threshold and current_shares > 0:
                current_shares = 0

            new_row = {"date": pd.to_datetime(date), "Ticker": t, "Shares": current_shares, "Cash": cash}
            self.portfolio = pd.concat([self.portfolio, pd.DataFrame([new_row])], ignore_index=True)

        self._save_portfolio()

    def run_on_market(self, tickers: List[str], start_date: str, end_date: str, feature_cols: List[str]):
        """Fetch data for tickers between start_date and end_date, compute features externally, and run scoring.

        This implementation expects caller to pass a features DataFrame for each day. For convenience the method will
        fetch raw OHLCV and raise if features are missing; the user can reuse `modeling.add_technical_features` to
        prepare the data in advance.
        """
        data = fetch_all(tickers, start_date, end_date)
        if data.empty:
            return
        # Expect data already has features and 'Ticker' column
        for date, group in data.groupby("timestamp"):
            self.score_and_update(tickers, date, group, feature_cols)
