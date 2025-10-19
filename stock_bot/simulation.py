"""Simulation module: fetch data, create features, train RandomForest, run simulations and baselines.

This is a mostly self-contained translation of the notebook cell logic into functions.
"""
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt

from .data_loading import get_polygon_data, fetch_all
from .modeling import add_technical_features
from .trading import TradingAgent


class Simulator:
    def __init__(self, tickers: List[str], start_date: str, end_date: str, initial_capital: float = 100_000.0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

    def fetch_all(self) -> pd.DataFrame:
        frames = []
        for t in self.tickers:
            df = get_polygon_data(t, self.start_date, self.end_date)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        return data

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        data = add_technical_features(data)
        return data

    def simulate_equal_weight(self, data: pd.DataFrame, model, features: List[str]):
        capital = self.initial_capital
        positions = {t: 0 for t in self.tickers}
        portfolio_values = []

        for date, group in data.groupby("timestamp"):
            X_day = group[features]
            probs = model.predict_proba(X_day)[:, 1]
            decisions = dict(zip(group["Ticker"], probs))

            equity_value = sum(
                positions[t] * group[group["Ticker"] == t]["Close"].values[0] for t in self.tickers
            )
            total_value = capital + equity_value

            bullish = [t for t in self.tickers if decisions.get(t, 0) > 0.55]
            if bullish:
                invest_per_stock = total_value / len(bullish)
                for t in self.tickers:
                    price = group[group["Ticker"] == t]["Close"].values[0]
                    if t in bullish:
                        target_value = invest_per_stock
                        current_value = positions[t] * price
                        if current_value < target_value:
                            shares_to_buy = int((target_value - current_value) / price)
                            cost = shares_to_buy * price
                            if cost <= capital and shares_to_buy > 0:
                                positions[t] += shares_to_buy
                                capital -= cost
                    else:
                        if positions[t] > 0:
                            capital += positions[t] * price
                            positions[t] = 0

            portfolio_values.append({"Date": date, "TotalValue": total_value, "Cash": capital})

        return pd.DataFrame(portfolio_values)

    def simulate_share_counts(self, data: pd.DataFrame, model, features: List[str], allocation_per_trade: float = 10_000):
        capital = self.initial_capital
        positions = {t: 0 for t in self.tickers}
        portfolio_value = []

        for date, group in data.groupby("timestamp"):
            if len(group) < len(self.tickers):
                continue
            X_day = group[features]
            probs = model.predict_proba(X_day)[:, 1]
            decisions = dict(zip(group["Ticker"], probs))

            for t in self.tickers:
                price = group[group["Ticker"] == t]["Close"].values[0]
                prob = decisions.get(t, 0)
                if prob > 0.55 and capital > price:
                    num_shares = int(allocation_per_trade // price)
                    cost = num_shares * price
                    if num_shares > 0 and capital >= cost:
                        positions[t] += num_shares
                        capital -= cost
                elif prob < 0.45 and positions[t] > 0:
                    proceeds = positions[t] * price
                    capital += proceeds
                    positions[t] = 0

            total_value = capital + sum(
                positions[t] * group[group["Ticker"] == t]["Close"].values[0] for t in self.tickers
            )
            portfolio_value.append({"date": date, "value": total_value})

        return pd.DataFrame(portfolio_value)

    def buy_and_hold_baseline(self, data: pd.DataFrame):
        baseline_values = {}
        for t in self.tickers:
            df_t = data[data["Ticker"] == t].copy().sort_values("timestamp")
            if df_t.empty:
                continue
            initial_price = df_t["Close"].iloc[0]
            shares = self.initial_capital / initial_price
            df_t["Portfolio_Value"] = shares * df_t["Close"]
            baseline_values[t] = df_t
        return baseline_values

    def plot_portfolio(self, portfolio_df: pd.DataFrame, baseline_values: Optional[Dict[str, pd.DataFrame]] = None, title: str = "Portfolio"):
        plt.figure(figsize=(10, 5))
        if baseline_values:
            for t, df_t in baseline_values.items():
                plt.plot(df_t["timestamp"], df_t["Portfolio_Value"], label=f"Buy & Hold {t}")
        plt.plot(portfolio_df.iloc[:, 0], portfolio_df.iloc[:, 1], color="black", linewidth=2, label="Strategy")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def backtest_with_agent(self, data: pd.DataFrame, feature_cols: List[str], agent: Optional[TradingAgent] = None, allocation_per_trade: float = 10_000) -> pd.DataFrame:
        """Backtest using the supplied TradingAgent (or create one if None).

        This routine will iterate over each date in `data` (grouped by 'timestamp'),
        use the agent's model to predict probabilities, execute buys/sells using
        the day's Close price, and return a history DataFrame with Date, TotalValue, Cash.

        - data: DataFrame with features and 'Close' and 'Ticker' columns
        - feature_cols: list of columns used by the model
        - agent: optional TradingAgent instance; if None a new one is created and expected to have a pickled model present
        - allocation_per_trade: dollar allocation to use when buying shares (per signal)
        """
        if agent is None:
            agent = TradingAgent()

        capital = self.initial_capital
        positions = {t: 0 for t in self.tickers}
        history = []

        # Ensure data sorted by timestamp
        data = data.sort_values("timestamp")

        for date, group in data.groupby("timestamp"):
            # Only price/account for tickers present that day
            # Prepare features for model
            X_day = group[feature_cols]
            probs = agent.model.predict_proba(X_day)[:, 1]
            decisions = dict(zip(group["Ticker"], probs))

            # Execute trades: buy fixed-dollar allocation when prob>0.55, sell when prob<0.45
            for t in self.tickers:
                row = group[group["Ticker"] == t]
                if row.empty:
                    continue
                price = float(row["Close"].values[0])
                prob = float(decisions.get(t, 0))

                if prob > 0.55:
                    num_shares = int(allocation_per_trade // price)
                    cost = num_shares * price
                    if num_shares > 0 and capital >= cost:
                        positions[t] += num_shares
                        capital -= cost
                elif prob < 0.45 and positions[t] > 0:
                    proceeds = positions[t] * price
                    capital += proceeds
                    positions[t] = 0

            total_value = capital + sum(positions[t] * float(group[group["Ticker"] == t]["Close"].values[0]) for t in self.tickers if not group[group["Ticker"] == t].empty)
            history.append({"Date": date, "TotalValue": total_value, "Cash": capital})

            # Append snapshot rows to agent portfolio CSV for traceability
            # We create one row per ticker showing holdings after today's trades
            for t in self.tickers:
                row = group[group["Ticker"] == t]
                if row.empty:
                    continue
                price = float(row["Close"].values[0])
                snapshot = {"date": pd.to_datetime(date), "Ticker": t, "Shares": positions[t], "Cash": capital}
                agent.portfolio = pd.concat([agent.portfolio, pd.DataFrame([snapshot])], ignore_index=True)
            agent._save_portfolio()

        return pd.DataFrame(history)
