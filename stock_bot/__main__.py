"""Quick CLI to exercise the stock_bot Simulator without running network calls by default."""
import os
from . import Simulator


def main():
    api_key = os.environ.get("POLYGON_API_KEY", "")
    sim = Simulator(api_key=api_key, tickers=["NVDA", "META", "MSFT"], start_date="2024-09-01", end_date="2024-10-15")
    print("Created Simulator:", sim)
    print("Methods: fetch_all, prepare, train, simulate_equal_weight, simulate_share_counts, buy_and_hold_baseline")


if __name__ == "__main__":
    main()
