"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

# --- FIX FOR NUMPY 2.0+ COMPATIBILITY WITH QUANTSTATS ---
# Quantstats relies on np.product which was removed in NumPy 2.0
if not hasattr(np, 'product'):
    np.product = np.prod
# --------------------------------------------------------

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=126, gamma=0):
        # Default lookback changed to 126 (approx 6 months) for robust momentum
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column (SPY)
        assets = self.price.columns[self.price.columns != self.exclude]
        
        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        lookback = self.lookback  # window for volatility estimation

        # If we have a benchmark (e.g. SPY), use it for a simple trend filter
        if self.exclude in self.price.columns:
            mkt_price = self.price[self.exclude]
            mkt_sma200 = mkt_price.rolling(200).mean()
        else:
            mkt_price = None
            mkt_sma200 = pd.Series(index=self.price.index, data=np.nan)

        for i, date in enumerate(self.price.index):
            # rolling window of past returns up to "date"
            # use at most `lookback` days, but if early in sample, just use whatever we have
            window_returns = self.returns[assets].iloc[max(0, i - lookback) : i]

            # need at least 2 observations to estimate volatility
            if len(window_returns) < 2:
                continue

            # daily volatility estimate for each asset
            vol = window_returns.std()

            # handle zero/NaN vol (replace with cross-sectional mean)
            vol = vol.replace(0, np.nan)
            if vol.isna().all():
                # pathological case: fall back to equal weights
                w_assets = pd.Series(1.0 / len(assets), index=assets)
            else:
                vol = vol.fillna(vol.mean())
                inv_vol = 1.0 / vol
                w_assets = inv_vol / inv_vol.sum()

            # --- Trend filter on SPY (or excluded asset) ---
            # when market is below its 200-day SMA, scale down risk
            scale = 1.0
            if mkt_price is not None:
                sma_val = mkt_sma200.loc[date]
                px_val = mkt_price.loc[date]
                if not np.isnan(sma_val) and px_val < sma_val:
                    # de-risk in downtrend; you can tweak 0.4 → 0.3/0.5 etc.
                    scale = 0.4

            # set weights for sector ETFs (assets)
            self.portfolio_weights.loc[date, assets] = (w_assets * scale).values

            # no allocation to the excluded asset (e.g. SPY) → remaining is "cash"
            if self.exclude in self.price.columns:
                self.portfolio_weights.loc[date, self.exclude] = 0.0
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)