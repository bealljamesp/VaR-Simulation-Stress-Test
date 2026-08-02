# VaR Simulation and Stress Test - Fully Encapsulated Portfolios

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


class PortfolioVaR:
    def __init__(
        self,
        name,
        tickers,
        weights,
        start_date,
        end_date,
        initial_capital=1_000_000,
        confidence_level=0.95,
    ):
        self.name = name
        self.tickers = tickers
        self.weights = np.array(weights)

        # Validate that weights sum to 1.0 upon instantiation
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-4):
            raise ValueError(
                f"Portfolio weights must sum to 1.0. Current sum for '{self.name}': {np.sum(self.weights)}"
            )

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.confidence_level = confidence_level

        self.market_data = None
        self.returns = None
        self.portfolio_returns = None

    def download_data(self):
        """Downloads historical market data for the portfolio tickers."""
        self.market_data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )
        return self.market_data

    def calculate_returns(self, method="log"):
        """Calculates historical asset returns."""
        if isinstance(self.market_data.columns, pd.MultiIndex):
            if "Adj Close" in self.market_data.columns.levels[0]:
                price_data = self.market_data["Adj Close"]
            else:
                price_data = self.market_data["Close"]
        else:
            price_data = (
                self.market_data[["Adj Close"]]
                if "Adj Close" in self.market_data.columns
                else self.market_data[["Close"]]
            )

        if method == "log":
            self.returns = np.log(price_data / price_data.shift(1))
        elif method == "simple":
            self.returns = price_data.pct_change()
        else:
            raise ValueError("Invalid method. Use 'log' or 'simple'.")

        self.returns = self.returns.dropna()
        return self.returns

    def compute_portfolio_returns(self):
        """Computes the aggregate weighted portfolio returns."""
        self.portfolio_returns = self.returns.dot(self.weights)
        return self.portfolio_returns

    def calculate_historical_var(self, series):
        """Calculates historical VaR using the empirical percentile distribution."""
        alpha = 1.0 - self.confidence_level
        return np.percentile(series, alpha * 100)

    def calculate_parametric_var(self):
        """Calculates Parametric (Variance-Covariance) VaR assuming normal distribution."""
        mu = np.mean(self.portfolio_returns)
        sigma = np.std(self.portfolio_returns, ddof=1)
        z_score = stats.norm.ppf(self.confidence_level)

        var_value = -(mu - z_score * sigma)
        return var_value

    def run_analysis(self):
        """Executes the pipeline and returns a summary dictionary of metrics."""
        self.download_data()
        self.calculate_returns(method="log")
        self.compute_portfolio_returns()

        hist_var_pct = self.calculate_historical_var(self.portfolio_returns)
        param_var_pct = self.calculate_parametric_var()

        results = {
            "name": self.name,
            "period": f"{self.start_date} to {self.end_date}",
            "hist_pct": hist_var_pct,
            "hist_dollar": hist_var_pct * self.initial_capital,
            "param_pct": param_var_pct,
            "param_dollar": param_var_pct * self.initial_capital,
        }
        return results


if __name__ == "__main__":
    # Define distinct portfolio instances fully with their own parameters
    portfolios = [
        PortfolioVaR(
            name="2008 Financial Crisis - Concentrated Financials",
            tickers=["SPY", "C", "BAC", "XLF", "GS"],
            weights=[0.2, 0.2, 0.2, 0.2, 0.2],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000,
            confidence_level=0.95,
        ),
        PortfolioVaR(
            name="2008 Financial Crisis - Multi-Asset Diverse",
            tickers=["SPY", "TLT", "GLD", "DBC"],
            weights=[0.4, 0.3, 0.2, 0.1],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000,
            confidence_level=0.95,
        ),
    ]

    # Iterate through the list and run analysis for each instance
    for port in portfolios:
        print("\n==================================================")
        print(f"Running Analysis for: {port.name}")
        print(f"Time Horizon: {port.start_date} to {port.end_date}")
        print("==================================================")

        metrics = port.run_analysis()

        print(f"Confidence Level: {port.confidence_level * 100}%")
        print(f"  - Historical VaR (%):        {metrics['hist_pct']:.4f}")
        print(f"  - Historical VaR ($):        ${abs(metrics['hist_dollar']):,.2f}")
        print(f"  - Parametric (Normal) VaR (%): {metrics['param_pct']:.4f}")
        print(f"  - Parametric (Normal) VaR ($): ${abs(metrics['param_dollar']):,.2f}")
