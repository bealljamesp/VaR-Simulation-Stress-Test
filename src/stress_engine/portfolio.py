from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


class PortfolioVaR:

    def __init__(
        self,
        name: str,
        tickers: list[str],
        weights: list[float],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        confidence_level: float = 0.95,
    ) -> None:
        self.name = name
        self.tickers = tickers
        self.weights = np.array(weights, dtype=np.float64)

        # Validate that weights sum to 1.0 upon instantiation
        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-4):
            raise ValueError(
                f"Portfolio weights must sum to 1.0. Current sum for '{self.name}':"
                f" {np.sum(self.weights)}"
            )

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.confidence_level = confidence_level

        self.market_data: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.portfolio_returns: np.ndarray | None = None

    def download_data(self) -> pd.DataFrame:
        """Downloads historical market data for the portfolio tickers."""
        self.market_data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )
        return self.market_data

    def calculate_returns(self, method: str = "log") -> pd.DataFrame:
        """Calculates historical asset returns using vectorized pandas methods."""
        if isinstance(self.market_data.columns, pd.MultiIndex):
            price_data = (
                self.market_data["Adj Close"]
                if "Adj Close" in self.market_data.columns.levels[0]
                else self.market_data["Close"]
            )
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

    def compute_portfolio_returns(self) -> np.ndarray:
        """Computes aggregate weighted portfolio returns via contiguous vector dot product."""
        self.portfolio_returns = self.returns.values @ self.weights
        return self.portfolio_returns

    def calculate_historical_var(self, series: np.ndarray) -> float:
        """Calculates historical VaR using the empirical percentile distribution."""
        alpha = 1.0 - self.confidence_level
        return float(np.percentile(series, alpha * 100))

    def calculate_parametric_var(self) -> float:
        """Calculates Parametric (Variance-Covariance) VaR assuming normal distribution."""
        mu = float(np.mean(self.portfolio_returns))
        sigma = float(np.std(self.portfolio_returns, ddof=1))
        z_score = float(stats.norm.ppf(self.confidence_level))

        var_value = -(mu - z_score * sigma)
        return var_value

    def run_analysis(self) -> dict[str, float | str]:
        """Executes the pipeline and returns a summary dictionary of metrics."""
        self.download_data()
        self.calculate_returns(method="log")
        self.compute_portfolio_returns()

        hist_var_pct = self.calculate_historical_var(self.portfolio_returns)
        param_var_pct = self.calculate_parametric_var()

        results: dict[str, float | str] = {
            "name": self.name,
            "period": f"{self.start_date} to {self.end_date}",
            "hist_pct": hist_var_pct,
            "hist_dollar": hist_var_pct * self.initial_capital,
            "param_pct": param_var_pct,
            "param_dollar": param_var_pct * self.initial_capital,
        }
        return results
