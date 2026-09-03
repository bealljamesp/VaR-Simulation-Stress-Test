"""Portfolio Value-at-Risk (VaR) calculation and risk analysis engine."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import yfinance as yf
from scipy import stats

from stress_engine.backtest import BacktestResult, run_full_var_backtest


class PortfolioVaR:
    """Multi-asset portfolio VaR calculator supporting Historical and Parametric methods."""

    def __init__(
        self,
        name: str,
        tickers: list[str],
        weights: list[float] | npt.NDArray[np.float64],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000.0,
        confidence_level: float = 0.95,
    ) -> None:
        self.name = name
        self.tickers = tickers
        self.weights: npt.NDArray[np.float64] = np.ascontiguousarray(
            weights, dtype=np.float64
        )

        if len(self.tickers) != self.weights.size:
            raise ValueError(
                f"Dimension mismatch: {len(self.tickers)} tickers provided, "
                f"but received {self.weights.size} weights."
            )

        if not np.isclose(np.sum(self.weights), 1.0, atol=1e-4):
            raise ValueError(
                f"Portfolio weights must sum to 1.0. Current sum for '{self.name}': "
                f"{np.sum(self.weights):.4f}"
            )

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.confidence_level = confidence_level

        self.market_data: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.portfolio_returns: npt.NDArray[np.float64] | None = None

    def download_data(self) -> pd.DataFrame:
        """Downloads historical market data for the target tickers."""
        self.market_data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False,
        )
        return self.market_data

    def calculate_returns(self, method: str = "log") -> pd.DataFrame:
        """Calculates asset returns using vectorized pandas transformations."""
        if self.market_data is None or self.market_data.empty:
            raise ValueError("Market data is empty. Call download_data() first.")

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

        # Enforce column order matches tickers list
        price_data = price_data.loc[:, self.tickers]

        if method == "log":
            self.returns = np.log(price_data / price_data.shift(1))
        elif method == "simple":
            self.returns = price_data.pct_change()
        else:
            raise ValueError("Invalid return method. Choose 'log' or 'simple'.")

        self.returns = self.returns.dropna()
        return self.returns

    def compute_portfolio_returns(self) -> npt.NDArray[np.float64]:
        """Computes portfolio returns via zero-copy NumPy matrix dot product."""
        if self.returns is None or self.returns.empty:
            raise ValueError("Returns not computed. Call calculate_returns() first.")

        # Zero-copy conversion guaranteed via .to_numpy()
        returns_mat: npt.NDArray[np.float64] = self.returns.to_numpy(
            dtype=np.float64, copy=False
        )
        self.portfolio_returns = np.ascontiguousarray(
            returns_mat @ self.weights, dtype=np.float64
        )
        return self.portfolio_returns

    def calculate_historical_var(self) -> float:
        """Calculates Historical VaR expressed as a positive loss percentage."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")

        alpha = 1.0 - self.confidence_level
        # The percentile cutoff is typically negative; we return positive loss magnitude
        loss_cutoff = float(np.percentile(self.portfolio_returns, alpha * 100))
        return max(0.0, -loss_cutoff)

    def calculate_parametric_var(self) -> float:
        """Calculates Parametric (Variance-Covariance) VaR assuming normal distribution."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not computed.")

        mu = float(np.mean(self.portfolio_returns))
        sigma = float(np.std(self.portfolio_returns, ddof=1))
        z_score = float(stats.norm.ppf(self.confidence_level))

        # VaR = z * sigma - mu (positive loss percentage)
        var_pct = (z_score * sigma) - mu
        return max(0.0, var_pct)

    def run_backtest(self) -> BacktestResult:
        """Runs Kupiec and Christoffersen backtests against the modeled Historical VaR cutoff."""
        if self.portfolio_returns is None:
            self.run_analysis()

        assert self.portfolio_returns is not None
        hist_var_loss = self.calculate_historical_var()
        # Loss cutoff as a negative threshold to compare against daily returns
        cutoff_threshold = -hist_var_loss

        return run_full_var_backtest(
            returns=self.portfolio_returns,
            var_thresholds=cutoff_threshold,
            confidence_level=self.confidence_level,
        )

    def run_analysis(self) -> dict[str, float | str]:
        """Executes the pipeline and returns a summary dictionary of metrics."""
        self.download_data()
        self.calculate_returns(method="log")
        self.compute_portfolio_returns()

        hist_var = self.calculate_historical_var()
        param_var = self.calculate_parametric_var()

        return {
            "name": self.name,
            "period": f"{self.start_date} to {self.end_date}",
            "hist_pct": hist_var,
            "hist_dollar": hist_var * self.initial_capital,
            "param_pct": param_var,
            "param_dollar": param_var * self.initial_capital,
        }
