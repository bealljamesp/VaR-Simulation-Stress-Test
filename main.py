"""Macroeconomic Crisis Stress Engine & VaR Backtesting Pipeline.

Orchestrates data ingestion, GARCH volatility filtering, historical crash
backtesting,
and stochastic Monte Carlo shock simulations.
"""

from stress_engine.backtest import run_var_backtest
from stress_engine.monte_carlo import run_monte_carlo_shocks
from stress_engine.pipeline import ingest_market_data, preprocess_returns
from stress_engine.volatility import fit_garch_model


def main():
    print("🚀 Initializing Macroeconomic Crisis Stress & VaR Backtesting Engine...")

    # 1. Ingest historical multi-asset market data (Equities, Rates, Credit Spreads)
    raw_data = ingest_market_data(source="data/historical_market_ledgers.parquet")
    returns = preprocess_returns(raw_data)
    print(
        f"Loaded and cleaned {returns.height} historical trading intervals"
        " successfully."
    )

    # 2. Fit GARCH(1,1) volatility models to capture time-varying variance clustering
    conditional_volatilities = fit_garch_model(returns)
    print("Fitted GARCH(1,1) conditional volatility models across portfolio assets.")

    # 3. Perform Historical Crash Backtesting & Evaluate VaR Breaches
    backtest_results = run_var_backtest(
        returns, conditional_volatilities, confidence_level=0.99
    )
    print(
        f"⚠️ Detected {backtest_results['total_breaches']} VaR exceptions during"
        " historical crash window."
    )

    # 4. Execute Stochastic Monte Carlo Shock Engine for Forward Stress Testing
    stress_scenarios = run_monte_carlo_shocks(n_simulations=10000, shock_multiplier=2.5)
    print(
        "Generated stochastic Monte Carlo shock scenarios modeling fat-tailed"
        " tail risk."
    )

    # 5. Output Summary Risk Report
    print("\n--- STRESS TESTING SUMMARY REPORT ---")
    print(backtest_results["summary_statistics"])


if __name__ == "__main__":
    main()
