"""Main entrypoint for historical VaR analysis and stress testing pipelines."""

from stress_engine.portfolio import PortfolioVaR


def main() -> None:
    portfolios = [
        PortfolioVaR(
            name="2008 Financial Crisis - Concentrated Financials",
            tickers=["C", "BAC", "XLF", "GS"],
            weights=[0.25, 0.25, 0.25, 0.25],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000.0,
            confidence_level=0.95,
        ),
        PortfolioVaR(
            name="2008 Financial Crisis - Multi-Asset Diverse",
            tickers=["SPY", "TLT", "GLD", "DBC"],
            weights=[0.40, 0.30, 0.20, 0.10],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000.0,
            confidence_level=0.95,
        ),
    ]

    for port in portfolios:
        print("\n" + "=" * 60)
        print(f"PORTFOLIO: {port.name}")
        print(f"HORIZON:   {port.start_date} to {port.end_date}")
        print("=" * 60)

        metrics = port.run_analysis()
        backtest = port.run_backtest()

        print(f"Confidence Level:           {port.confidence_level * 100:.1f}%")
        print(
            f"  Historical VaR (1-Day):   {metrics['hist_pct'] * 100:.2f}%  (${metrics['hist_dollar']:,.2f})"
        )
        print(
            f"  Parametric VaR (1-Day):   {metrics['param_pct'] * 100:.2f}%  (${metrics['param_dollar']:,.2f})"
        )
        print("-" * 60)
        print("REGULATORY BACKTESTING (Kupiec & Christoffersen):")
        print(f"  Observations:             {backtest.total_observations}")
        print(
            f"  Breaches:                 {backtest.total_exceptions} (Empirical: {backtest.empirical_rate * 100:.2f}%)"
        )
        print(
            f"  Kupiec POF LR:            {backtest.kupiec_stat:.3f} (p={backtest.kupiec_p_value:.4f}) -> {'REJECT H0' if backtest.kupiec_reject else 'ACCEPT H0'}"
        )
        print(
            f"  Christoffersen Indep LR:  {backtest.christoffersen_stat:.3f} (p={backtest.christoffersen_p_value:.4f}) -> {'REJECT H0' if backtest.christoffersen_reject else 'ACCEPT H0'}"
        )
        print(
            f"  Combined Conditional LR:  {backtest.combined_stat:.3f} (p={backtest.combined_p_value:.4f}) -> {'REJECT H0' if backtest.combined_reject else 'ACCEPT H0'}"
        )


if __name__ == "__main__":
    main()
