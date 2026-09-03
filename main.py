# FILE: ./main.py
from stress_engine.portfolio import PortfolioVaR


def main() -> None:
    """Main entrypoint for historical VaR analysis and stress testing pipelines."""
    portfolios = [
        PortfolioVaR(
            name="2008 Financial Crisis - Concentrated Financials",
            tickers=["C", "BAC", "XLF", "GS"],
            weights=[0.25, 0.25, 0.25, 0.25],  # 4 assets = 4 weights (sum = 1.0)
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000.0,
            confidence_level=0.95,
        ),
        PortfolioVaR(
            name="2008 Financial Crisis - Multi-Asset Diverse",
            tickers=["SPY", "TLT", "GLD", "DBC"],
            weights=[0.4, 0.3, 0.2, 0.1],
            start_date="2007-01-01",
            end_date="2009-12-31",
            initial_capital=1_000_000.0,
            confidence_level=0.95,
        ),
    ]

    for port in portfolios:
        print("\n" + "=" * 50)
        print(f"Running Analysis for: {port.name}")
        print(f"Time Horizon: {port.start_date} to {port.end_date}")
        print("=" * 50)

        metrics = port.run_analysis()

        print(f"Confidence Level: {port.confidence_level * 100:.1f}%")
        print(f"  - Historical VaR (%):          {metrics['hist_pct']:.4f}")
        print(
            f"  - Historical VaR ($):          ${abs(float(metrics['hist_dollar'])):,.2f}"
        )
        print(f"  - Parametric (Normal) VaR (%): {metrics['param_pct']:.4f}")
        print(
            f"  - Parametric (Normal) VaR ($): ${abs(float(metrics['param_dollar'])):,.2f}"
        )


if __name__ == "__main__":
    main()
