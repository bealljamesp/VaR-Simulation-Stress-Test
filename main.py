# VaR Simulation and Stress Test

import yfinance as yf


def download_market_data(tickers, start_date, end_date):
    """
    Downloads historical market data for the specified tickers and date range.

    Parameters:
        tickers (list): List of ticker symbols to download.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing historical market data for the specified tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data


if __name__ == "__main__":
    # Example usage: Download historical data for S&P 500 and NASDAQ indices
    tickers = [
        "SPY",
        "C",
        "BAC",
        "XLF",
        "GS",
    ]  # S&P 500, Citigroup, Bank of America, Financial Sector ETF, Goldman Sachs
    start_date = "2007-01-01"
    end_date = "2009-12-31"

    market_data = download_market_data(tickers, start_date, end_date)
    print(market_data.head())
