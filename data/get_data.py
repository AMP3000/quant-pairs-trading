import yfinance as yf
import pandas as pd
from pytickersymbols import PyTickerSymbols

def get_price(ticker_list, date_interval, interval="1d"):
    """
    Fetches stock prices for a list of tickers over a specified date range from yfinance.
    Args:
        ticker_list (list): List of stock ticker symbols (e.g., ['META', 'AMZN'])
        date_interval (list): List of two dates [start_date, end_date] in 'YYYY-MM-DD' format
        interval (str): Data interval, default is "1d" (1 day)
    Returns:
        DataFrame containing stock prices with tickers as columns and dates as index.
    """
    all_data = pd.DataFrame()
    for symbol in ticker_list:
        #Keeps only close price column, discards high, low, etc. 
        #adjusted for dividends, stock splits, etc. 
        data = yf.download(symbol, start=min(date_interval), end=max(date_interval), \
                           auto_adjust=True, interval=interval, progress=False)["Close"] 
        all_data[symbol] = data
    return all_data


#Example usage
#print(get_price(["AAPL", "TSLA", "NVDA"], ["2024-01-01", "2025-01-01"]))


def normalize_prices(df, norm_type):
    """
    Normalizes stock prices in a DataFrame based on the specified normalization type.
    Args:
        df (Pandas df): DataFrame containing stock prices (from yfinance)
        norm_type (str): Normalization type ('normal', 'min_max', 'first_price', 'relative_change')
    Returns:
        DataFrame with normalized stock prices.
    Raises:
        ValueError: If the DataFrame is empty.
    """

    if df.empty:
        raise ValueError("normalize_prices received an empty DataFrame")

    if norm_type == "normal":
        return df
    elif norm_type == "min_max":
        df = (df - df.min()) / (df.max() - df.min())
    elif norm_type == "first_price":
        df = df / df.iloc[0] 
    elif norm_type == "relative_change":
        df = (df - df.iloc[0]) / df.iloc[0] 
    else:
        print("Unrecognized normalization type.")
    return df

#Example usage
# print (normalize_prices(get_price(["GOOG", "MSFT", "AMD"], ("2024-01-01", "2025-01-01")), "min_max_normalized"))


def get_tickers(index, source="yahoo"):
    """
    Fetches stock tickers for a given index from either Yahoo or Google Finance.
    Args:
        index (str): The stock index to fetch tickers for (e.g., 'sp_500', 'nasdaq_100', 'dow_30').
        source (str): The source to fetch tickers from ('yahoo' or 'google').
    Returns:
        List of stock tickers for the specified index.
    Raises:
        ValueError: If the index or source is invalid.
    """
    # Initialize the stock data object
    stock_data = PyTickerSymbols()

    # Construct the getter method name dynamically
    if source == "yahoo":
        getter_name = f"get_{index}_nyc_yahoo_tickers"
    elif source == "google":
        getter_name = f"get_{index}_nyc_google_tickers"
    else:
        raise ValueError("Invalid source. Choose either 'yahoo' or 'google'.")

    # Check if the method exists in the PyTickerSymbols object
    if hasattr(stock_data, getter_name):
        # Dynamically fetch tickers using getattr()
        tickers = getattr(stock_data, getter_name)()
        return tickers
    else:
        raise ValueError(f"Invalid index or source. Could not find {getter_name} method.")


# Example: Getting S&P 500 tickers from Yahoo
#print (get_tickers("sp_500"))