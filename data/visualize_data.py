import get_data as gd
import matplotlib.pyplot as plt
import mplcyberpunk
import pandas as pd
import numpy as np


def interpolate(ticker, dates, norm_type, degree=3, interval="1d"):
    """
    Interpolates stock prices using polynomial regression.
    Args:
        ticker (str): Stock ticker symbol (e.g., 'TSLA')
        dates (list): List of two dates [start_date, end_date] in 'YYYY-MM-DD' format
        norm_type (str): Normalization type ('normal', 'min_max', 'first_price', 'relative_change')
        degree (int): Degree of the polynomial for regression (default is 3)
        interval (str): Data interval, default is "1d" (1 day)
    Returns:
        None (displays a plot)
    """

    df = gd.get_price([ticker], dates, interval)
    df = gd.normalize_prices(df, norm_type)

    x_coords = np.arange(len(df))  # [0, 1, 2, ..., N-1]
    y_coords = df[ticker].values

    points = np.column_stack((x_coords, y_coords))
    np.set_printoptions(suppress=True)

    # Create the design matrix X (with x^0, x^1, ..., x^degree)
    X = np.ones((len(x_coords), degree + 1))  # Start with x^0 (constant term)
    for i in range(1, degree + 1):
        X[:, i] = x_coords**i  # Add x^i as each column

    # Convert Y to a column vector
    Y = y_coords.reshape(-1, 1)

    # Regression using pseudo-inverse
    beta = np.linalg.pinv(X) @ Y
    y_pred = X @ beta  # Predicted y-values from regression

    # Plot actual data points
    plt.style.use("cyberpunk")
    plt.figure(figsize=(12, 6))
    plt.scatter(df.index, y_coords, label="Actual", s=10)

    # Plot regression line
    plt.plot(df.index, y_pred.flatten(), color="orange", linewidth=2, label=f"Polynomial Regression (deg {degree})")

    plt.title(f"{ticker} Normalized Price + Polynomial Regression (deg {degree})")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Example usage
# interpolate("GOOG", ["2024-03-21", "2025-03-23"], "relative_change", degree=250)


def graph_stocks(tickers, dates, norm_type="relative_change", interval="1d"):
    """
    Plots normalized stock prices for a list of tickers over a specified date range.
    Args:
        tickers (list): List of stock ticker symbols (e.g., ['META', 'AMZN'])
        dates (list): List of two dates [start_date, end_date] in 'YYYY-MM-DD' format
        norm_type (str): Normalization type ('normal', 'min_max', 'first_price', 'relative_change')
        interval (str): Data interval, default is "1d" (1 day)
    Returns:
        None (displays a plot)
    """
    
    plt.style.use("cyberpunk")
    fig, ax = plt.subplots(figsize=(14, 8))

    xticks = None  # Shared x-axis ticks (time index)

    for ticker in tickers:
        price_data = gd.get_price([ticker], dates, interval)
        price_data = gd.normalize_prices(price_data, norm_type)

        if price_data.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        if xticks is None:
            xticks = price_data.index

        # Plot against index to skip time gaps
        ax.plot(range(len(price_data)), price_data[ticker], label=ticker, linewidth=2)

    # Set x-ticks to actual timestamps (but show only a few to prevent crowding)
    if xticks is not None:
        skip = max(1, len(xticks) // 10)  # Adjust label frequency
        ax.set_xticks(range(0, len(xticks), skip))
        ax.set_xticklabels([xticks[i].strftime('%Y-%m-%d\n%H:%M') for i in range(0, len(xticks), skip)], rotation=45)

    ax.set_title(f"Stock Prices from {dates[0]} to {dates[1]}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Normalized Price ({norm_type})")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
#graph_stocks(["COST", "AMZN", "AMD"], ["2022-05-10", "2025-05-14"], interval="1d")