import numpy as np
import get_data as gd
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mplcyberpunk


def plot_ols_returns(df, tickers):
    """
    Plots OLS regression line between returns of two stocks where slope is hedge ratio.
    Args:
        df (pandas.DataFrame): DataFrame containing stock prices (from yfinance)
        tickers (list): List of two stock tickers (e.g., ['GOOG', 'NVDA'])
    Returns:
        None  
    """
    plt.style.use("cyberpunk")

    # Calculate returns
    returns = df[[tickers[0], tickers[1]]].pct_change().dropna()
    x = returns[tickers[1]]
    y = returns[tickers[0]]

    # Fit OLS regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = model.params[0] + model.params[1] * x_vals

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, label="Returns Data", color='cyan')
    ax.plot(x_vals, y_vals, color='white', linewidth=2, label='OLS Regression Line')

    # Add correlation annotation
    corr = x.corr(y)
    ax.text(x.max(), y.min(), f"Correlation: {corr:.2f}", ha='right', fontsize=12, color='white')

    ax.set_xlabel(f'Returns of {tickers[1]}')
    ax.set_ylabel(f'Returns of {tickers[0]}')
    ax.set_title(f'OLS Regression of Returns: {tickers[0]} ~ {tickers[1]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage
# tickers = ['MSFT', 'NVDA']
# dates = ['2020-01-01', '2022-12-31']
# data = gd.get_price(tickers, dates, interval="1d")
# plot_ols_returns(data, tickers)



def kpss_test(df):
    """
    Assumes df has two columns representing the already-transformed price series (e.g., logs).
    Uses OLS calculated constant hedge ratio for spread, although will use dynamic ratio updated
    by Kalman Filters in backtesting
    Args:
        df (pandas.DataFrame): DataFrame containing stock prices (from yfinance)
        tickers (list): List of two stock tickers (e.g., ['GOOG', 'NVDA'])
    Returns:
        None
    Outputs whether the KPSS test indicates cointegration or not
    """

    df = df.dropna()

    # Calculate hedge ratio (via linear regression)
    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
    hedge_ratio = model.params.iloc[1]

    # Compute spread
    spread = df.iloc[:, 0] - hedge_ratio * df.iloc[:, 1]

    # Run KPSS test
    kpss_statistic, p_value, _, _ = sm.tsa.stattools.kpss(spread, regression='ct')
    
    # Interpret result
    if p_value >= 0.05:
        print ("KPSS Test - ✅ Good for Cointegration")
    else:
        print ("KPSS Test - ❌ Bad for Cointegration (Non-stationary spread)")


#Example usage
# tickers = ['AAPL', 'GOOG']
# dates = ['2020-01-01', '2022-12-31']
# data = gd.get_price(tickers, dates, interval="1d")
# data = np.log(data)
# kpss(data)


def engle_granger_test(df):
    """
    Assumes df has two columns representing the already-transformed price series (e.g., logs).
    Uses OLS calculated constant hedge ratio for spread. Runs ADF test on the spread.
    Args:
        df (pandas.DataFrame): DataFrame containing stock prices (from yfinance)
    Returns:
        None
    Outputs whether the ADF test indicates cointegration or not
    """
    df = df.dropna()

    # Calculate hedge ratio (via linear regression)
    model = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
    hedge_ratio = model.params.iloc[1]

    # Compute spread
    spread = df.iloc[:, 0] - hedge_ratio * df.iloc[:, 1]

    # Run ADF test
    adf_statistic, p_value, _, _, _, _ = adfuller(spread)
    
    # Interpret result
    if p_value <= 0.05:
        print ("Engle-Granger Test - ✅ Good for Cointegration")
    else:
        print ("Engle-Granger Test - ❌ Bad for Cointegration")


# Example usage
# tickers = ['COST', 'NVDA']
# dates = ['2020-01-01', '2022-12-31']
# data = gd.get_price(tickers, dates, interval="1d")
# data = np.log(data)
# engle_granger_test(data)
