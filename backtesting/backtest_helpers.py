import yfinance as yf, pandas as pd, numpy as np
import get_data as gd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import mplcyberpunk
from pykalman import KalmanFilter


def calculate_sharpe(constant_pnl, risk_free_rate_annual=0.045, trading_days_per_year=252):
    """
    Calculates the real, annualized Sharpe ratio using daily portfolio values.

    Args:
        constant_pnl (list or np.array): List of daily portfolio values.
        risk_free_rate_annual (float): Annual risk-free rate (default 5% = 0.05).
        trading_days_per_year (int): Number of trading days per year (default 252).

    Returns:
        float: Annualized Sharpe ratio, risk-free adjusted.
    """
    if len(constant_pnl) < 2:
        return None  # Need at least 2 points to calculate returns

    pnl_array = np.array(constant_pnl)

    # If the first value is 0, skip it to avoid division by zero
    if pnl_array[0] == 0:
        pnl_array = pnl_array[1:]

    if len(pnl_array) < 2:
        return None  # still not enough points

    # Calculate daily returns
    daily_returns = np.diff(pnl_array) / pnl_array[:-1]

    # Remove any nan or inf values caused by bad data
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    daily_returns = daily_returns[~np.isinf(daily_returns)]

    # Mean and std dev of daily returns
    mean_return_daily = np.mean(daily_returns)
    std_return_daily = np.std(daily_returns, ddof=1)

    risk_free_daily = risk_free_rate_annual / trading_days_per_year

    # Sharpe ratio
    if std_return_daily == 0:
        return None

    sharpe_ratio = ((mean_return_daily - risk_free_daily) / std_return_daily) * np.sqrt(trading_days_per_year)

    return sharpe_ratio


def initial_kalman_hedge(log_t1, log_t2):
    """
    Initializes the Kalman Filter for the first time using the training data.
    Calculates the hedge ratio and intercept for the training data.

    Args:
        log_t1: Log prices of stock 1 
        log_t2: Log prices of stock 2
    Returns:
        last_state_mean (np.ndarray): Final state estimate after training, [hedge_ratio, intercept].
        last_state_cov (np.ndarray): Covariance matrix of the final state estimate.
    """
    X = np.vstack([log_t2, np.ones(len(log_t2))]).T
    y = log_t1

    # Initialize Kalman Filter for regression
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=X[:, np.newaxis, :],
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        observation_covariance=1.0,
        transition_covariance=0.01 * np.eye(2)
    )

    # Run filter on training data to get last state
    state_means, state_covs = kf.filter(y.values)
    last_state_mean = state_means[-1]
    last_state_cov = state_covs[-1]

    # Note - last_state_mean = [hedge_ratio, intercept]
    return last_state_mean, last_state_cov


def kalman_update(last_state_mean, last_state_cov, t1_log_price, t2_log_price):
    """
    Performs one-step Kalman update for hedge ratio and intercept.

    Args:
        last_state_mean (np.array): Previous [hedge_ratio, intercept]
        last_state_cov (np.array): Previous covariance matrix
        t1_log_price (float): Single log price of stock 1
        t2_log_price (float): Single log price of stock 2

    Returns:
        updated_state_mean (np.array): Updated [hedge_ratio, intercept]
        updated_state_cov (np.array): Updated covariance matrix

    """
    x = t2_log_price
    y = t1_log_price
    obs_mat = np.array([[x, 1.0]])

    # One-step Kalman update
    kf_obs_cov = 1.0
    kf_trans_cov = 0.01 * np.eye(2)

    # Kalman update step
    predicted_state_mean = last_state_mean
    predicted_state_cov = last_state_cov + kf_trans_cov

    S = obs_mat @ predicted_state_cov @ obs_mat.T + kf_obs_cov
    K = predicted_state_cov @ obs_mat.T @ np.linalg.inv(S)

    residual = y - obs_mat @ predicted_state_mean
    updated_state_mean = predicted_state_mean + K.flatten() * residual
    updated_state_cov = (np.eye(2) - K @ obs_mat) @ predicted_state_cov

    # Use updated slope and intercept
    hedge_ratio = updated_state_mean[0]
    intercept = updated_state_mean[1]

    # Store updated values
    last_state_mean = updated_state_mean
    last_state_cov = updated_state_cov

    # Note - last_state_mean = [hedge_ratio, intercept]
    return last_state_mean, last_state_cov, hedge_ratio, intercept


def plot_z_scores(z_score_list, entry_threshold_list, all_dates):
    """
    Plots Z-Scores over time with dynamic entry threshold lines.
    
    Args:
        z_score_list (list): List of Z-scores throughout the time period.  
        entry_threshold_list (list): List of dynamic entry thresholds throughout the time period.
        all_dates (list): Full list of dates corresponding to the Z-scores.
    """

    plt.style.use("cyberpunk")
    plt.figure(figsize=(14, 6))

    # Ensure all_dates is datetime for x-axis
    all_dates = pd.to_datetime(all_dates)

    # Plot Z-scores and thresholds using actual dates
    plt.plot(all_dates, z_score_list, label='Z-Scores', alpha=0.9, linewidth=2)
    plt.plot(all_dates, entry_threshold_list, color='red', linestyle='--', label='Dynamic Upper Threshold (+)')
    plt.plot(all_dates, [-t for t in entry_threshold_list], color='green', linestyle='--', label='Dynamic Lower Threshold (-)')

    # Add threshold crossing markers
    for i in range(1, len(z_score_list)):
        if z_score_list[i-1] < entry_threshold_list[i-1] and z_score_list[i] >= entry_threshold_list[i]:
            plt.scatter(all_dates[i], z_score_list[i] + 0.2, color='red', marker='v', s=100, label='Crossed Upper Threshold' if i == 1 else "")
        elif z_score_list[i-1] > -entry_threshold_list[i-1] and z_score_list[i] <= -entry_threshold_list[i]:
            plt.scatter(all_dates[i], z_score_list[i] - 0.2, color='green', marker='^', s=100, label='Crossed Lower Threshold' if i == 1 else "")

    # Decorations
    plt.title('Z-Scores Over Time with Dynamic Thresholds')
    plt.xlabel('Date')
    plt.ylabel('Z-Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add cyberpunk glow
    mplcyberpunk.add_glow_effects()
    
    plt.show()


def plot_stocks(df, t1, t2):
    """
    Plots two stock prices over time.

    Args:
        df (pandas.DataFrame): DataFrame containing stock prices (from yfinance)
        t1 (str): Ticker 1 (e.g., 'NVDA')
        t2 (str): Ticker 2 (e.g., 'AMD')
    Returns:
        None (displays a plot)
    """

    plt.style.use("cyberpunk")
    plt.figure(figsize=(14, 7))
    
    # Plot both stocks with _nolegend_ label to avoid legend spam
    line1, = plt.plot(df.index, df[t1], label='_nolegend_', linewidth=2.5, alpha=0.9)
    line2, = plt.plot(df.index, df[t2], label='_nolegend_', linewidth=2.5, alpha=0.9)

    mplcyberpunk.make_lines_glow()
    

    # Plot lines for legend only
    plt.plot([], [], color=line1.get_color(), label=t1, linewidth=2.5)
    plt.plot([], [], color=line2.get_color(), label=t2, linewidth=2.5)

    plt.title(f'{t1} vs {t2} Stock Prices', fontsize=18, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_pnl(constant_pnl, dates, capital, all_dates):
    """
    Plots the PnL of the strategy and compares it to the S&P 500 index, comparing the equity curves

    Args:
        constant_pnl (list): List of daily portfolio values.
        dates (list): List of dates corresponding to the PnL.
        capital (int): Initial capital for the strategy.
        all_dates (list): List of all dates for the S&P 500 data.
    Returns:
        None (displays a plot)
    """
    # Make sure dates are datetime
    all_dates = pd.to_datetime(all_dates)

    # Convert equity to percent returns
    equity_pct = (np.array(constant_pnl) / capital) * 100

    # Get S&P 500 data
    spx500_data = gd.get_price(["^GSPC"], dates)
    first_price = spx500_data.iloc[1, 0]

    shares = capital // first_price

    profits = []
    for price in spx500_data.iloc[:, 0]:
        profits.append(shares * (price - first_price))

    # Convert S&P500 profits to percent returns too
    profits_pct = (np.array(profits) / capital) * 100

    # Plotting
    plt.style.use("cyberpunk")
    plt.figure(figsize=(12, 6))

    # Plot strategy percent return
    plt.plot(all_dates, equity_pct, linewidth=2, label="Strategy Return (%)")

    # Plot SPX500 percent return
    plt.plot(spx500_data.index, profits_pct, linewidth=2, label="S&P 500 Return (%)")

    plt.title("Pairs Trading Strategy Return vs S&P 500 Return", fontsize=18)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Return", fontsize=14)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    
    mplcyberpunk.add_glow_effects()
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.3)

    plt.tight_layout()
    plt.show()

def plot_hedge_ratio(ratios_list, all_dates):
    """
    Plots the evolution of the hedge ratio over time.

    Args:
        ratios_list (list): List of hedge ratios over time.
        all_dates (list): List of dates corresponding to the hedge ratios.
    Returns:
        None (displays a plot)
    """
    
    all_dates = pd.to_datetime(all_dates)

    plt.style.use('cyberpunk')
    plt.figure(figsize=(12, 6))

    plt.plot(all_dates, ratios_list, linewidth=2, label="Hedge Ratio Over Time")

    plt.title("Dynamic Hedge Ratio Evolution", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Hedge Ratio", fontsize=14)

    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    
    mplcyberpunk.make_lines_glow()

    plt.tight_layout()
    plt.show()


def plot_trade_pnl(trades):
    """
    Plots a histogram of trade PnLs.

    Args:
        trades (list): List of trade PnLs (profits/losses).
    Returns:
        None (displays a plot)
    """
    trades_array = np.array(trades)

    plt.style.use('cyberpunk')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Simple histogram first
    counts, bins, bars = ax.hist(trades_array, bins=30, edgecolor='white', align='mid')

    # Color the bars
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for bar, center in zip(bars, bin_centers):
        if center >= 0:
            bar.set_facecolor('#00FF00')  # neon green
        else:
            bar.set_facecolor('#FF007F')  # neon magenta/red

    # Add cyberpunk glow after coloring
    mplcyberpunk.add_bar_gradient(bars=bars)

    # legend
    legend_elements = [
        Patch(facecolor='#00FF00', label='Winning Trades'),
        Patch(facecolor='#FF007F', label='Losing Trades')
    ]
    ax.legend(handles=legend_elements)

    # Decorations
    ax.set_title('Histogram of Trade PnLs', fontsize=16)
    ax.set_xlabel('PnL per Trade ($)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, alpha=0.3, zorder=1)
    plt.tight_layout()
    plt.show()
