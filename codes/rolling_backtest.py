import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import dateutil.relativedelta as rd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib

%matplotlib inline

def load_strategy(class_name):
    module = importlib.import_module(class_name)
    strategy = getattr(module, class_name)
    return strategy



def report_stats(df):
    returns = pd.Series(df['return_pct'], dtype=float).dropna()

    # Equity curve from compounding window returns (start at 1.0)
    equity = (1.0 + returns / 100.0).cumprod()

    # Total period returns (from equity curve)
    total_period_returns = equity.pct_change().dropna()

    # Overall metrics for the whole backtest
    total_return_pct = (equity.iloc[-1] - 1.0) * 100.0 if not equity.empty else np.nan
    win_rate_pct = (returns > 0).mean() * 100.0 if len(returns) else np.nan

    # Max drawdown
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_drawdown_pct = drawdown.min() * 100.0 if not drawdown.empty else np.nan

    # Total-period Sharpe ratio
    total_period_sharpe = (total_period_returns.mean() / total_period_returns.std(ddof=0)
                           if total_period_returns.std(ddof=0) > 0 else np.nan)

    stats = {
        'Mean Return % (per window)': returns.mean(),
        'Median Return % (per window)': returns.median(),
        'Std Dev % (per window)': returns.std(ddof=0),
        'Min Return % (per window)': returns.min(),
        'Max Return % (per window)': returns.max(),
        'Sharpe Ratio (per window)': returns.mean() / returns.std(ddof=0) if returns.std(ddof=0) > 0 else np.nan,
        'Total Return % (whole backtest)': total_return_pct,
        'Max Drawdown % (whole backtest)': max_drawdown_pct,
        'Win Rate % (whole backtest)': win_rate_pct,
        'Total-Period Sharpe Ratio': total_period_sharpe,
        'Windows': int(len(returns))
    }

    print("\n=== ROLLING BACKTEST STATISTICS ===")
    for k, v in stats.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")
    return stats




def plot_period_returns(df, ticker, start, end, window_months):
    periods = list(range(len(df)))
    returns = df['return_pct']
    colors = ['green' if r >= 0 else 'red' for r in returns]
    title = f'{ticker} - Period Returns\n{start} to {end} | {window_months}-month windows'

    plt.figure(figsize=(10, 6))
    plt.bar(periods, returns, color=colors, alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Period')
    plt.ylabel('Return %')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(df, ticker, start, end, window_months):
    periods = list(range(len(df)))
    returns = df['return_pct']
    cumulative_returns = (1 + returns / 100).cumprod() * 100 - 100
    title = f'{ticker} - Cumulative Returns\n{start} to {end} | {window_months}-month windows'

    plt.figure(figsize=(10, 6))
    plt.plot(periods, cumulative_returns, marker='o', linewidth=2, markersize=6, color='blue')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Period')
    plt.ylabel('Cumulative Return %')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(df, rolling_sharpe_window, ticker, start, end, window_months):
    returns = df['return_pct']
    rolling_sharpe = returns.rolling(window=rolling_sharpe_window).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
    )
    valid_mask = ~rolling_sharpe.isna()
    valid_periods = [i for i, valid in enumerate(valid_mask) if valid]
    valid_sharpe = rolling_sharpe[valid_mask]
    title = f'{ticker} - Rolling Sharpe Ratio ({rolling_sharpe_window}-period)\n{start} to {end} | {window_months}-month windows'

    plt.figure(figsize=(10, 6))
    plt.plot(valid_periods, valid_sharpe, marker='o', linewidth=2, markersize=6, color='orange')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Period')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_return_distribution(df, ticker, start, end, window_months):
    returns = df['return_pct']
    bins = min(15, max(5, len(returns) // 2))
    mean_return = returns.mean()
    title = f'{ticker} - Return Distribution\n{start} to {end} | {window_months}-month windows'

    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Return %')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_returns_overlay(df, ticker, start, end, window_months, cumulative_on_secondary=True):
    """
    Overlay period returns (bars) and cumulative returns (line) in one figure.

    df must contain a 'return_pct' column where each row is the % return of a window.
    """
    periods = np.arange(len(df))
    period_returns = df['return_pct'].astype(float).values
    cumulative_returns_pct = (1 + period_returns / 100.0).cumprod() * 100.0 - 100.0  # in %

    title = f'{ticker} - Period & Cumulative Returns\n{start} to {end} | {window_months}-month windows'

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars: period returns on primary y-axis
    colors = ['green' if r >= 0 else 'red' for r in period_returns]
    bars = ax.bar(periods, period_returns, alpha=0.6, color=colors, label='Period Return %')
    ax.set_xlabel('Period')
    ax.set_ylabel('Period Return %')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Line: cumulative returns (secondary y-axis by default)
    if cumulative_on_secondary:
        ax2 = ax.twinx()
        line, = ax2.plot(periods, cumulative_returns_pct, marker='o', linewidth=2, label='Cumulative Return %')
        ax2.set_ylabel('Cumulative Return %')
        # Build a combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + [line], labels1 + labels2, loc='best')
    else:
        # Plot on same axis if you prefer one scale
        line, = ax.plot(periods, cumulative_returns_pct, marker='o', linewidth=2, label='Cumulative Return %')
        ax.legend(loc='best')

    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_returns_with_cumulative_side_by_side(df, ticker, start, end, window_months):
    """
    Plots period returns (bar chart) on the left subplot
    and cumulative returns (line chart) on the right subplot.
    """
    periods = np.arange(len(df))
    period_returns = df['return_pct'].astype(float).values
    cumulative_returns_pct = (1 + period_returns / 100.0).cumprod() * 100.0 - 100.0  # in %

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Left subplot: Period returns
    colors = ['green' if r >= 0 else 'red' for r in period_returns]
    ax1.bar(periods, period_returns, alpha=0.7, color=colors)
    ax1.set_title(f'{ticker} - Period Returns\n{start} to {end} | {window_months}-month windows',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Return %')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Right subplot: Cumulative returns
    ax2.plot(periods, cumulative_returns_pct, marker='o', linewidth=2, markersize=6, color='blue')
    ax2.set_title(f'{ticker} - Cumulative Returns\n{start} to {end} | {window_months}-month windows',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Cumulative Return %')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

def plot_returns_with_cumulative_overlay(df, ticker, start, end, window_months):
    """
    Plots period returns (bar chart) and cumulative returns (line chart) overlaid
    on a single plot with 3:4 aspect ratio and years as x-axis labels.
    """
    # Assume df has a date column or create one based on periods
    if 'date' not in df.columns:
        # Create dates assuming monthly windows starting from start date
        start_date = datetime.strptime(start, '%Y-%m-%d') if isinstance(start, str) else start
        dates = [start_date + pd.DateOffset(months=i*window_months) for i in range(len(df))]
        df = df.copy()
        df['date'] = dates
    
    period_returns = df['return_pct'].astype(float).values
    cumulative_returns_pct = (1 + period_returns / 100.0).cumprod() * 100.0 - 100.0
    
    # Create figure with 3:4 aspect ratio
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 9))  # 3:4 ratio (width:height)
    
    # Set up the primary y-axis for period returns (bars)
    colors = ['#10B981' if r >= 0 else '#EF4444' for r in period_returns]
    bars = ax1.bar(df['date'], period_returns, alpha=0.6, color=colors, 
                   label='Period Returns', width=pd.Timedelta(days=window_months*20))
    
    ax1.set_ylabel('Period Return %', fontsize=12, fontweight='bold', color='#2D3748')
    ax1.tick_params(axis='y', labelcolor='#2D3748')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Create secondary y-axis for cumulative returns (line)
    ax2 = ax1.twinx()
    line = ax2.plot(df['date'], cumulative_returns_pct, 
                    color='#3B82F6', linewidth=3, marker='o', 
                    markersize=6, markerfacecolor='#60A5FA', 
                    markeredgecolor='#1E40AF', markeredgewidth=1,
                    label='Cumulative Returns', alpha=0.9)
    
    ax2.set_ylabel('Cumulative Return %', fontsize=12, fontweight='bold', color='#1E40AF')
    ax2.tick_params(axis='y', labelcolor='#1E40AF')
    
    # Format x-axis to show years
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))  # Jan and July
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # Title and styling
    plt.title(f'{ticker} - Period & Cumulative Returns\n{start} to {end} | {window_months}-month windows',
              fontsize=14, fontweight='bold', pad=20, color='#1A202C')
    
    # Add legends
    bars_legend = ax1.legend(loc='upper left', framealpha=0.9, 
                            fancybox=True, shadow=True)
    line_legend = ax2.legend(loc='upper right', framealpha=0.9, 
                            fancybox=True, shadow=True)
    
    # Improve overall appearance
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#CBD5E0')
    ax1.spines['bottom'].set_color('#CBD5E0')
    ax2.spines['right'].set_color('#CBD5E0')
    
    # Set background color
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#FAFAFA')
    
    # Add subtle grid to cumulative line
    ax2.grid(True, alpha=0.1, linestyle='-', color='#3B82F6')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Add some padding around the data
    ax1.margins(x=0.02)
    
    # Optional: Add annotations for max/min values
    max_period_return = period_returns.max()
    min_period_return = period_returns.min()
    max_cumulative = cumulative_returns_pct.max()
    final_cumulative = cumulative_returns_pct[-1]
    
    # Add text box with key statistics
    stats_text = f'Max Period: {max_period_return:.1f}%\nMin Period: {min_period_return:.1f}%\nFinal Cumulative: {final_cumulative:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CBD5E0')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, fontfamily='monospace')
    
    plt.show()


# Alternative version with enhanced styling and annotations
def plot_returns_enhanced_overlay(df, ticker, start, end, window_months):
    """
    Enhanced version with more sophisticated styling and annotations.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # --- Font size definitions ---
    FONT_SIZES = {
        'title': 20,
        'subtitle': 20,
        'axis_label': 20,
        'axis_ticks': 20,
        'legend': 20,
        'annotation': 20,
        'stats_box': 15
    }

    # Prepare data
    if 'date' not in df.columns:
        start_date = datetime.strptime(start, '%Y-%m-%d') if isinstance(start, str) else start
        dates = [start_date + pd.DateOffset(months=i*window_months) for i in range(len(df))]
        df = df.copy()
        df['date'] = dates

    period_returns = df['return_pct'].astype(float).values
    cumulative_returns_pct = (1 + period_returns / 100.0).cumprod() * 100.0 - 100.0

    plt.style.use('default')
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10), layout="constrained")
  

    # Enhanced bar styling with gradient effect
    max_abs_return = max(abs(period_returns))
    colors = []
    for r in period_returns:
        intensity = min(abs(r) / max_abs_return, 1.0)
        if r >= 0:
            colors.append((0.06, 0.72, 0.51, 0.6 + 0.3 * intensity))
        else:
            colors.append((0.94, 0.27, 0.27, 0.6 + 0.3 * intensity))

    bars = ax1.bar(df['date'], period_returns, color=colors,
                   width=pd.Timedelta(days=window_months*15),
                   edgecolor='white', linewidth=0.5)

    # Outlier labels
    for i, (bar, value) in enumerate(zip(bars, period_returns)):
        height = bar.get_height()
        ax1.annotate(f'{value:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -15),
                     textcoords="offset points",
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=FONT_SIZES['annotation'], fontweight='bold',
                     color='#2D3748', alpha=0.8)

    # Cumulative returns line
    ax2 = ax1.twinx()

    ax2.plot(df['date'], cumulative_returns_pct,
             color='#3B82F6', linewidth=4, alpha=0.9,
             marker='o', markersize=5, markerfacecolor='#60A5FA',
             markeredgecolor='white', markeredgewidth=1.5)
    ax2.fill_between(df['date'], cumulative_returns_pct, alpha=0.1, color='#3B82F6')

    # Labels
    ax1.set_ylabel('Period Return %', fontsize=FONT_SIZES['axis_label'], fontweight='bold', color='#2D3748')
    ax2.set_ylabel('Cumulative Return %', fontsize=FONT_SIZES['axis_label'], fontweight='bold', color='#1E40AF')

    ax1.tick_params(axis='y', labelcolor='#2D3748', labelsize=FONT_SIZES['axis_ticks'])
    ax2.tick_params(axis='y', labelcolor='#1E40AF', labelsize=FONT_SIZES['axis_ticks'])
    ax1.tick_params(axis='x', labelsize=FONT_SIZES['axis_ticks'])

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))

    ax1.axhline(y=0, color='#4A5568', linestyle='-', alpha=0.5, linewidth=1.5)
    ax1.grid(True, alpha=0.3, linestyle='--', color='#A0AEC0')
    ax2.grid(True, alpha=0.15, linestyle='-', color='#3B82F6')

    # Titles
    fig.suptitle(
        f'{ticker} Investment Performance Dashboard\n'
        f'{start} to {end} • Rolling {window_months}-Month Windows',
        fontsize=FONT_SIZES['title'], fontweight='bold', color='#1A202C'
    )
    # Legends
    
    ax2.legend(['Cumulative Returns'], loc='upper right',
               framealpha=0.9, fancybox=True, shadow=True, fontsize=FONT_SIZES['legend'])

    # Stats box
    stats_text = (f'Key Metrics\n'
                  f'Best Period: +{period_returns.max():.1f}%\n'
                  f'Worst Period: {period_returns.min():.1f}%\n'
                  f'Final Return: {cumulative_returns_pct[-1]:.1f}%\n'
                  f'Volatility: {np.std(period_returns):.1f}%\n'
                  f'Win Rate: {(period_returns > 0).sum() / len(period_returns) * 100:.0f}%')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                 edgecolor='#CBD5E0', linewidth=1.5)
    ax1.text(0.02, 0.80, stats_text, transform=ax1.transAxes,
             fontsize=FONT_SIZES['stats_box'], verticalalignment='bottom',
             bbox=props, fontfamily='monospace')

    # Spine styling
    for spine in ax1.spines.values():
        spine.set_color('#CBD5E0')
        spine.set_linewidth(1)
    for spine in ax2.spines.values():
        spine.set_color('#CBD5E0')
        spine.set_linewidth(1)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#F7FAFC')

    # --- after fig, ax1 = plt.subplots(...) -------------------------------------
    fig.subplots_adjust(
        left=0.10,   # increase left margin
        right=0.85,  # increase right margin (room for the second y-axis & legend)
        top=0.90,    # increase top margin (room for the title)
        bottom=0.12  # increase bottom margin (room for x-axis labels)
    )    
    
    plt.show()

    
def run_rolling_backtest(
    ticker,
    start,
    end,
    window_months,
    strategy_params=None
):
    strategy_params = strategy_params or {}
    all_results = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    current_start = start_dt

    while True:
        current_end = current_start + rd.relativedelta(months=window_months)
        if current_end > end_dt:
            break

        print(f"\nROLLING BACKTEST: {current_start.date()} to {current_end.date()}")

        data = yf.download(ticker, start=current_start, end=current_end, progress=False)
        if data.empty or len(data) < 90:
            print("Not enough data.")
            current_start += rd.relativedelta(months=window_months)
            continue

        data = data.droplevel(1, 1) if isinstance(data.columns, pd.MultiIndex) else data

        feed = bt.feeds.PandasData(dataname=data)
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy, **strategy_params)
        cerebro.adddata(feed)
        cerebro.broker.setcash(100000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

        start_val = cerebro.broker.getvalue()
        cerebro.run()
        final_val = cerebro.broker.getvalue()
        ret = (final_val - start_val) / start_val * 100

        all_results.append({
            'start': current_start.date(),
            'end': current_end.date(),
            'return_pct': ret,
            'final_value': final_val,
        })

        print(f"Return: {ret:.2f}% | Final Value: {final_val:.2f}")
        current_start += rd.relativedelta(months=window_months)

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    
    strategy = load_strategy("KeltnerBreakoutStrategy")
    
    ticker = "DOGE-USD"
    start = "2018-01-01"
    end = "2025-01-01"
    window_months = 12

    df = run_rolling_backtest(ticker=ticker, start=start, end=end, window_months=window_months)

    print("\n=== ROLLING BACKTEST RESULTS ===")
    print(df)

    stats = report_stats(df)

    # plot_period_returns(df, ticker, start, end, window_months)
    # plot_cumulative_returns(df, ticker, start, end, window_months)
    # after df = run_rolling_backtest(...)
    # plot_returns_overlay(df, ticker, start, end, window_months, cumulative_on_secondary=True)
    plot_returns_enhanced_overlay(df, ticker, start, end, window_months)


    # plot_rolling_sharpe(df, 4, ticker, start, end, window_months)
    # plot_return_distribution(df, ticker, start, end, window_months)
