import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Custom analyzer for reliable performance metrics
class CustomPerformanceAnalyzer(bt.analyzers.Analyzer):
    params = (('riskfreerate', 0.01),)
    
    def __init__(self):
        self.portfolio_values = []
        self.trade_count = 0
        
    def next(self):
        self.portfolio_values.append(self.strategy.broker.getvalue())
        
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
        
    def get_analysis(self):
        if len(self.portfolio_values) < 2:
            return {'sharperatio': 0, 'total_return': 0, 'max_drawdown': 0}
            
        # Calculate metrics
        returns = []
        max_value = self.portfolio_values[0]
        max_drawdown = 0
        
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] / self.portfolio_values[i-1]) - 1
            returns.append(daily_return)
            
            # Track drawdown
            if self.portfolio_values[i] > max_value:
                max_value = self.portfolio_values[i]
            else:
                drawdown = (max_value - self.portfolio_values[i]) / max_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        if not returns:
            return {'sharperatio': 0, 'total_return': 0, 'max_drawdown': 0}
            
        returns = np.array(returns)
        
        # Calculate performance metrics
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        avg_daily_return = np.mean(returns)
        std_daily_return = np.std(returns)
        
        # Annualize
        annual_return = avg_daily_return * 252
        annual_volatility = std_daily_return * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.params.riskfreerate) / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'sharperatio': sharpe_ratio,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'trade_count': self.trade_count,
            'avg_daily_return': avg_daily_return,
            'daily_volatility': std_daily_return
        }

# Simple trend following strategy for testing
class TrendFollowingStrategy(bt.Strategy):
    params = (
        ('period', 50),
        ('stake', 1000),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.period
        )
        self.crossover = bt.indicators.CrossOver(self.data.close, self.sma)
        
    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = self.params.stake // self.data.close[0]
                self.buy(size=size)
        else:
            if self.crossover < 0:
                self.sell(size=self.position.size)

# Multi-asset diversified strategy
class DiversifiedTrendStrategy(bt.Strategy):
    params = (
        ('period', 50),
        ('stake_per_asset', 2500),
    )
    
    def __init__(self):
        self.smas = {}
        self.crossovers = {}
        
        for i, data in enumerate(self.datas):
            self.smas[data] = bt.indicators.SimpleMovingAverage(
                data.close, period=self.params.period
            )
            self.crossovers[data] = bt.indicators.CrossOver(
                data.close, self.smas[data]
            )
        
    def next(self):
        for data in self.datas:
            pos = self.getposition(data)
            
            if not pos:
                if self.crossovers[data] > 0:
                    size = self.params.stake_per_asset // data.close[0]
                    self.buy(data=data, size=size)
            else:
                if self.crossovers[data] < 0:
                    self.sell(data=data, size=pos.size)

def download_data(tickers, start_date, end_date):
    """Download data for multiple tickers"""
    data_feeds = {}
    raw_data = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False).droplevel(1, 1)
            if not df.empty:
                df.index.name = 'datetime'
                data_feeds[ticker] = bt.feeds.PandasData(dataname=df)
                raw_data[ticker] = df
                print(f"Downloaded data for {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return data_feeds, raw_data

def apply_volatility_shock(df, multiplier=1.5, shock_std=0.2):
    """Apply volatility shock to price data"""
    df_shocked = df.copy()
    
    # Calculate returns
    returns = df['Close'].pct_change().dropna()
    
    # Create volatility multiplier (random around the base multiplier)
    np.random.seed(42)  # For reproducibility
    vol_multipliers = np.random.normal(loc=multiplier, scale=shock_std, size=len(returns))
    vol_multipliers = np.clip(vol_multipliers, 0.5, 3.0)  # Reasonable bounds
    
    # Apply shock to returns
    shocked_returns = returns * vol_multipliers
    
    # Rebuild price series
    shocked_prices = [df['Close'].iloc[0]]
    for ret in shocked_returns:
        shocked_prices.append(shocked_prices[-1] * (1 + ret))
    
    # Update all OHLC data proportionally
    price_ratio = np.array(shocked_prices[1:]) / df['Close'].iloc[1:].values
    
    df_shocked.loc[df_shocked.index[1:], 'Open'] *= price_ratio
    df_shocked.loc[df_shocked.index[1:], 'High'] *= price_ratio
    df_shocked.loc[df_shocked.index[1:], 'Low'] *= price_ratio
    df_shocked.loc[df_shocked.index[1:], 'Close'] = shocked_prices[1:]
    df_shocked.loc[df_shocked.index[1:], 'Adj Close'] = shocked_prices[1:]
    
    return df_shocked

def apply_correlation_shock(data_dict, target_correlation=0.9):
    """Make all assets highly correlated (correlation breakdown scenario)"""
    shocked_data = {}
    
    # Use SPY as the base asset
    base_data = data_dict['SPY'].copy()
    base_returns = base_data['Close'].pct_change().dropna()
    
    for ticker, df in data_dict.items():
        df_shocked = df.copy()
        
        if ticker == 'SPY':
            shocked_data[ticker] = df_shocked
            continue
            
        # Get original returns
        original_returns = df['Close'].pct_change().dropna()
        
        # Create new returns that are correlated with SPY
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(base_returns))
        
        # Weighted combination: target_correlation * SPY_returns + (1-target_correlation) * noise
        correlated_returns = target_correlation * base_returns + (1 - target_correlation) * noise
        
        # Rebuild price series
        shocked_prices = [df['Close'].iloc[0]]
        for ret in correlated_returns:
            shocked_prices.append(shocked_prices[-1] * (1 + ret))
        
        # Update OHLC data
        price_ratio = np.array(shocked_prices[1:]) / df['Close'].iloc[1:].values
        
        df_shocked.loc[df_shocked.index[1:], 'Open'] *= price_ratio
        df_shocked.loc[df_shocked.index[1:], 'High'] *= price_ratio
        df_shocked.loc[df_shocked.index[1:], 'Low'] *= price_ratio
        df_shocked.loc[df_shocked.index[1:], 'Close'] = shocked_prices[1:]
        df_shocked.loc[df_shocked.index[1:], 'Adj Close'] = shocked_prices[1:]
        
        shocked_data[ticker] = df_shocked
    
    return shocked_data

def create_flash_crash_scenario(df, crash_day_index=100, crash_magnitude=-0.3, recovery_days=5):
    """Create a flash crash scenario"""
    df_crashed = df.copy()
    
    if crash_day_index >= len(df):
        crash_day_index = len(df) // 2
    
    # Apply crash
    crash_multiplier = 1 + crash_magnitude
    df_crashed.iloc[crash_day_index:crash_day_index+1, df_crashed.columns.get_loc('Open')] *= crash_multiplier
    df_crashed.iloc[crash_day_index:crash_day_index+1, df_crashed.columns.get_loc('High')] *= crash_multiplier
    df_crashed.iloc[crash_day_index:crash_day_index+1, df_crashed.columns.get_loc('Low')] *= crash_multiplier
    df_crashed.iloc[crash_day_index:crash_day_index+1, df_crashed.columns.get_loc('Close')] *= crash_multiplier
    df_crashed.iloc[crash_day_index:crash_day_index+1, df_crashed.columns.get_loc('Adj Close')] *= crash_multiplier
    
    # Gradual recovery over next few days
    for i in range(1, recovery_days + 1):
        if crash_day_index + i < len(df_crashed):
            recovery_factor = 1 + (abs(crash_magnitude) * (1 - i/recovery_days) * 0.2)
            df_crashed.iloc[crash_day_index + i:crash_day_index + i + 1, df_crashed.columns.get_loc('Open')] *= recovery_factor
            df_crashed.iloc[crash_day_index + i:crash_day_index + i + 1, df_crashed.columns.get_loc('High')] *= recovery_factor
            df_crashed.iloc[crash_day_index + i:crash_day_index + i + 1, df_crashed.columns.get_loc('Low')] *= recovery_factor
            df_crashed.iloc[crash_day_index + i:crash_day_index + i + 1, df_crashed.columns.get_loc('Close')] *= recovery_factor
            df_crashed.iloc[crash_day_index + i:crash_day_index + i + 1, df_crashed.columns.get_loc('Adj Close')] *= recovery_factor
    
    return df_crashed

def run_backtest(data_feeds, strategy_class, test_name, high_costs=False, **strategy_params):
    """Run a backtest with given parameters"""
    cerebro = bt.Cerebro()
    
    # Add data feeds
    if isinstance(data_feeds, dict):
        for name, feed in data_feeds.items():
            cerebro.adddata(feed, name=name)
    else:
        cerebro.adddata(data_feeds)
    
    # Add strategy
    cerebro.addstrategy(strategy_class, **strategy_params)
    
    # Set broker parameters
    cerebro.broker.setcash(10000)
    
    if high_costs:
        # High cost environment
        cerebro.broker.setcommission(commission=0.01)  # 1% commission
        # Note: backtrader doesn't have built-in slippage, but you can implement it
    else:
        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Add analyzer
    cerebro.addanalyzer(CustomPerformanceAnalyzer, _name='performance')
    
    # Run backtest
    print(f"\nRunning {test_name}...")
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    # Extract results
    performance = results[0].analyzers.performance.get_analysis()
    
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {performance['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {performance['sharperatio']:.3f}")
    print(f"Max Drawdown: {performance['max_drawdown']*100:.2f}%")
    print(f"Trades: {performance['trade_count']}")
    
    return {
        'test_name': test_name,
        'final_value': final_value,
        'total_return': performance['total_return'] * 100,
        'sharpe_ratio': performance['sharperatio'],
        'max_drawdown': performance['max_drawdown'] * 100,
        'annual_return': performance['annual_return'] * 100,
        'annual_volatility': performance['annual_volatility'] * 100,
        'trade_count': performance['trade_count']
    }

def run_parameter_sensitivity_test(data_feed, strategy_class, base_params, test_param, test_values):
    """Test sensitivity to parameter changes"""
    results = []
    
    for value in test_values:
        params = base_params.copy()
        params[test_param] = value
        
        test_name = f"{test_param}={value}"
        result = run_backtest(data_feed, strategy_class, test_name, **params)
        results.append(result)
    
    return results

def plot_stress_test_results(results):
    """Plot stress test comparison"""
    test_names = [r['test_name'] for r in results]
    returns = [r['total_return'] for r in results]
    sharpe_ratios = [r['sharpe_ratio'] for r in results]
    max_drawdowns = [r['max_drawdown'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Returns
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax1.bar(range(len(test_names)), returns, color=colors)
    ax1.set_title('Total Returns (%)')
    ax1.set_ylabel('Return (%)')
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Sharpe Ratios
    colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
    ax2.bar(range(len(test_names)), sharpe_ratios, color=colors)
    ax2.set_title('Sharpe Ratios')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Max Drawdowns
    ax3.bar(range(len(test_names)), max_drawdowns, color='orange')
    ax3.set_title('Maximum Drawdowns (%)')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_xticks(range(len(test_names)))
    ax3.set_xticklabels(test_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main stress testing workflow"""
    print("=" * 70)
    print("STRESS TESTING AND SCENARIO ANALYSIS DEMO")
    print("=" * 70)
    
    # Download data
    tickers = ['SPY', 'GLD', 'TLT', 'VTI']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    print("Downloading market data...")
    data_feeds, raw_data = download_data(tickers, start_date, end_date)
    
    if len(data_feeds) < 2:
        print("Not enough data. Exiting.")
        return
    
    # Results storage
    all_results = []
    
    # 1. BASELINE BACKTEST
    print("\n" + "="*50)
    print("1. BASELINE PERFORMANCE")
    print("="*50)
    
    # Single asset baseline
    baseline_single = run_backtest(
        data_feeds['SPY'], TrendFollowingStrategy, 
        "Baseline (SPY Only)", period=50, stake=1000
    )
    all_results.append(baseline_single)
    
    # Multi-asset baseline
    baseline_multi = run_backtest(
        data_feeds, DiversifiedTrendStrategy,
        "Baseline (Multi-Asset)", period=50, stake_per_asset=2500
    )
    all_results.append(baseline_multi)
    
    # 2. VOLATILITY SHOCK TEST
    print("\n" + "="*50)
    print("2. VOLATILITY SHOCK TESTS")
    print("="*50)
    
    # Create volatility shocked data
    shocked_spy = apply_volatility_shock(raw_data['SPY'], multiplier=2.0)
    shocked_data_feed = bt.feeds.PandasData(dataname=shocked_spy)
    
    vol_shock_result = run_backtest(
        shocked_data_feed, TrendFollowingStrategy,
        "Volatility Shock (2x)", period=50, stake=1000
    )
    all_results.append(vol_shock_result)
    
    # 3. CORRELATION BREAKDOWN TEST
    print("\n" + "="*50)
    print("3. CORRELATION BREAKDOWN TEST")
    print("="*50)
    
    # Create correlation shocked data
    corr_shocked_data = apply_correlation_shock(raw_data, target_correlation=0.95)
    corr_shocked_feeds = {}
    for ticker, df in corr_shocked_data.items():
        corr_shocked_feeds[ticker] = bt.feeds.PandasData(dataname=df)
    
    corr_shock_result = run_backtest(
        corr_shocked_feeds, DiversifiedTrendStrategy,
        "Correlation Shock (0.95)", period=50, stake_per_asset=2500
    )
    all_results.append(corr_shock_result)
    
    # 4. HIGH COST ENVIRONMENT TEST
    print("\n" + "="*50)
    print("4. HIGH COST ENVIRONMENT TEST")
    print("="*50)
    
    high_cost_result = run_backtest(
        data_feeds['SPY'], TrendFollowingStrategy,
        "High Costs (1% commission)", high_costs=True, period=50, stake=1000
    )
    all_results.append(high_cost_result)
    
    # 5. FLASH CRASH SCENARIO
    print("\n" + "="*50)
    print("5. FLASH CRASH SCENARIO")
    print("="*50)
    
    # Create flash crash scenario
    crash_spy = create_flash_crash_scenario(
        raw_data['SPY'], crash_day_index=200, 
        crash_magnitude=-0.35, recovery_days=10
    )
    crash_data_feed = bt.feeds.PandasData(dataname=crash_spy)
    
    crash_result = run_backtest(
        crash_data_feed, TrendFollowingStrategy,
        "Flash Crash (-35%)", period=50, stake=1000
    )
    all_results.append(crash_result)
    
    # 6. PARAMETER SENSITIVITY TEST
    print("\n" + "="*50)
    print("6. PARAMETER SENSITIVITY TEST")
    print("="*50)
    
    # Test different SMA periods
    period_test_results = run_parameter_sensitivity_test(
        data_feeds['SPY'], TrendFollowingStrategy,
        {'stake': 1000}, 'period', [20, 35, 50, 75, 100]
    )
    all_results.extend(period_test_results)
    
    # 7. HISTORICAL CRISIS SCENARIOS
    print("\n" + "="*50)
    print("7. HISTORICAL CRISIS SCENARIOS")
    print("="*50)
    
    # COVID Crash period
    try:
        covid_data = yf.download('SPY', start='2020-02-01', end='2020-05-01', auto_adjust=False).droplevel(1, 1)
        covid_data.index.name = 'datetime'
        covid_feed = bt.feeds.PandasData(dataname=covid_data)
        
        covid_result = run_backtest(
            covid_feed, TrendFollowingStrategy,
            "COVID Crisis (Feb-May 2020)", period=20, stake=1000  # Shorter period for crisis
        )
        all_results.append(covid_result)
    except:
        print("Could not download COVID crisis data")
    
    # 8. RESULTS ANALYSIS
    print("\n" + "="*70)
    print("STRESS TEST RESULTS SUMMARY")
    print("="*70)
    
    # Create summary table
    print(f"{'Test Name':<30} {'Return (%)':<12} {'Sharpe':<8} {'Max DD (%)':<12} {'Trades':<8}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['test_name']:<30} {result['total_return']:<12.2f} "
              f"{result['sharpe_ratio']:<8.3f} {result['max_drawdown']:<12.2f} "
              f"{result['trade_count']:<8}")
    
    # Plot results
    print("\nGenerating stress test comparison charts...")
    plot_stress_test_results(all_results)
    
    # 9. ROBUSTNESS ANALYSIS
    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS")
    print("="*70)
    
    baseline_return = baseline_single['total_return']
    baseline_sharpe = baseline_single['sharpe_ratio']
    baseline_dd = baseline_single['max_drawdown']
    
    print(f"Baseline Performance: {baseline_return:.2f}% return, {baseline_sharpe:.3f} Sharpe, {baseline_dd:.2f}% max DD")
    print()
    
    # Analyze stress test impacts
    stress_tests = [r for r in all_results if 'Shock' in r['test_name'] or 'Flash Crash' in r['test_name'] or 'High Costs' in r['test_name']]
    
    for test in stress_tests:
        return_impact = test['total_return'] - baseline_return
        sharpe_impact = test['sharpe_ratio'] - baseline_sharpe
        dd_impact = test['max_drawdown'] - baseline_dd
        
        print(f"{test['test_name']}:")
        print(f"  Return Impact: {return_impact:+.2f}% ({return_impact/baseline_return*100:+.1f}%)")
        print(f"  Sharpe Impact: {sharpe_impact:+.3f} ({sharpe_impact/abs(baseline_sharpe)*100 if baseline_sharpe != 0 else 0:+.1f}%)")
        print(f"  Max DD Impact: {dd_impact:+.2f}% ({dd_impact/baseline_dd*100:+.1f}%)")
        print()
    
    print("KEY INSIGHTS:")
    print("• Multi-asset diversification provides some protection against individual asset shocks")
    print("• Volatility shocks can significantly impact trend-following strategies")
    print("• High transaction costs erode performance substantially")
    print("• Parameter sensitivity reveals optimization robustness")
    print("• Flash crashes test stop-loss and risk management effectiveness")

if __name__ == "__main__":
    main()