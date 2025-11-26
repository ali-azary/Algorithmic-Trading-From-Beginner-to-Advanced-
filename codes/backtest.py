import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import dateutil.relativedelta as rd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import importlib

def load_strategy(class_name):
    module = importlib.import_module(class_name)
    strategy = getattr(module, class_name)
    return strategy

ticker = "BTC-USD"
start = "2018-01-01"
end = "2025-01-01"

start_dt = pd.to_datetime(start)
end_dt = pd.to_datetime(end)


strategy = load_strategy("QuantileChannelStrategy")

data = yf.download(ticker, start=start, end=end, progress=False)


data = data.droplevel(1, 1) if isinstance(data.columns, pd.MultiIndex) else data

feed = bt.feeds.PandasData(dataname=data)
cerebro = bt.Cerebro()
cerebro.addstrategy(strategy)
cerebro.adddata(feed)
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.001)
# cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='dd')
cerebro.addanalyzer(bt.analyzers.Returns,     _name='rets')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

start_val = cerebro.broker.getvalue()
results = cerebro.run()
strat = results[0]
final_val = cerebro.broker.getvalue()
ret = (final_val - start_val) / start_val * 100


print(f"Return: {ret:.2f}% | Final Value: {final_val:.2f}")



sh = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
dd = strat.analyzers.dd.get_analysis()
rets = strat.analyzers.rets.get_analysis()
tr = strat.analyzers.trades.get_analysis()

print(f"Sharpe: {sh}")
print(f"Max DD: {dd.get('max', {}).get('drawdown', 0):.2f}%  Len: {dd.get('max', {}).get('len', 0)}")
print(f"CAGR:   {rets.get('rnorm100', 0):.2f}%  Total: {rets.get('rtot', 0)*100:.2f}%")
print(f"Trades: {tr.get('total', {}).get('total', 0)}  Won: {tr.get('won', {}).get('total', 0)}  Lost: {tr.get('lost', {}).get('total', 0)}")    

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10
cerebro.plot(iplot=False)
