import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Custom Keltner Channel Indicator
class KeltnerChannel(bt.Indicator):
    """
    Keltner Channel indicator with EMA centerline and ATR-based bands
    """
    lines = ('mid', 'top', 'bot')
    params = (
        ('ema_period', 30),
        ('atr_period', 14), 
        ('atr_multiplier', 1.0),
    )

    def __init__(self):
        # EMA for centerline
        self.lines.mid = bt.indicators.EMA(self.data.close, period=self.params.ema_period)
        
        # ATR for band width
        atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # Upper and lower bands
        self.lines.top = self.lines.mid + (atr * self.params.atr_multiplier)
        self.lines.bot = self.lines.mid - (atr * self.params.atr_multiplier)

# Keltner Channel Breakout Strategy
class KeltnerBreakoutStrategy(bt.Strategy):
    params = (
        ('ema_period', 30),
        ('atr_period', 7),
        ('atr_multiplier', 1.0),
        ('printlog', True),
    )

    def log(self, txt, dt=None):
        """Logging utility"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def __init__(self):
        # Keep reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        
        # Initialize Keltner Channel
        self.keltner = KeltnerChannel(
            self.data,
            ema_period=self.params.ema_period,
            atr_period=self.params.atr_period,
            atr_multiplier=self.params.atr_multiplier
        )
        
        # Track order and buy price
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Add indicators to plot
        self.keltner.plotinfo.subplot = False
        self.keltner.plotlines.mid._plotskip = False
        self.keltner.plotlines.top._plotskip = False
        self.keltner.plotlines.bot._plotskip = False

    def notify_order(self, order):
        """Track order execution"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        """Track completed trades"""
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT: Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}')

    def next(self):
        """Main strategy logic"""
        # Skip if we don't have enough data
        if len(self.data) < max(self.params.ema_period, self.params.atr_period):
            return
            
        # Check if we have a pending order
        if self.order:
            return

        # Get previous day's values for signal generation
        if len(self.data) < 2:
            return
            
        prev_close = self.dataclose[-1]
        prev_upper = self.keltner.top[-1]
        prev_lower = self.keltner.bot[-1]
        current_ema = self.keltner.mid[0]
        current_close = self.dataclose[0]

        if not self.position:  # Not in market
            # Long entry: Previous close > Previous upper band
            if prev_close > prev_upper:
                self.log(f'BUY CREATE: Price {self.dataopen[0]:.2f} (Prev Close: {prev_close:.2f} > Upper: {prev_upper:.2f})')
                self.order = self.buy()
                
            # Short entry: Previous close < Previous lower band  
            elif prev_close < prev_lower:
                self.log(f'SELL CREATE: Price {self.dataopen[0]:.2f} (Prev Close: {prev_close:.2f} < Lower: {prev_lower:.2f})')
                self.order = self.sell()

        else:  # In market
            # Exit conditions based on current close vs EMA
            if self.position.size > 0:  # Long position
                if current_close < current_ema:
                    self.log(f'CLOSE LONG: Price {current_close:.2f} < EMA {current_ema:.2f}')
                    self.order = self.close()
                    
            elif self.position.size < 0:  # Short position
                if current_close > current_ema:
                    self.log(f'CLOSE SHORT: Price {current_close:.2f} > EMA {current_ema:.2f}')
                    self.order = self.close()

# Analyzer to track additional metrics
class TradeAnalyzer(bt.Analyzer):
    def create_analysis(self):
        self.rets = {}
        self.vals = []

    def notify_cashvalue(self, cash, value):
        self.vals.append(value)

    def get_analysis(self):
        return {'final_value': self.vals[-1] if self.vals else 0,
                'max_value': max(self.vals) if self.vals else 0}

