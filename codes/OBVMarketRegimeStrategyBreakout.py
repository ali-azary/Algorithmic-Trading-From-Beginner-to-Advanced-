import backtrader as bt

# Custom On-Balance Volume Indicator (remains the same)
class CustomOBV(bt.Indicator):
    lines = ('obv',)
    plotinfo = dict(subplot=True)
    def next(self):
        if len(self) == 1:
            if self.data.close[0] > self.data.close[-1]:
                self.lines.obv[0] = self.data.volume[0]
            elif self.data.close[0] < self.data.close[-1]:
                self.lines.obv[0] = -self.data.volume[0]
            else:
                self.lines.obv[0] = 0
        else:
            prev_obv = self.lines.obv[-1]
            if self.data.close[0] > self.data.close[-1]:
                self.lines.obv[0] = prev_obv + self.data.volume[0]
            elif self.data.close[0] < self.data.close[-1]:
                self.lines.obv[0] = prev_obv - self.data.volume[0]
            else:
                self.lines.obv[0] = prev_obv


class OBVMarketRegimeStrategyBreakout(bt.Strategy):
    params = (
        ('obv_ma_period', 7),
        ('rsi_period', 14),
        ('volume_ma_period', 7),
        ('adx_period', 7),
        ('adx_threshold', 20),
        ('breakout_lookback', 7), # Lookback for highest high/lowest low
        ('trail_percent', 0.03),
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low

        self.obv = CustomOBV(self.datas[0])
        self.obv_ma = bt.indicators.SMA(self.obv.lines.obv, period=self.p.obv_ma_period)
        self.obv_cross = bt.indicators.CrossOver(self.obv.lines.obv, self.obv_ma)
        
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ma_period)

        self.adx = bt.indicators.ADX(self.datas[0], period=self.p.adx_period)
        
        # Highest high and Lowest low for breakout
        self.highest_high = bt.indicators.Highest(self.datahigh, period=self.p.breakout_lookback)
        self.lowest_low = bt.indicators.Lowest(self.datalow, period=self.p.breakout_lookback)


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.sell(exectype=bt.Order.StopTrail, trailpercent=self.p.trail_percent)
            elif order.issell():
                self.buy(exectype=bt.Order.StopTrail, trailpercent=self.p.trail_percent)
        self.order = None

    def next(self):
        if self.order:
            return

        if len(self) < max(self.p.obv_ma_period, self.p.rsi_period, self.p.volume_ma_period, self.p.adx_period, self.p.breakout_lookback):
            return

        is_trending = self.adx.adx[0] > self.p.adx_threshold

        # Price Breakout Confirmation
        is_bullish_breakout = self.dataclose[0] > self.highest_high[-1] # Current close above previous N-period high
        is_bearish_breakout = self.dataclose[0] < self.lowest_low[-1]   # Current close below previous N-period low

        if not self.position:
            if (self.obv_cross[0] > 0.0 and
                self.rsi[0] < 70 and
                self.data.volume[0] > self.volume_ma[0] and
                is_trending and
                is_bullish_breakout): # Added price breakout
                self.order = self.buy()
            
            elif (self.obv_cross[0] < 0.0 and
                  self.rsi[0] > 30 and
                  self.data.volume[0] > self.volume_ma[0] and
                  is_trending and
                  is_bearish_breakout): # Added price breakout
                self.order = self.sell()