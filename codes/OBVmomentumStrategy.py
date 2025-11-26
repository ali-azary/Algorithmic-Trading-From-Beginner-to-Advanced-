import backtrader as bt


# Custom On-Balance Volume Indicator
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


class OBVmomentumStrategy(bt.Strategy):
    params = (
        ('obv_ma_period', 30),
        ('trail_percent', 0.02),
        ('rsi_period', 14),
        ('volume_ma_period', 7),
    )
    
    def __init__(self):
        self.order = None
        
        # Original indicators
        self.obv = CustomOBV(self.datas[0])
        self.obv_ma = bt.indicators.SimpleMovingAverage(
            self.obv.lines.obv, period=self.params.obv_ma_period
        )
        self.obv_cross = bt.indicators.CrossOver(self.obv.lines.obv, self.obv_ma)
        
        # Two simple filters for better signal quality
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.params.volume_ma_period)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.sell(exectype=bt.Order.StopTrail, trailpercent=self.params.trail_percent)
            elif order.issell():
                self.buy(exectype=bt.Order.StopTrail, trailpercent=self.params.trail_percent)
        self.order = None
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            # Long signal: OBV crosses up + RSI not overbought + volume above average
            if (self.obv_cross[0] > 0.0 and 
                self.rsi[0] < 70 and 
                self.data.volume[0] > self.volume_ma[0]):
                self.order = self.buy()
                
            # Short signal: OBV crosses down + RSI not oversold + volume above average
            elif (self.obv_cross[0] < 0.0 and 
                  self.rsi[0] > 30 and 
                  self.data.volume[0] > self.volume_ma[0]):
                self.order = self.sell()