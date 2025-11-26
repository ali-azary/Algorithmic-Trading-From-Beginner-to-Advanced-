import backtrader as bt

class KeltnerChannelRSIBreakoutStrategy(bt.Strategy):
    params = (
        ('ema_period', 30),
        ('atr_period', 7),
        ('atr_multiplier', 1.),
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('trailing_stop_atr', 1.0),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        
        self.upper_band = self.ema + (self.atr * self.params.atr_multiplier)
        self.lower_band = self.ema - (self.atr * self.params.atr_multiplier)
        
        self.order = None
        self.trailing_stop_price = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Long: price above upper band + RSI confirmation
            if (self.data.close[0] > self.upper_band[0] and 
                self.rsi[0] > self.params.rsi_low):
                self.order = self.buy()
                
            # Short: price below lower band + RSI confirmation  
            elif (self.data.close[0] < self.lower_band[0] and 
                  self.rsi[0] < self.params.rsi_high):
                self.order = self.sell()
        else:
            # Update trailing stop
            if self.position.size > 0:  # Long position
                new_stop = self.data.close[0] - (self.atr[0] * self.params.trailing_stop_atr)
                if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                if self.data.close[0] <= self.trailing_stop_price:
                    self.order = self.close()
                    self.trailing_stop_price = None
                    
            elif self.position.size < 0:  # Short position
                new_stop = self.data.close[0] + (self.atr[0] * self.params.trailing_stop_atr)
                if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                if self.data.close[0] >= self.trailing_stop_price:
                    self.order = self.close()
                    self.trailing_stop_price = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None