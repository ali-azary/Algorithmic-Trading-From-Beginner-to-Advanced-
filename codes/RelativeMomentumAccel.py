import backtrader as bt

class RelativeMomentumAccel(bt.Strategy):
    """
    A strategy that enters on a statistical breakout of a custom oscillator
    measuring the acceleration of price away from its adaptive baseline trend.
    """
    params = (
        # Baseline Trend and Momentum
        ('kama_period', 30),
        ('fast_ema_period', 7),
        # Thrust Oscillator Breakout
        ('thrust_bb_period', 7),
        ('thrust_bb_devfactor', 1.),
        # Risk Management
        ('atr_period', 7),
        ('atr_stop_multiplier', 3.0),
    )

    def __init__(self):
        self.order = None

        # --- Indicators ---
        self.kama = bt.indicators.KAMA(self.data.close, period=self.p.kama_period)
        self.fast_ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.fast_ema_period)

        # --- Custom Thrust Oscillator ---
        # Note: A small number is added to self.kama to prevent division by zero in rare cases
        self.thrust_osc = (self.fast_ema - self.kama) / (self.kama + 1e-6)

        # --- Breakout Bands on the Oscillator ---
        self.thrust_bbands = bt.indicators.BollingerBands(
            self.thrust_osc,
            period=self.p.thrust_bb_period,
            devfactor=self.p.thrust_bb_devfactor
        )

        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.p.atr_period)

        # --- Trailing Stop State ---
        self.stop_price = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if self.position and self.stop_price is None:
                if order.isbuy():
                    self.highest_price_since_entry = self.data.high[0]
                    self.stop_price = self.highest_price_since_entry - (self.atr[0] * self.p.atr_stop_multiplier)
                elif order.issell():
                    self.lowest_price_since_entry = self.data.low[0]
                    self.stop_price = self.lowest_price_since_entry + (self.atr[0] * self.p.atr_stop_multiplier)
            elif not self.position:
                self.stop_price = None; self.highest_price_since_entry = None; self.lowest_price_since_entry = None
        self.order = None

    def next(self):
        if self.order: return

        if not self.position:
            # --- Entry Logic ---
            # Buy when the thrust oscillator breaks above its upper band
            if self.thrust_osc[0] > self.thrust_bbands.top[0]:
                self.order = self.buy()
            # Sell when the thrust oscillator breaks below its lower band
            elif self.thrust_osc[0] < self.thrust_bbands.bot[0]:
                self.order = self.sell()
                
        elif self.position:
            # --- Manual ATR Trailing Stop Logic ---
            if self.position.size > 0: # Long
                self.highest_price_since_entry = max(self.highest_price_since_entry, self.data.high[0])
                new_stop = self.highest_price_since_entry - (self.atr[0] * self.p.atr_stop_multiplier)
                self.stop_price = max(self.stop_price, new_stop)
                if self.data.close[0] < self.stop_price: self.order = self.close()
            elif self.position.size < 0: # Short
                self.lowest_price_since_entry = min(self.lowest_price_since_entry, self.data.low[0])
                new_stop = self.lowest_price_since_entry + (self.atr[0] * self.p.atr_stop_multiplier)
                self.stop_price = min(self.stop_price, new_stop)
                if self.data.close[0] > self.stop_price: self.order = self.close()