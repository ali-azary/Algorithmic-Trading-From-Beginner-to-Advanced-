import backtrader as bt

class MomentumIgnitionStrategy(bt.Strategy):
    """
    Identifies periods of low price volatility and enters on a statistical
    breakout in the Rate of Change (ROC) momentum indicator, aligned
    with the long-term trend.
    """
    params = (
        # Volatility Filter
        ('consolidation_period', 30),
        ('consolidation_threshold', 0.1), # Max StdDev as % of price
        # Momentum Breakout
        ('roc_period', 7),
        ('roc_ma_period', 30),
        ('roc_breakout_std', 1.0), # ROC must exceed N StdDevs of its MA
        # Trend Filter
        ('trend_period', 30),
        # Risk Management
        ('atr_period', 7),
        ('atr_stop_multiplier', 1.0),
    )

    def __init__(self):
        self.order = None

        # --- Indicators ---
        self.price_stddev = bt.indicators.StandardDeviation(self.data.close, period=self.p.consolidation_period)
        self.roc = bt.indicators.RateOfChange(self.data.close, period=self.p.roc_period)
        self.roc_ma = bt.indicators.SimpleMovingAverage(self.roc, period=self.p.roc_ma_period)
        self.roc_stddev = bt.indicators.StandardDeviation(self.roc, period=self.p.roc_ma_period)
        self.trend_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.trend_period)
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
            # --- Filter Conditions ---
            # 1. Is the market consolidating (low price volatility)?
            is_consolidating = (self.price_stddev[0] / self.data.close[0]) < self.p.consolidation_threshold
            
            # 2. Is the macro trend aligned?
            is_macro_uptrend = self.data.close[0] > self.trend_sma[0]
            is_macro_downtrend = self.data.close[0] < self.trend_sma[0]

            if is_consolidating:
                # 3. Has momentum "ignited" with a statistical breakout?
                roc_upper_band = self.roc_ma[0] + (self.roc_stddev[0] * self.p.roc_breakout_std)
                roc_lower_band = self.roc_ma[0] - (self.roc_stddev[0] * self.p.roc_breakout_std)
                
                is_mom_breakout_up = self.roc[0] > roc_upper_band
                is_mom_breakout_down = self.roc[0] < roc_lower_band
                
                # --- Entry Logic ---
                if is_macro_uptrend and is_mom_breakout_up:
                    self.order = self.buy()
                elif is_macro_downtrend and is_mom_breakout_down:
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