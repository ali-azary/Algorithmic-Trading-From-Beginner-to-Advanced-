import backtrader as bt

class IchimokuCloudStrategy(bt.Strategy):
    """
    Trades on a confirmed breakout from the Ichimoku Cloud (Kumo).
    1.  Price breaks out of the Kumo.
    2.  Tenkan/Kijun cross confirms momentum.
    3.  Chikou Span confirms the trend.
    4.  Exit is managed with a trailing stop-loss.
    """
    params = (
        # Default Ichimoku parameters
        ('tenkan', 7),
        ('kijun', 14),
        ('senkou', 30),
        ('senkou_lead', 14),  # How far forward to plot the cloud
        ('chikou', 14),      # How far back to plot the lagging span
        # Strategy parameters
        ('trail_percent', 0.02), # Trailing stop loss of 4%
    )

    def __init__(self):
        self.order = None

        # Add the Ichimoku indicator with its parameters
        self.ichimoku = bt.indicators.Ichimoku(
            self.datas[0],
            tenkan=self.p.tenkan,
            kijun=self.p.kijun,
            senkou=self.p.senkou,
            senkou_lead=self.p.senkou_lead,
            chikou=self.p.chikou
        )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                # Place a trailing stop for the long position
                self.sell(exectype=bt.Order.StopTrail, trailpercent=self.p.trail_percent)
            elif order.issell():
                # Place a trailing stop for the short position
                self.buy(exectype=bt.Order.StopTrail, trailpercent=self.p.trail_percent)
        
        self.order = None

    def next(self):
        # Check for pending orders
        if self.order:
            return
        
        # Check if we are in a position
        if not self.position:
            # --- Bullish Entry Conditions ---
            # 1. Price is above both lines of the Kumo cloud
            is_above_cloud = (self.data.close[0] > self.ichimoku.senkou_span_a[0] and
                              self.data.close[0] > self.ichimoku.senkou_span_b[0])
            
            # 2. Tenkan-sen is above Kijun-sen
            is_tk_cross_bullish = self.ichimoku.tenkan_sen[0] > self.ichimoku.kijun_sen[0]
            
            # 3. Chikou Span is above the price from 26 periods ago
            is_chikou_bullish = self.ichimoku.chikou_span[0] > self.data.high[-self.p.chikou]

            if is_above_cloud and is_tk_cross_bullish and is_chikou_bullish:
                self.order = self.buy()

            # --- Bearish Entry Conditions ---
            # 1. Price is below both lines of the Kumo cloud
            is_below_cloud = (self.data.close[0] < self.ichimoku.senkou_span_a[0] and
                              self.data.close[0] < self.ichimoku.senkou_span_b[0])
            
            # 2. Tenkan-sen is below Kijun-sen
            is_tk_cross_bearish = self.ichimoku.tenkan_sen[0] < self.ichimoku.kijun_sen[0]
            
            # 3. Chikou Span is below the price from 26 periods ago
            is_chikou_bearish = self.ichimoku.chikou_span[0] < self.data.low[-self.p.chikou]

            if is_below_cloud and is_tk_cross_bearish and is_chikou_bearish:
                self.order = self.sell()