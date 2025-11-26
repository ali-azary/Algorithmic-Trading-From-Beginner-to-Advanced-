import backtrader as bt
import numpy as np

class RegimeFilteredTrendStrategy(bt.Strategy):
    params = (
        ('ma_fast', 7),            # Fast moving average
        ('ma_slow', 30),            # Slow moving average
        ('adx_period', 14),         # ADX period
        ('adx_trending_threshold', 20), # ADX threshold for trending regime
        ('bb_period', 7),          # Bollinger Bands period
        ('bb_width_threshold', 0.01), # BB width threshold for trending (3%)
        ('volatility_lookback', 7), # Volatility measurement period
        ('vol_trending_threshold', 0.01), # Volatility threshold for trending
        ('atr_period', 14),         # ATR period
        ('trail_atr_mult', 3.0),    # Trailing stop multiplier (trending)
        ('range_atr_mult', 1.),    # Trailing stop multiplier (ranging)
        ('max_position_pct', 0.80), # Maximum position size
        ('min_position_pct', 0.20), # Minimum position size
        ('regime_confirmation', 3), # Bars to confirm regime change
    )
    
    def __init__(self):
        self.dataclose = self.datas[0].close
        
        # Moving averages for trend following
        self.ma_fast = bt.indicators.SMA(period=self.params.ma_fast)
        self.ma_slow = bt.indicators.SMA(period=self.params.ma_slow)
        
        # Regime classification indicators
        self.adx = bt.indicators.ADX(period=self.params.adx_period)
        self.bb = bt.indicators.BollingerBands(period=self.params.bb_period)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        
        # Track orders and regime state
        self.order = None
        self.trail_order = None
        
        # Regime tracking
        self.current_regime = "unknown"  # "trending", "ranging", "unknown"
        self.regime_history = []
        self.regime_confidence = 0
        
        # Volatility tracking
        self.volatility_history = []

    def cancel_trail(self):
        if self.trail_order:
            self.cancel(self.trail_order)
            self.trail_order = None

    def calculate_volatility(self):
        """Calculate normalized volatility measure"""
        if len(self.atr) == 0 or self.dataclose[0] <= 0:
            return 0
        
        try:
            return self.atr[0] / self.dataclose[0]
        except:
            return 0

    def classify_market_regime(self):
        """Classify current market regime using multiple indicators"""
        if (len(self.adx) == 0 or len(self.bb) == 0 or 
            len(self.volatility_history) < 5):
            return "unknown", 0
        
        try:
            trending_signals = 0
            total_signals = 0
            
            # ADX Signal
            total_signals += 1
            if self.adx[0] > self.params.adx_trending_threshold:
                trending_signals += 1
            
            # Bollinger Band Width Signal
            total_signals += 1
            bb_width = (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0]
            if bb_width > self.params.bb_width_threshold:
                trending_signals += 1
            
            # Volatility Signal
            total_signals += 1
            current_vol = self.volatility_history[-1]
            if current_vol > self.params.vol_trending_threshold:
                trending_signals += 1
            
            # Moving Average Separation Signal
            total_signals += 1
            ma_separation = abs(self.ma_fast[0] - self.ma_slow[0]) / self.ma_slow[0]
            if ma_separation > 0.02:  # 2% separation
                trending_signals += 1
            
            # Calculate confidence
            confidence = trending_signals / total_signals
            
            # Classify regime
            if confidence >= 0.75:  # 3+ out of 4 signals
                regime = "trending"
            elif confidence <= 0.25:  # 1 or fewer signals
                regime = "ranging"
            else:
                regime = "uncertain"
            
            return regime, confidence
            
        except Exception as e:
            return "unknown", 0

    def update_regime_state(self):
        """Update regime state with confirmation logic"""
        new_regime, confidence = self.classify_market_regime()
        
        # Add to regime history
        self.regime_history.append(new_regime)
        if len(self.regime_history) > self.params.regime_confirmation * 2:
            self.regime_history = self.regime_history[-self.params.regime_confirmation * 2:]
        
        # Confirm regime change only if consistent over multiple bars
        if len(self.regime_history) >= self.params.regime_confirmation:
            recent_regimes = self.regime_history[-self.params.regime_confirmation:]
            
            # Check for consistency
            if all(r == new_regime for r in recent_regimes):
                if self.current_regime != new_regime:
                    # Regime change confirmed
                    self.current_regime = new_regime
                    self.regime_confidence = confidence
            else:
                # Mixed signals, keep current regime but update confidence
                self.regime_confidence = confidence

    def calculate_regime_position_size(self):
        """Calculate position size based on regime and confidence"""
        try:
            base_size = self.params.max_position_pct
            
            if self.current_regime == "trending":
                # Full size in trending markets
                size_factor = 1.0
            elif self.current_regime == "ranging":
                # Reduced size in ranging markets
                size_factor = 0.3
            else:  # uncertain or unknown
                # Minimal size in uncertain markets
                size_factor = 0.1
            
            # Adjust by confidence
            confidence_factor = max(0.5, self.regime_confidence)
            
            final_size = base_size * size_factor * confidence_factor
            
            return max(self.params.min_position_pct, 
                      min(self.params.max_position_pct, final_size))
            
        except Exception as e:
            return self.params.min_position_pct

    def get_adaptive_stop_multiplier(self):
        """Get ATR multiplier based on regime and volatility"""
        if self.current_regime == "trending":
            # Wider stops in trending markets
            base_mult = self.params.trail_atr_mult
            
            # Adjust for volatility
            if len(self.volatility_history) >= 5:
                current_vol = self.volatility_history[-1]
                avg_vol = np.mean(self.volatility_history[-10:])
                
                if current_vol > avg_vol * 1.2:  # High volatility
                    return base_mult * 1.3
                elif current_vol < avg_vol * 0.8:  # Low volatility
                    return base_mult * 0.8
            
            return base_mult
        else:
            # Tighter stops in ranging markets
            return self.params.range_atr_mult

    def should_trade_trend_following(self):
        """Determine if trend following should be active"""
        if self.current_regime == "trending":
            return True
        elif self.current_regime == "ranging":
            return False  # No trend following in ranging markets
        else:
            # Uncertain regime - very cautious
            return self.regime_confidence > 0.7
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        # log optional
        # print(self.data.datetime.date(0), order.getstatusname())
        self.order = None  # <- CRITICAL: free the lock when done/rejected/canceled
    
    def notify_trade(self, trade):
        if trade.isclosed:
            # optional: print realized PnL
            # print(f"{self.data.datetime.date(0)} | PnL: {trade.pnlcomm:.2f}")
            pass

    
            
    
    def next(self):
        # Skip if order is pending
        if self.order:
            return
    
        # Update volatility tracking
        current_vol = self.calculate_volatility()
        if current_vol > 0:
            self.volatility_history.append(current_vol)
            if len(self.volatility_history) > self.params.volatility_lookback:
                self.volatility_history = self.volatility_history[-self.params.volatility_lookback:]
    
        # Update regime classification
        self.update_regime_state()
    
        # Handle existing positions with adaptive stops
        if self.position:
            stop_multiplier = self.get_adaptive_stop_multiplier()
    
            # (re)arm/refresh trailing stop if missing
            if not self.trail_order:
                if self.position.size > 0:
                    self.trail_order = self.sell(
                        exectype=bt.Order.StopTrail,
                        trailamount=self.atr[0] * stop_multiplier,
                        size=self.position.size              # <- add size
                    )
                else:  # short
                    self.trail_order = self.buy(
                        exectype=bt.Order.StopTrail,
                        trailamount=self.atr[0] * stop_multiplier,
                        size=abs(self.position.size)         # <- add size
                    )
    
            # ALSO exit on signal flip or regime change (don’t rely only on stop)
            if self.position.size > 0:
                if (self.ma_fast[0] < self.ma_slow[0]) or (self.current_regime != "trending"):
                    self.cancel_trail()
                    self.order = self.close()                # <- create explicit exit
            else:  # short
                if (self.ma_fast[0] > self.ma_slow[0]) or (self.current_regime != "trending"):
                    self.cancel_trail()
                    self.order = self.close()
    
            return
    
        # Ensure sufficient data
        required_bars = max(self.params.ma_slow, self.params.adx_period, self.params.bb_period)
        if len(self) < required_bars:
            return
    
        # Check if we should engage in trend following
        if not self.should_trade_trend_following():
            return  # Stay out during non-trending regimes
    
        # Moving average crossover signals (only in trending regimes)
        ma_bullish_cross = (self.ma_fast[0] > self.ma_slow[0] and self.ma_fast[-1] <= self.ma_slow[-1])
        ma_bearish_cross = (self.ma_fast[0] < self.ma_slow[0] and self.ma_fast[-1] >= self.ma_slow[-1])
    
        # Position sizing based on regime
        position_size_pct = self.calculate_regime_position_size()
        current_price = float(self.dataclose[0])
    
        # LONG ENTRY
        if ma_bullish_cross:
            self.cancel_trail()
            cash = float(self.broker.getcash())
            target_value = cash * position_size_pct
            shares = target_value / max(current_price, 1e-12)
            self.order = self.buy(size=shares)               # keep float sizing for crypto
    
        # SHORT ENTRY
        elif ma_bearish_cross:
            self.cancel_trail()
            cash = float(self.broker.getcash())
            target_value = cash * position_size_pct
            shares = target_value / max(current_price, 1e-12)
            self.order = self.sell(size=shares)
    
        # Alternative entry: strong continuation
        elif (self.current_regime == "trending" and self.regime_confidence > 0.8):
            ma_spread = (self.ma_fast[0] - self.ma_slow[0]) / self.ma_slow[0]
            if ma_spread > 0.03:
                cash = float(self.broker.getcash())
                target_value = cash * (position_size_pct * 0.7)
                shares = target_value / max(current_price, 1e-12)
                self.order = self.buy(size=shares)
            elif ma_spread < -0.03:
                cash = float(self.broker.getcash())
                target_value = cash * (position_size_pct * 0.7)
                shares = target_value / max(current_price, 1e-12)
                self.order = self.sell(size=shares)
