import backtrader as bt
import numpy as np
from scipy.optimize import minimize

class QuantileRegression:
    """Quantile Regression implementation for channel estimation"""
    
    def __init__(self, tau=0.5):
        self.tau = tau  # Quantile level (0.5 = median)
        
    def quantile_loss(self, y_true, y_pred):
        """Quantile loss function (pinball loss)"""
        residual = y_true - y_pred
        return np.mean(np.maximum(self.tau * residual, (self.tau - 1) * residual))
    
    def fit(self, X, y):
        """Fit quantile regression using optimization"""
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        
        # Initialize coefficients
        initial_params = np.zeros(n_features + 1)  # +1 for intercept
        
        def objective(params):
            """Objective function to minimize"""
            if len(X.shape) == 1:
                y_pred = params[0] + params[1] * X
            else:
                y_pred = params[0] + np.dot(X, params[1:])
            return self.quantile_loss(y, y_pred)
        
        # Optimize
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B')
            self.coef_ = result.x
            return self
        except:
            # Fallback to simple quantile
            self.coef_ = np.array([np.quantile(y, self.tau), 0])
            return self
    
    def predict(self, X):
        """Predict using fitted model"""
        if not hasattr(self, 'coef_'):
            raise ValueError("Model must be fitted before prediction")
            
        if len(X.shape) == 1:
            return self.coef_[0] + self.coef_[1] * X
        else:
            return self.coef_[0] + np.dot(X, self.coef_[1:])

class QuantileChannelStrategy(bt.Strategy):
    params = (
        ('lookback_period', 30),      # Lookback for channel estimation
        ('upper_quantile', 0.8),      # Upper channel quantile (80th percentile)
        ('lower_quantile', 0.2),      # Lower channel quantile (20th percentile)
        ('trend_quantile', 0.5),      # Trend line quantile (median)
        ('breakout_threshold', 1.01), # Breakout confirmation (2% above/below)
        ('stop_loss_pct', 0.05),      # 8% stop loss
        ('rebalance_period', 7),      # Daily rebalancing
        ('min_channel_width', 0.01),  # Minimum 2% channel width
        ('volume_confirm', False),    # Volume confirmation (if available)
    )
    
    def __init__(self):
        # Price and time data
        self.prices = []
        self.time_indices = []
        
        # Channel estimates
        self.upper_channel = []
        self.lower_channel = []
        self.trend_line = []
        self.channel_width = []
        
        # Quantile regression models
        self.upper_qr = QuantileRegression(tau=self.params.upper_quantile)
        self.lower_qr = QuantileRegression(tau=self.params.lower_quantile)
        self.trend_qr = QuantileRegression(tau=self.params.trend_quantile)
        
        # Trading variables
        self.rebalance_counter = 0
        self.stop_price = 0
        self.trade_count = 0
        self.breakout_direction = 0  # 1=upper, -1=lower, 0=none
        self.channel_confidence = 0
        
        # Track breakouts
        self.upper_breakouts = 0
        self.lower_breakouts = 0
        self.false_breakouts = 0
        
    def estimate_channels(self):
        """Estimate quantile regression channels"""
        if len(self.prices) < self.params.lookback_period:
            return None, None, None, 0
        
        # Get recent data
        recent_prices = np.array(self.prices[-self.params.lookback_period:])
        recent_times = np.array(self.time_indices[-self.params.lookback_period:])
        
        # Normalize time for better numerical stability
        time_normalized = (recent_times - recent_times[0]) / (recent_times[-1] - recent_times[0] + 1e-8)
        
        try:
            # Fit quantile regressions
            self.upper_qr.fit(time_normalized, recent_prices)
            self.lower_qr.fit(time_normalized, recent_prices)
            self.trend_qr.fit(time_normalized, recent_prices)
            
            # Predict current levels
            current_time_norm = 1.0  # Current time (end of normalized period)
            
            upper_level = self.upper_qr.predict(np.array([current_time_norm]))[0]
            lower_level = self.lower_qr.predict(np.array([current_time_norm]))[0]
            trend_level = self.trend_qr.predict(np.array([current_time_norm]))[0]
            
            # Calculate channel width and confidence
            channel_width = (upper_level - lower_level) / trend_level
            
            # Ensure minimum channel width
            if channel_width < self.params.min_channel_width:
                mid_price = (upper_level + lower_level) / 2
                half_width = mid_price * self.params.min_channel_width / 2
                upper_level = mid_price + half_width
                lower_level = mid_price - half_width
                channel_width = self.params.min_channel_width
            
            # Channel confidence based on data dispersion
            price_std = np.std(recent_prices)
            expected_width = 2 * price_std / np.mean(recent_prices)  # 2-sigma as reference
            confidence = min(1.0, expected_width / (channel_width + 1e-8))
            
            return upper_level, lower_level, trend_level, confidence
            
        except Exception as e:
            # Fallback to simple quantiles
            upper_level = np.quantile(recent_prices, self.params.upper_quantile)
            lower_level = np.quantile(recent_prices, self.params.lower_quantile)
            trend_level = np.quantile(recent_prices, self.params.trend_quantile)
            confidence = 0.5
            
            return upper_level, lower_level, trend_level, confidence
    
    def detect_breakout(self, current_price, upper_channel, lower_channel):
        """Detect channel breakout with confirmation"""
        breakout = 0
        
        # Upper breakout
        if current_price > upper_channel * self.params.breakout_threshold:
            breakout = 1
            self.upper_breakouts += 1
            
        # Lower breakout  
        elif current_price < lower_channel / self.params.breakout_threshold:
            breakout = -1
            self.lower_breakouts += 1
            
        return breakout
    
    def next(self):
        # Collect price and time data
        current_price = self.data.close[0]
        current_time = len(self.prices)
        
        self.prices.append(current_price)
        self.time_indices.append(current_time)
        
        # Keep only recent history
        if len(self.prices) > self.params.lookback_period * 2:
            self.prices = self.prices[-self.params.lookback_period * 2:]
            self.time_indices = self.time_indices[-self.params.lookback_period * 2:]
        
        # Estimate channels
        upper_channel, lower_channel, trend_line, confidence = self.estimate_channels()
        
        if upper_channel is None:
            return  # Not enough data yet
        
        # Store channel estimates
        self.upper_channel.append(upper_channel)
        self.lower_channel.append(lower_channel)
        self.trend_line.append(trend_line)
        self.channel_confidence = confidence
        
        # Calculate channel width
        width = (upper_channel - lower_channel) / trend_line
        self.channel_width.append(width)
        
        # Rebalancing logic
        self.rebalance_counter += 1
        if self.rebalance_counter < self.params.rebalance_period:
            # Check stop loss
            if self.position.size > 0 and current_price <= self.stop_price:
                self.close()
            elif self.position.size < 0 and current_price >= self.stop_price:
                self.close()
            return
        
        # Reset rebalance counter
        self.rebalance_counter = 0
        
        # Detect breakout
        breakout = self.detect_breakout(current_price, upper_channel, lower_channel)
        
        # Current position
        current_pos = 0
        if self.position.size > 0:
            current_pos = 1
        elif self.position.size < 0:
            current_pos = -1
        
        # Trading logic with channel confirmation
        if breakout != 0 and confidence > 0.3:  # Require minimum confidence
            # Close existing position if direction changed
            if current_pos != 0 and current_pos != breakout:
                self.close()
                current_pos = 0
            
            # Open new position on breakout
            if current_pos == 0:
                if breakout == 1:  # Upper breakout - go long
                    self.buy()
                    self.stop_price = lower_channel  # Use lower channel as stop
                    self.trade_count += 1
                    self.breakout_direction = 1
                    
                elif breakout == -1:  # Lower breakout - go short
                    self.sell()
                    self.stop_price = upper_channel  # Use upper channel as stop
                    self.trade_count += 1
                    self.breakout_direction = -1
        
        # Exit on return to channel (mean reversion)
        elif self.position.size != 0:
            in_channel = lower_channel <= current_price <= upper_channel
            
            if in_channel and abs(current_price - trend_line) / trend_line < 0.02:
                self.close()
        
        # Update trailing stops
        if self.position.size > 0:  # Long position
            new_stop = max(self.stop_price, lower_channel)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                
        elif self.position.size < 0:  # Short position
            new_stop = min(self.stop_price, upper_channel)
            if new_stop < self.stop_price:
                self.stop_price = new_stop