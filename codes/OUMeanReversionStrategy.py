import backtrader as bt
import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

class OUMeanReversionStrategy(bt.Strategy):
    """
    Ornstein-Uhlenbeck Mean Reversion Strategy
    
    The strategy estimates OU process parameters over a rolling window
    and generates trading signals based on deviations from the estimated mean.
    """
    
    params = (
        ('lookback', 30),           # Rolling window for OU parameter estimation
        ('sma_period', 30),         # sma period for trend
        ('entry_threshold', 1.),   # Z-score threshold for entry
        ('exit_threshold', 0.),    # Z-score threshold for exit
        ('printlog', False),        # Print trade logs
    )
    
    def __init__(self):
        # Data feeds
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.dataclose, period=self.params.sma_period)  # Add this line
        
        # Track our position
        self.order = None
        self.position_type = None  # 'long', 'short', or None
        
        # Store OU parameters and signals
        self.ou_params = []
        self.z_scores = []
        
    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price: {order.executed.price:.4f}, '
                         f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED: Price: {order.executed.price:.4f}, '
                         f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        self.order = None
    
    def estimate_ou_parameters(self, log_prices):
        """
        Estimate Ornstein-Uhlenbeck parameters using OLS regression
        
        OU process: dX = θ(μ - X)dt + σdW
        Discretized: X_t - X_{t-1} = θμΔt - θX_{t-1}Δt + ε_t
        
        Returns: (mu, theta, sigma, equilibrium_std)
        """
        if len(log_prices) < 10:  # Need minimum data points
            return None, None, None, None
            
        # Prepare regression data
        x_lag = log_prices[:-1]  # X_{t-1}
        dx = np.diff(log_prices)  # X_t - X_{t-1}
        
        try:
            # OLS regression: dx = alpha + beta * x_lag
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_lag, dx)
            
            # Convert to OU parameters (assuming dt = 1)
            theta = -slope
            mu = intercept / theta if theta > 1e-6 else np.mean(log_prices)
            
            # Estimate sigma from residuals
            residuals = dx - (intercept + slope * x_lag)
            sigma = np.std(residuals)
            
            # Equilibrium standard deviation
            equilibrium_std = sigma / np.sqrt(2 * theta) if theta > 1e-6 else sigma
            
            return mu, theta, sigma, equilibrium_std
            
        except Exception as e:
            return None, None, None, None
    
    def next(self):
        """Main strategy logic called on each bar"""
        
        # Need enough data for parameter estimation
        if len(self.dataclose) < self.params.lookback:
            return
            
        # Get recent log prices for parameter estimation
        recent_log_prices = np.array([np.log(self.dataclose[-i]) for i in range(self.params.lookback-1, -1, -1)])
        
        # Estimate OU parameters
        mu, theta, sigma, eq_std = self.estimate_ou_parameters(recent_log_prices)
        
        if mu is None or eq_std is None or eq_std <= 0:
            return
            
        # Calculate current deviation and z-score
        current_log_price = np.log(self.dataclose[0])
        deviation = current_log_price - mu
        z_score = deviation / eq_std
        
        # Store for analysis
        self.ou_params.append({'mu': mu, 'theta': theta, 'sigma': sigma, 'eq_std': eq_std})
        self.z_scores.append(z_score)
        
        self.log(f'Close: {self.dataclose[0]:.4f}, Log Price: {current_log_price:.4f}, '
                 f'μ: {mu:.4f}, Z-Score: {z_score:.2f}')
        
        # Skip if we have a pending order
        if self.order:
            return
        
        # Trading logic
        if not self.position:  # No position
            if z_score < -self.params.entry_threshold and self.dataclose[0] > self.sma[0]:
                # Price below mean AND uptrending - go long (expect reversion up)
                self.log(f'LONG SIGNAL: Z-Score {z_score:.2f}')
                self.order = self.buy()
                self.position_type = 'long'
                
            elif z_score > self.params.entry_threshold and self.dataclose[0] < self.sma[0]:
                # Price above mean AND downtrending - go short (expect reversion down)
                self.log(f'SHORT SIGNAL: Z-Score {z_score:.2f}')
                self.order = self.sell()
                self.position_type = 'short'
                
        else:  # We have a position
            if self.position_type == 'long' and z_score > -self.params.exit_threshold:
                # Exit long position
                self.log(f'EXIT LONG: Z-Score {z_score:.2f}')
                self.order = self.sell()
                self.position_type = None
                
            elif self.position_type == 'short' and z_score < self.params.exit_threshold:
                # Exit short position
                self.log(f'EXIT SHORT: Z-Score {z_score:.2f}')
                self.order = self.buy()
                self.position_type = None

