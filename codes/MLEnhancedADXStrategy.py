import backtrader as bt
import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MLEnhancedADXStrategy(bt.Strategy):
    params = (
        ('adx_period', 14),          # ADX period
        ('adx_threshold', 20),       # ADX threshold for trend strength
        ('rsi_period', 14),          # RSI period (sentiment proxy)
        ('bb_period', 7),           # Bollinger Bands period (volatility)
        ('atr_period', 14),          # ATR period
        ('lookback', 7),            # Lookback period for ML features
        ('retrain_freq', 30),       # Retrain model every N bars
        ('atr_multiplier', 3.0),     # ATR multiplier for trailing stops
        ('printlog', False),
    )

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f"{dt.isoformat()} - {txt}")

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume

        # Traditional indicators
        self.adx = bt.indicators.ADX(self.datas[0], period=self.params.adx_period)
        self.plusdi = bt.indicators.PlusDI(self.datas[0], period=self.params.adx_period)
        self.minusdi = bt.indicators.MinusDI(self.datas[0], period=self.params.adx_period)
        
        # Sentiment and volatility indicators
        self.rsi = bt.indicators.RSI(self.dataclose, period=self.params.rsi_period)
        self.bb = bt.indicators.BollingerBands(self.dataclose, period=self.params.bb_period)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
        
        # Additional features
        self.sma_short = bt.indicators.SMA(self.dataclose, period=10)
        self.sma_long = bt.indicators.SMA(self.dataclose, period=30)
        self.volume_sma = bt.indicators.SMA(self.datavolume, period=20)
        
        # ML components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_history = []
        self.target_history = []
        self.model_trained = False
        self.last_retrain = 0
        
        # Track orders
        self.order = None
        self.trail_order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED at {order.executed.price:.2f}")
            elif order.issell():
                self.log(f"SELL EXECUTED at {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")
            
        if order == self.order:
            self.order = None
        if order == self.trail_order:
            self.trail_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"Trade Profit: GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def cancel_trail(self):
        if self.trail_order:
            self.cancel(self.trail_order)
            self.trail_order = None

    def get_features(self):
        """Extract features for ML model"""
        if len(self) < self.params.lookback:
            return None
            
        features = []
        
        # Traditional ADX features
        features.append(self.adx[0])
        features.append(self.plusdi[0])
        features.append(self.minusdi[0])
        features.append(self.plusdi[0] - self.minusdi[0])  # DI spread
        
        # Price action features
        features.append((self.dataclose[0] - self.dataclose[-5]) / self.dataclose[-5] * 100)  # 5-day return
        features.append((self.dataclose[0] - self.dataclose[-10]) / self.dataclose[-10] * 100)  # 10-day return
        features.append(self.dataclose[0] / self.sma_short[0] - 1)  # Price vs short MA
        features.append(self.dataclose[0] / self.sma_long[0] - 1)   # Price vs long MA
        
        # Volatility features (ML-based volatility forecasting)
        features.append(self.atr[0] / self.dataclose[0] * 100)  # ATR as % of price
        bb_width = (self.bb.top[0] - self.bb.bot[0]) / self.bb.mid[0] * 100
        features.append(bb_width)  # BB width
        features.append((self.dataclose[0] - self.bb.mid[0]) / (self.bb.top[0] - self.bb.bot[0]))  # BB position
        
        # Volume features (sentiment proxy)
        features.append(self.datavolume[0] / self.volume_sma[0])  # Volume ratio
        
        # Sentiment indicators (RSI as VIX proxy)
        features.append(self.rsi[0])
        features.append(50 - self.rsi[0])  # RSI divergence from neutral
        
        # Recent volatility
        recent_highs = [self.datahigh[-i] for i in range(min(5, len(self)))]
        recent_lows = [self.datalow[-i] for i in range(min(5, len(self)))]
        if len(recent_highs) > 1:
            volatility = (max(recent_highs) - min(recent_lows)) / self.dataclose[0] * 100
            features.append(volatility)
        else:
            features.append(0)
        
        return np.array(features).reshape(1, -1)

    def get_target(self, future_bars=5):
        """Get target for training (future return)"""
        if len(self) < future_bars:
            return 0
        
        current_price = self.dataclose[0]
        future_price = self.dataclose[-future_bars] if len(self) > future_bars else current_price
        return_pct = (future_price - current_price) / current_price * 100
        
        # Convert to classification: 1 for up, -1 for down, 0 for sideways
        if return_pct > 1.0:
            return 1
        elif return_pct < -1.0:
            return -1
        else:
            return 0

    def train_model(self):
        """Train the ML model"""
        if len(self.feature_history) < 50:  # Need minimum data
            return False
            
        try:
            X = np.vstack(self.feature_history)
            y = np.array(self.target_history)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                min_samples_split=5
            )
            self.model.fit(X_scaled, y)
            self.model_trained = True
            self.log(f"ML Model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            self.log(f"Model training error: {e}")
            return False

    def get_ml_signal(self):
        """Get ML prediction"""
        if not self.model_trained:
            return 0
            
        features = self.get_features()
        if features is None:
            return 0
            
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            
            # Get prediction probability for confidence
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = max(proba)
            
            # Only use high-confidence predictions
            if confidence > 0.6:
                return prediction
            else:
                return 0
                
        except Exception as e:
            self.log(f"Prediction error: {e}")
            return 0

    def next(self):
        # Skip if order is pending
        if self.order:
            return

        # Handle trailing stops for existing positions
        if self.position:
            if not self.trail_order:
                if self.position.size > 0:
                    self.trail_order = self.sell(
                        exectype=bt.Order.StopTrail,
                        trailamount=self.atr[0] * self.params.atr_multiplier)
                elif self.position.size < 0:
                    self.trail_order = self.buy(
                        exectype=bt.Order.StopTrail,
                        trailamount=self.atr[0] * self.params.atr_multiplier)
            return

        # Ensure sufficient data
        if len(self) < self.params.lookback:
            return

        # Collect features for ML
        features = self.get_features()
        if features is not None and len(self) > 10:  # Need some history for target
            target = self.get_target()
            self.feature_history.append(features[0])
            self.target_history.append(target)
            
            # Keep only recent history
            if len(self.feature_history) > 500:
                self.feature_history = self.feature_history[-400:]
                self.target_history = self.target_history[-400:]

        # Retrain model periodically
        if (len(self) - self.last_retrain) > self.params.retrain_freq:
            if self.train_model():
                self.last_retrain = len(self)

        # Traditional ADX signal
        if self.adx[0] < self.params.adx_threshold:
            return

        # Traditional directional signals
        adx_long = self.plusdi[0] > self.minusdi[0]
        adx_short = self.minusdi[0] > self.plusdi[0]

        # Get ML ensemble signal
        ml_signal = self.get_ml_signal()

        # Combine traditional and ML signals
        long_signal = adx_long and ml_signal == 1
        short_signal = adx_short and ml_signal == -1

        if long_signal:
            self.log(f"ML ENHANCED LONG signal at {self.dataclose[0]:.2f}")
            self.log(f"ADX: {self.adx[0]:.2f}, +DI: {self.plusdi[0]:.2f}, ML: {ml_signal}")
            
            self.cancel_trail()
            if self.position and self.position.size < 0:
                self.order = self.buy()
            elif not self.position:
                self.order = self.buy()

        elif short_signal:
            self.log(f"ML ENHANCED SHORT signal at {self.dataclose[0]:.2f}")
            self.log(f"ADX: {self.adx[0]:.2f}, -DI: {self.minusdi[0]:.2f}, ML: {ml_signal}")
            
            self.cancel_trail()
            if self.position and self.position.size > 0:
                self.order = self.sell()
            elif not self.position:
                self.order = self.sell()

    def stop(self):
        self.log(f"Ending Portfolio Value: {self.broker.getvalue():.2f}", doprint=True)
        if self.model_trained:
            self.log(f"Model trained on {len(self.feature_history)} samples", doprint=True)

