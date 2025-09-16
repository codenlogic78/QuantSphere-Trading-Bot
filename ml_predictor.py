#!/usr/bin/env python3
"""
QuantSphere AI Trading Platform - Machine Learning Predictor
Advanced ML models for price prediction and trading signals
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

class MLPredictor:
    """Machine Learning predictor for stock price forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower',
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'volatility', 'price_change', 'volume_change'
        ]
        
        self.prediction_horizon = 5  # Predict 5 days ahead
        self.model_performance = {}
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical features for ML models"""
        features_df = df.copy()
        
        # Basic price features
        features_df['price_change'] = features_df['close'].pct_change()
        features_df['volume_change'] = features_df['volume'].pct_change()
        features_df['volatility'] = features_df['close'].rolling(window=20).std()
        
        # Moving averages
        features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
        features_df['sma_10'] = features_df['close'].rolling(window=10).mean()
        features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
        features_df['ema_12'] = features_df['close'].ewm(span=12).mean()
        features_df['ema_26'] = features_df['close'].ewm(span=26).mean()
        
        # Technical indicators (simplified versions)
        features_df['rsi'] = self.calculate_rsi(features_df['close'])
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_sma = features_df['close'].rolling(window=bb_period).mean()
        bb_std_dev = features_df['close'].rolling(window=bb_period).std()
        features_df['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
        features_df['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
        
        # Price position relative to moving averages
        features_df['price_vs_sma20'] = features_df['close'] / features_df['sma_20'] - 1
        features_df['price_vs_sma10'] = features_df['close'] / features_df['sma_10'] - 1
        
        # Volume indicators
        features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
        
        return features_df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML training"""
        # Generate features
        features_df = self.generate_features(df)
        
        # Create target variable (future price)
        features_df['target'] = features_df[target_col].shift(-self.prediction_horizon)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < 50:  # Minimum data requirement
            raise ValueError("Insufficient data for training")
        
        # Select feature columns that exist in the dataframe
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        
        # Add lag features and other generated features
        additional_features = [col for col in features_df.columns 
                             if col.startswith(('close_lag_', 'volume_lag_', 'price_vs_', 'volume_ratio'))]
        available_features.extend(additional_features)
        
        X = features_df[available_features].values
        y = features_df['target'].values
        
        return X, y
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, symbol: str) -> Dict:
        """Train Random Forest model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Store model and scaler
        self.models[f'{symbol}_rf'] = model
        self.scalers[f'{symbol}_rf'] = scaler
        
        performance = {
            'model_type': 'RandomForest',
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(range(len(model.feature_importances_)), 
                                         model.feature_importances_))
        }
        
        self.model_performance[f'{symbol}_rf'] = performance
        return performance
    
    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray, symbol: str) -> Dict:
        """Train Gradient Boosting model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        # Store model and scaler
        self.models[f'{symbol}_gb'] = model
        self.scalers[f'{symbol}_gb'] = scaler
        
        performance = {
            'model_type': 'GradientBoosting',
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(range(len(model.feature_importances_)), 
                                         model.feature_importances_))
        }
        
        self.model_performance[f'{symbol}_gb'] = performance
        return performance
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, symbol: str) -> Dict:
        """Train LSTM model (if TensorFlow is available)"""
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow not available for LSTM training'}
        
        # Reshape data for LSTM (samples, time steps, features)
        # For simplicity, using last 60 time steps
        time_steps = 60
        if len(X) < time_steps + self.prediction_horizon:
            return {'error': 'Insufficient data for LSTM training'}
        
        # Prepare LSTM data
        X_lstm = []
        y_lstm = []
        
        for i in range(time_steps, len(X)):
            X_lstm.append(X[i-time_steps:i])
            y_lstm.append(y[i])
        
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        # Split data
        split_idx = int(len(X_lstm) * 0.8)
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
        
        # Scale data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test_scaled, y_test),
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled, verbose=0)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and scaler
        self.models[f'{symbol}_lstm'] = model
        self.scalers[f'{symbol}_lstm'] = scaler
        
        performance = {
            'model_type': 'LSTM',
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'training_loss': history.history['loss'][-1],
            'validation_loss': history.history['val_loss'][-1]
        }
        
        self.model_performance[f'{symbol}_lstm'] = performance
        return performance
    
    def train_ensemble(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Train ensemble of models"""
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            
            results = {}
            
            # Train Random Forest
            print(f"Training Random Forest for {symbol}...")
            rf_performance = self.train_random_forest(X, y, symbol)
            results['random_forest'] = rf_performance
            
            # Train Gradient Boosting
            print(f"Training Gradient Boosting for {symbol}...")
            gb_performance = self.train_gradient_boosting(X, y, symbol)
            results['gradient_boosting'] = gb_performance
            
            # Train LSTM if available
            if TENSORFLOW_AVAILABLE:
                print(f"Training LSTM for {symbol}...")
                lstm_performance = self.train_lstm(X, y, symbol)
                results['lstm'] = lstm_performance
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def predict_price(self, df: pd.DataFrame, symbol: str, model_type: str = 'ensemble') -> Dict:
        """Make price predictions"""
        try:
            # Generate features for the latest data
            features_df = self.generate_features(df)
            
            # Get the latest feature values
            available_features = [col for col in self.feature_columns if col in features_df.columns]
            additional_features = [col for col in features_df.columns 
                                 if col.startswith(('close_lag_', 'volume_lag_', 'price_vs_', 'volume_ratio'))]
            available_features.extend(additional_features)
            
            latest_features = features_df[available_features].iloc[-1:].values
            
            predictions = {}
            
            if model_type == 'ensemble' or model_type == 'all':
                # Get predictions from all available models
                model_predictions = []
                
                # Random Forest prediction
                if f'{symbol}_rf' in self.models:
                    scaler = self.scalers[f'{symbol}_rf']
                    model = self.models[f'{symbol}_rf']
                    scaled_features = scaler.transform(latest_features)
                    rf_pred = model.predict(scaled_features)[0]
                    predictions['random_forest'] = rf_pred
                    model_predictions.append(rf_pred)
                
                # Gradient Boosting prediction
                if f'{symbol}_gb' in self.models:
                    scaler = self.scalers[f'{symbol}_gb']
                    model = self.models[f'{symbol}_gb']
                    scaled_features = scaler.transform(latest_features)
                    gb_pred = model.predict(scaled_features)[0]
                    predictions['gradient_boosting'] = gb_pred
                    model_predictions.append(gb_pred)
                
                # LSTM prediction (more complex due to sequence requirement)
                if f'{symbol}_lstm' in self.models and TENSORFLOW_AVAILABLE:
                    # For LSTM, we need the last 60 time steps
                    if len(features_df) >= 60:
                        lstm_features = features_df[available_features].iloc[-60:].values
                        scaler = self.scalers[f'{symbol}_lstm']
                        model = self.models[f'{symbol}_lstm']
                        
                        # Reshape for LSTM
                        lstm_features = lstm_features.reshape(1, 60, -1)
                        scaled_features = scaler.transform(lstm_features.reshape(-1, lstm_features.shape[-1])).reshape(lstm_features.shape)
                        lstm_pred = model.predict(scaled_features, verbose=0)[0][0]
                        predictions['lstm'] = lstm_pred
                        model_predictions.append(lstm_pred)
                
                # Ensemble prediction (average)
                if model_predictions:
                    predictions['ensemble'] = np.mean(model_predictions)
                    predictions['ensemble_std'] = np.std(model_predictions)
            
            else:
                # Single model prediction
                model_key = f'{symbol}_{model_type}'
                if model_key in self.models:
                    scaler = self.scalers[model_key]
                    model = self.models[model_key]
                    scaled_features = scaler.transform(latest_features)
                    pred = model.predict(scaled_features)[0]
                    predictions[model_type] = pred
            
            # Add current price for comparison
            current_price = df['close'].iloc[-1]
            predictions['current_price'] = current_price
            
            # Calculate predicted change
            if 'ensemble' in predictions:
                predicted_change = (predictions['ensemble'] - current_price) / current_price * 100
                predictions['predicted_change_pct'] = predicted_change
                
                # Generate trading signal based on prediction
                if predicted_change > 2:  # 2% threshold
                    predictions['signal'] = 'STRONG_BUY'
                elif predicted_change > 0.5:
                    predictions['signal'] = 'BUY'
                elif predicted_change < -2:
                    predictions['signal'] = 'STRONG_SELL'
                elif predicted_change < -0.5:
                    predictions['signal'] = 'SELL'
                else:
                    predictions['signal'] = 'HOLD'
            
            return predictions
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_performance(self, symbol: str = None) -> Dict:
        """Get model performance metrics"""
        if symbol:
            return {k: v for k, v in self.model_performance.items() if symbol in k}
        return self.model_performance
    
    def save_models(self, symbol: str, filepath: str = None):
        """Save trained models to disk"""
        if filepath is None:
            filepath = f"models_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        
        model_data = {
            'models': {k: v for k, v in self.models.items() if symbol in k},
            'scalers': {k: v for k, v in self.scalers.items() if symbol in k},
            'performance': {k: v for k, v in self.model_performance.items() if symbol in k},
            'feature_columns': self.feature_columns,
            'prediction_horizon': self.prediction_horizon
        }
        
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        model_data = joblib.load(filepath)
        
        self.models.update(model_data['models'])
        self.scalers.update(model_data['scalers'])
        self.model_performance.update(model_data['performance'])
        self.feature_columns = model_data.get('feature_columns', self.feature_columns)
        self.prediction_horizon = model_data.get('prediction_horizon', self.prediction_horizon)

# Test the ML predictor
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic stock data
    base_price = 100
    prices = [base_price]
    volumes = []
    
    for i in range(1, len(dates)):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
        
        # Volume inversely correlated with price change
        volume = np.random.randint(100000, 1000000) * (1 + abs(change) * 5)
        volumes.append(int(volume))
    
    volumes.append(volumes[-1])  # Add last volume
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'date': dates,
        'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.03) for p in prices],
        'low': [p * np.random.uniform(0.97, 1.00) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    test_data.set_index('date', inplace=True)
    
    print("=== QUANTSPHERE ML PREDICTOR TEST ===")
    print(f"Training on {len(test_data)} days of data")
    
    # Initialize predictor
    predictor = MLPredictor()
    
    # Train models
    print("\nTraining ensemble models...")
    results = predictor.train_ensemble(test_data, 'TEST')
    
    print("\nTraining Results:")
    for model_type, performance in results.items():
        if 'error' not in performance:
            print(f"\n{model_type.upper()}:")
            print(f"  R² Score: {performance['r2']:.4f}")
            print(f"  MAE: {performance['mae']:.4f}")
            print(f"  MSE: {performance['mse']:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict_price(test_data, 'TEST')
    
    print("\nPrediction Results:")
    for key, value in predictions.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save models
    model_file = predictor.save_models('TEST')
    print(f"\nModels saved to: {model_file}")
