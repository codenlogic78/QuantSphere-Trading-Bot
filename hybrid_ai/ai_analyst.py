"""
AI Analysis Engine for Hybrid Trading System
Based on Cao et al. (2024) - Machine component of "Man + Machine" approach

This module provides the AI/Machine learning analysis capabilities:
- Technical pattern recognition
- Quantitative financial analysis
- ML-based price prediction
- Risk assessment and position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIAnalysisResult:
    """Result of AI analysis for a stock"""
    symbol: str
    timestamp: datetime
    
    # Technical Analysis
    technical_signal: str  # 'BUY', 'SELL', 'HOLD'
    technical_confidence: float  # 0-1
    technical_reasoning: List[str]
    
    # Price Prediction
    price_prediction: float
    price_confidence: float
    prediction_horizon: str  # '1d', '5d', '1m'
    
    # Risk Assessment
    volatility_forecast: float
    risk_score: float  # 0-1, higher = riskier
    max_position_size: float  # 0-1, fraction of portfolio
    
    # Overall AI Recommendation
    ai_recommendation: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    ai_confidence: float  # 0-1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'technical_signal': self.technical_signal,
            'technical_confidence': self.technical_confidence,
            'technical_reasoning': self.technical_reasoning,
            'price_prediction': self.price_prediction,
            'price_confidence': self.price_confidence,
            'prediction_horizon': self.prediction_horizon,
            'volatility_forecast': self.volatility_forecast,
            'risk_score': self.risk_score,
            'max_position_size': self.max_position_size,
            'ai_recommendation': self.ai_recommendation,
            'ai_confidence': self.ai_confidence
        }

class TechnicalAnalyzer:
    """Advanced technical analysis using AI/ML approaches"""
    
    def __init__(self):
        self.pattern_classifier = None
        self.scaler = StandardScaler()
        
    def analyze_technical_patterns(self, market_data: pd.DataFrame) -> Tuple[str, float, List[str]]:
        """Analyze technical patterns and generate signals"""
        try:
            if market_data.empty or len(market_data) < 50:
                return 'HOLD', 0.5, ['Insufficient data for technical analysis']
            
            signals = []
            reasoning = []
            
            # Get latest data
            latest = market_data.iloc[-1]
            recent = market_data.tail(20)
            
            # 1. Moving Average Analysis
            ma_signal, ma_reason = self._analyze_moving_averages(market_data)
            signals.append(ma_signal)
            reasoning.extend(ma_reason)
            
            # 2. RSI Analysis
            rsi_signal, rsi_reason = self._analyze_rsi(market_data)
            signals.append(rsi_signal)
            reasoning.extend(rsi_reason)
            
            # 3. Momentum Analysis
            momentum_signal, momentum_reason = self._analyze_momentum(market_data)
            signals.append(momentum_signal)
            reasoning.extend(momentum_reason)
            
            # 4. Volume Analysis
            volume_signal, volume_reason = self._analyze_volume(market_data)
            signals.append(volume_signal)
            reasoning.extend(volume_reason)
            
            # Combine signals
            signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            for signal in signals:
                signal_scores[signal] += 1
            
            # Determine overall signal
            max_signal = max(signal_scores, key=signal_scores.get)
            confidence = signal_scores[max_signal] / len(signals)
            
            return max_signal, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return 'HOLD', 0.5, ['Error in technical analysis']
    
    def _analyze_moving_averages(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """Analyze moving average crossovers and trends"""
        try:
            latest = data.iloc[-1]
            
            if 'SMA_20' not in data.columns or 'SMA_50' not in data.columns:
                return 'HOLD', ['Missing moving average data']
            
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            price = latest['Close']
            
            reasons = []
            
            # Price vs MA analysis
            if price > sma_20 > sma_50:
                reasons.append(f"Price (${price:.2f}) above both MA20 (${sma_20:.2f}) and MA50 (${sma_50:.2f})")
                return 'BUY', reasons
            elif price < sma_20 < sma_50:
                reasons.append(f"Price (${price:.2f}) below both MA20 (${sma_20:.2f}) and MA50 (${sma_50:.2f})")
                return 'SELL', reasons
            else:
                reasons.append(f"Mixed MA signals - Price: ${price:.2f}, MA20: ${sma_20:.2f}, MA50: ${sma_50:.2f}")
                return 'HOLD', reasons
                
        except Exception as e:
            return 'HOLD', [f'MA analysis error: {str(e)}']
    
    def _analyze_rsi(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """Analyze RSI for overbought/oversold conditions"""
        try:
            if 'RSI' not in data.columns:
                return 'HOLD', ['Missing RSI data']
            
            latest_rsi = data['RSI'].iloc[-1]
            reasons = []
            
            if latest_rsi < 30:
                reasons.append(f"RSI ({latest_rsi:.1f}) indicates oversold condition")
                return 'BUY', reasons
            elif latest_rsi > 70:
                reasons.append(f"RSI ({latest_rsi:.1f}) indicates overbought condition")
                return 'SELL', reasons
            else:
                reasons.append(f"RSI ({latest_rsi:.1f}) in neutral range")
                return 'HOLD', reasons
                
        except Exception as e:
            return 'HOLD', [f'RSI analysis error: {str(e)}']
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """Analyze price momentum"""
        try:
            if 'Momentum' not in data.columns:
                return 'HOLD', ['Missing momentum data']
            
            recent_momentum = data['Momentum'].tail(5).mean()
            reasons = []
            
            if recent_momentum > 0.02:  # 2% positive momentum
                reasons.append(f"Strong positive momentum ({recent_momentum:.1%})")
                return 'BUY', reasons
            elif recent_momentum < -0.02:  # 2% negative momentum
                reasons.append(f"Strong negative momentum ({recent_momentum:.1%})")
                return 'SELL', reasons
            else:
                reasons.append(f"Weak momentum ({recent_momentum:.1%})")
                return 'HOLD', reasons
                
        except Exception as e:
            return 'HOLD', [f'Momentum analysis error: {str(e)}']
    
    def _analyze_volume(self, data: pd.DataFrame) -> Tuple[str, List[str]]:
        """Analyze volume patterns"""
        try:
            if 'Volume' not in data.columns:
                return 'HOLD', ['Missing volume data']
            
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].mean()
            price_change = data['Close'].pct_change().iloc[-1]
            
            reasons = []
            
            if recent_volume > avg_volume * 1.5 and price_change > 0:
                reasons.append(f"High volume ({recent_volume/1e6:.1f}M) with price increase")
                return 'BUY', reasons
            elif recent_volume > avg_volume * 1.5 and price_change < 0:
                reasons.append(f"High volume ({recent_volume/1e6:.1f}M) with price decrease")
                return 'SELL', reasons
            else:
                reasons.append(f"Normal volume pattern")
                return 'HOLD', reasons
                
        except Exception as e:
            return 'HOLD', [f'Volume analysis error: {str(e)}']

class MLPredictor:
    """Machine learning-based price prediction"""
    
    def __init__(self):
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.direction_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['returns_1d'] = market_data['Close'].pct_change()
            features['returns_5d'] = market_data['Close'].pct_change(periods=5)
            features['returns_20d'] = market_data['Close'].pct_change(periods=20)
            
            # Technical indicators
            if 'RSI' in market_data.columns:
                features['rsi'] = market_data['RSI']
            if 'SMA_20' in market_data.columns:
                features['price_to_sma20'] = market_data['Close'] / market_data['SMA_20']
            if 'SMA_50' in market_data.columns:
                features['price_to_sma50'] = market_data['Close'] / market_data['SMA_50']
            if 'Volatility' in market_data.columns:
                features['volatility'] = market_data['Volatility']
            
            # Volume features
            if 'Volume' in market_data.columns:
                features['volume_ratio'] = market_data['Volume'] / market_data['Volume'].rolling(20).mean()
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f'return_lag_{lag}'] = features['returns_1d'].shift(lag)
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def train_models(self, market_data: pd.DataFrame) -> bool:
        """Train ML models on historical data"""
        try:
            features = self.prepare_features(market_data)
            
            if features.empty or len(features) < 100:
                logger.warning("Insufficient data for ML training")
                return False
            
            # Prepare targets
            returns = market_data['Close'].pct_change().shift(-1)  # Next day return
            prices = market_data['Close'].shift(-1)  # Next day price
            
            # Align features and targets
            min_len = min(len(features), len(returns))
            features = features.iloc[:min_len]
            returns = returns.iloc[:min_len]
            prices = prices.iloc[:min_len]
            
            # Remove NaN values
            valid_idx = ~(returns.isna() | prices.isna())
            features = features[valid_idx]
            returns = returns[valid_idx]
            prices = prices[valid_idx]
            
            if len(features) < 50:
                logger.warning("Insufficient valid data for ML training")
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train price prediction model
            self.price_model.fit(features_scaled, prices)
            
            # Train direction prediction model
            directions = (returns > 0).astype(int)  # 1 for up, 0 for down
            self.direction_model.fit(features_scaled, directions)
            
            self.is_trained = True
            logger.info("ML models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False
    
    def predict_price(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Predict next price and confidence"""
        try:
            if not self.is_trained:
                if not self.train_models(market_data):
                    return market_data['Close'].iloc[-1], 0.5
            
            features = self.prepare_features(market_data)
            if features.empty:
                return market_data['Close'].iloc[-1], 0.5
            
            # Get latest features
            latest_features = features.iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Predict price
            predicted_price = self.price_model.predict(latest_features_scaled)[0]
            
            # Calculate confidence based on model performance
            # For simplicity, using feature importance as proxy
            feature_importance = np.mean(self.price_model.feature_importances_)
            confidence = min(0.9, max(0.1, feature_importance * 2))
            
            return predicted_price, confidence
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return market_data['Close'].iloc[-1], 0.5

class AIAnalyst:
    """Main AI analyst combining all AI/ML capabilities"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_predictor = MLPredictor()
        
    def analyze_stock(self, symbol: str, market_data: pd.DataFrame, 
                     fundamentals: Dict, sentiment: Dict) -> AIAnalysisResult:
        """Perform comprehensive AI analysis of a stock"""
        try:
            logger.info(f"Starting AI analysis for {symbol}")
            
            # Technical Analysis
            tech_signal, tech_confidence, tech_reasoning = self.technical_analyzer.analyze_technical_patterns(market_data)
            
            # Price Prediction
            predicted_price, price_confidence = self.ml_predictor.predict_price(market_data)
            current_price = market_data['Close'].iloc[-1]
            
            # Risk Assessment
            volatility = market_data['Close'].pct_change().std() * np.sqrt(252)  # Annualized
            risk_score = min(1.0, volatility / 0.5)  # Normalize to 0-1
            
            # Position sizing based on risk
            max_position = max(0.05, min(0.25, 1 - risk_score))
            
            # Overall AI recommendation
            ai_recommendation, ai_confidence = self._generate_overall_recommendation(
                tech_signal, tech_confidence, predicted_price, current_price, 
                price_confidence, risk_score
            )
            
            result = AIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                technical_signal=tech_signal,
                technical_confidence=tech_confidence,
                technical_reasoning=tech_reasoning,
                price_prediction=predicted_price,
                price_confidence=price_confidence,
                prediction_horizon='1d',
                volatility_forecast=volatility,
                risk_score=risk_score,
                max_position_size=max_position,
                ai_recommendation=ai_recommendation,
                ai_confidence=ai_confidence
            )
            
            logger.info(f"AI analysis completed for {symbol}: {ai_recommendation} (confidence: {ai_confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            # Return default result
            return AIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                technical_signal='HOLD',
                technical_confidence=0.5,
                technical_reasoning=['Error in analysis'],
                price_prediction=market_data['Close'].iloc[-1] if not market_data.empty else 100.0,
                price_confidence=0.5,
                prediction_horizon='1d',
                volatility_forecast=0.2,
                risk_score=0.5,
                max_position_size=0.1,
                ai_recommendation='HOLD',
                ai_confidence=0.5
            )
    
    def _generate_overall_recommendation(self, tech_signal: str, tech_confidence: float,
                                       predicted_price: float, current_price: float,
                                       price_confidence: float, risk_score: float) -> Tuple[str, float]:
        """Generate overall AI recommendation combining all factors"""
        try:
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price
            
            # Weight different signals
            tech_weight = 0.4
            price_weight = 0.4
            risk_weight = 0.2
            
            # Convert technical signal to score
            tech_score = {'BUY': 1, 'HOLD': 0, 'SELL': -1}[tech_signal]
            
            # Convert price prediction to score
            if expected_return > 0.05:  # 5%+ expected return
                price_score = 1
            elif expected_return < -0.05:  # 5%+ expected loss
                price_score = -1
            else:
                price_score = 0
            
            # Risk adjustment
            risk_adjustment = 1 - risk_score  # Lower risk = higher score
            
            # Combined score
            combined_score = (tech_score * tech_weight * tech_confidence + 
                            price_score * price_weight * price_confidence) * risk_adjustment
            
            # Convert to recommendation
            if combined_score > 0.6:
                recommendation = 'STRONG_BUY'
            elif combined_score > 0.2:
                recommendation = 'BUY'
            elif combined_score > -0.2:
                recommendation = 'HOLD'
            elif combined_score > -0.6:
                recommendation = 'SELL'
            else:
                recommendation = 'STRONG_SELL'
            
            # Calculate overall confidence
            confidence = (tech_confidence + price_confidence) / 2 * risk_adjustment
            confidence = max(0.1, min(0.9, confidence))
            
            return recommendation, confidence
            
        except Exception as e:
            logger.error(f"Error generating overall recommendation: {e}")
            return 'HOLD', 0.5

# Example usage and testing
if __name__ == "__main__":
    # Test the AI analyst
    from data_collector import MultiSourceDataCollector
    
    config = {}
    collector = MultiSourceDataCollector(config)
    ai_analyst = AIAnalyst()
    
    # Test with a symbol
    symbol = 'AAPL'
    print(f"\n{'='*60}")
    print(f"Testing AI Analysis for {symbol}")
    print(f"{'='*60}")
    
    # Collect data
    data = collector.collect_comprehensive_data(symbol)
    market_data = pd.DataFrame(data['market_data'])
    
    if not market_data.empty:
        # Convert index to datetime if needed
        if 'Date' in market_data.columns:
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            market_data.set_index('Date', inplace=True)
        
        # Perform AI analysis
        ai_result = ai_analyst.analyze_stock(
            symbol, 
            market_data, 
            data['fundamentals'], 
            data['sentiment']
        )
        
        # Display results
        print(f"\n🤖 AI ANALYSIS RESULTS:")
        print(f"Technical Signal: {ai_result.technical_signal} (confidence: {ai_result.technical_confidence:.2f})")
        print(f"Price Prediction: ${ai_result.price_prediction:.2f} (confidence: {ai_result.price_confidence:.2f})")
        print(f"Risk Score: {ai_result.risk_score:.2f}")
        print(f"Max Position Size: {ai_result.max_position_size:.1%}")
        print(f"Overall Recommendation: {ai_result.ai_recommendation} (confidence: {ai_result.ai_confidence:.2f})")
        
        print(f"\n📊 Technical Reasoning:")
        for reason in ai_result.technical_reasoning:
            print(f"  • {reason}")
    else:
        print("❌ No market data available for analysis")
