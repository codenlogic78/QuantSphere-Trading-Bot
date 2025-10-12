#!/usr/bin/env python3
"""
Enhanced Trading Engine for QuantSphere AI Trading Platform
Integrates Hybrid AI Analyst with existing trading system
Based on "From Man vs. Machine to Man + Machine" research
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from hybrid_ai_analyst import HybridAIAnalyst, AnalysisType, MarketCondition
from ml_predictor import MLPredictor
from technical_indicators import TechnicalIndicators

class TradingSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class SignalSource(Enum):
    TECHNICAL = "technical"
    AI_ONLY = "ai_only"
    HUMAN_ENHANCED = "human_enhanced"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

@dataclass
class TradingDecision:
    symbol: str
    signal: TradingSignal
    confidence: float
    source: SignalSource
    reasoning: str
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    risk_score: Optional[float] = None

class EnhancedTradingEngine:
    """
    Enhanced trading engine that combines multiple analysis methods
    following the research findings from Cao et al. (2024)
    """
    
    def __init__(self, openai_api_key: str = None):
        # Initialize components
        self.hybrid_analyst = HybridAIAnalyst(openai_api_key)
        self.ml_predictor = MLPredictor()
        self.technical_indicators = TechnicalIndicators()
        
        # Trading parameters
        self.confidence_threshold = 0.6  # Minimum confidence for trading
        self.position_size_base = 0.1  # 10% of portfolio per position
        self.max_positions = 5  # Maximum concurrent positions
        
        # Performance tracking
        self.trading_history = []
        self.signal_performance = {
            'technical': {'correct': 0, 'total': 0},
            'ai_only': {'correct': 0, 'total': 0},
            'human_enhanced': {'correct': 0, 'total': 0},
            'hybrid': {'correct': 0, 'total': 0},
            'ensemble': {'correct': 0, 'total': 0}
        }
        
        # Risk management
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk per trade
        self.correlation_threshold = 0.7  # Max correlation between positions
        
    def generate_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive analysis using all available methods"""
        
        print(f"🔍 Generating comprehensive analysis for {symbol}...")
        
        analysis_results = {}
        
        try:
            # 1. Technical Analysis
            print("  📊 Running technical analysis...")
            technical_signals = self.technical_indicators.generate_signals(symbol)
            analysis_results['technical'] = {
                'signal': technical_signals.get('signal', 'HOLD'),
                'confidence': technical_signals.get('strength', 0.5) / 100,  # Convert to 0-1
                'indicators': technical_signals.get('indicators', {}),
                'reasoning': f"Technical analysis based on RSI, MACD, Bollinger Bands"
            }
            
        except Exception as e:
            print(f"  ⚠️ Technical analysis error: {e}")
            analysis_results['technical'] = {
                'signal': 'HOLD',
                'confidence': 0.5,
                'indicators': {},
                'reasoning': f"Technical analysis error: {str(e)}"
            }
        
        try:
            # 2. AI-Only Analysis
            print("  🤖 Running AI-only analysis...")
            ai_result = self.hybrid_analyst.analyze_stock(symbol, AnalysisType.AI_ONLY)
            analysis_results['ai_only'] = ai_result
            
        except Exception as e:
            print(f"  ⚠️ AI-only analysis error: {e}")
            analysis_results['ai_only'] = {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'reasoning': f"AI analysis error: {str(e)}"
            }
        
        try:
            # 3. Human-Enhanced Analysis
            print("  🧠 Running human-enhanced analysis...")
            human_result = self.hybrid_analyst.analyze_stock(symbol, AnalysisType.HUMAN_ENHANCED)
            analysis_results['human_enhanced'] = human_result
            
        except Exception as e:
            print(f"  ⚠️ Human-enhanced analysis error: {e}")
            analysis_results['human_enhanced'] = {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'reasoning': f"Human analysis error: {str(e)}"
            }
        
        try:
            # 4. Hybrid Analysis
            print("  🔄 Running hybrid analysis...")
            hybrid_result = self.hybrid_analyst.analyze_stock(symbol, AnalysisType.HYBRID)
            analysis_results['hybrid'] = hybrid_result
            
        except Exception as e:
            print(f"  ⚠️ Hybrid analysis error: {e}")
            analysis_results['hybrid'] = {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'reasoning': f"Hybrid analysis error: {str(e)}"
            }
        
        # 5. Ensemble Analysis (combine all methods)
        print("  🎯 Creating ensemble analysis...")
        ensemble_result = self.create_ensemble_analysis(analysis_results)
        analysis_results['ensemble'] = ensemble_result
        
        return analysis_results
    
    def create_ensemble_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble analysis combining all methods with adaptive weighting"""
        
        # Adaptive weights based on historical performance
        weights = self.calculate_adaptive_weights()
        
        # Collect predictions and confidences
        predictions = {}
        confidences = {}
        
        for method, result in analysis_results.items():
            if method == 'ensemble':  # Skip to avoid recursion
                continue
                
            if method == 'technical':
                pred = result.get('signal', 'HOLD')
                conf = result.get('confidence', 0.5)
            else:
                pred = result.get('prediction', 'HOLD')
                conf = result.get('confidence', 0.5)
            
            predictions[method] = pred
            confidences[method] = conf
        
        # Calculate weighted scores
        signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_weight = 0
        
        for method, pred in predictions.items():
            weight = weights.get(method, 0.25)
            confidence = confidences[method]
            
            # Convert predictions to consistent format
            if pred in ['STRONG_BUY', 'BUY']:
                signal_scores['BUY'] += weight * confidence
            elif pred in ['STRONG_SELL', 'SELL']:
                signal_scores['SELL'] += weight * confidence
            else:
                signal_scores['HOLD'] += weight * confidence
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for signal in signal_scores:
                signal_scores[signal] /= total_weight
        
        # Determine final signal
        final_signal = max(signal_scores, key=signal_scores.get)
        final_confidence = signal_scores[final_signal]
        
        # Enhance signal strength based on consensus
        consensus_count = sum(1 for pred in predictions.values() 
                            if pred in ['BUY', 'STRONG_BUY'] and final_signal == 'BUY' or
                               pred in ['SELL', 'STRONG_SELL'] and final_signal == 'SELL' or
                               pred == 'HOLD' and final_signal == 'HOLD')
        
        consensus_ratio = consensus_count / len(predictions)
        
        # Adjust signal strength based on consensus
        if consensus_ratio >= 0.75 and final_confidence > 0.7:
            if final_signal == 'BUY':
                final_signal = 'STRONG_BUY'
            elif final_signal == 'SELL':
                final_signal = 'STRONG_SELL'
        
        # Create reasoning summary
        reasoning_parts = []
        for method, result in analysis_results.items():
            if method == 'ensemble':
                continue
            pred = predictions.get(method, 'HOLD')
            conf = confidences.get(method, 0.5)
            reasoning_parts.append(f"{method}: {pred} ({conf:.1%})")
        
        reasoning = f"Ensemble analysis - {'; '.join(reasoning_parts)}. Consensus: {consensus_ratio:.1%}"
        
        return {
            'prediction': final_signal,
            'confidence': final_confidence,
            'consensus_ratio': consensus_ratio,
            'signal_scores': signal_scores,
            'method_weights': weights,
            'reasoning': reasoning,
            'method': 'ensemble'
        }
    
    def calculate_adaptive_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights based on historical performance"""
        
        base_weights = {
            'technical': 0.25,
            'ai_only': 0.20,
            'human_enhanced': 0.25,
            'hybrid': 0.30  # Highest weight based on research findings
        }
        
        # Adjust weights based on performance
        for method, performance in self.signal_performance.items():
            if performance['total'] > 10:  # Minimum sample size
                accuracy = performance['correct'] / performance['total']
                
                # Boost weight for high-performing methods
                if accuracy > 0.6:
                    base_weights[method] = base_weights.get(method, 0.25) * (1 + (accuracy - 0.6))
                # Reduce weight for poor-performing methods
                elif accuracy < 0.4:
                    base_weights[method] = base_weights.get(method, 0.25) * accuracy / 0.4
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            for method in base_weights:
                base_weights[method] /= total_weight
        
        return base_weights
    
    def generate_trading_decision(self, symbol: str, current_price: float, 
                                portfolio_value: float = 100000) -> Optional[TradingDecision]:
        """Generate trading decision based on comprehensive analysis"""
        
        print(f"\n🎯 Generating trading decision for {symbol} at ${current_price:.2f}")
        
        # Get comprehensive analysis
        analysis = self.generate_comprehensive_analysis(symbol)
        
        # Use ensemble result as primary decision
        ensemble_result = analysis.get('ensemble', {})
        signal = ensemble_result.get('prediction', 'HOLD')
        confidence = ensemble_result.get('confidence', 0.5)
        reasoning = ensemble_result.get('reasoning', 'No analysis available')
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            print(f"  ⚠️ Confidence {confidence:.1%} below threshold {self.confidence_threshold:.1%}")
            return None
        
        # Convert to TradingSignal enum
        try:
            trading_signal = TradingSignal(signal)
        except ValueError:
            trading_signal = TradingSignal.HOLD
        
        # Calculate position size based on confidence and risk
        position_size = self.calculate_position_size(confidence, portfolio_value)
        
        # Calculate price targets and stop loss
        price_target, stop_loss = self.calculate_price_targets(
            current_price, trading_signal, confidence, analysis
        )
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(symbol, trading_signal, analysis)
        
        decision = TradingDecision(
            symbol=symbol,
            signal=trading_signal,
            confidence=confidence,
            source=SignalSource.ENSEMBLE,
            reasoning=reasoning,
            timestamp=datetime.now(),
            price_target=price_target,
            stop_loss=stop_loss,
            position_size=position_size,
            risk_score=risk_score
        )
        
        print(f"  ✅ Decision: {decision.signal.value} with {decision.confidence:.1%} confidence")
        print(f"  📊 Position size: {decision.position_size:.1%} of portfolio")
        print(f"  🎯 Price target: ${decision.price_target:.2f}" if decision.price_target else "")
        print(f"  🛡️ Stop loss: ${decision.stop_loss:.2f}" if decision.stop_loss else "")
        
        return decision
    
    def calculate_position_size(self, confidence: float, portfolio_value: float) -> float:
        """Calculate position size based on confidence and risk management"""
        
        # Base position size adjusted by confidence
        base_size = self.position_size_base
        confidence_multiplier = min(2.0, confidence / 0.5)  # Scale with confidence
        
        position_size = base_size * confidence_multiplier
        
        # Apply risk management constraints
        max_position = portfolio_value * self.max_portfolio_risk / 100  # Convert to dollar amount
        position_size = min(position_size, max_position / portfolio_value)
        
        return max(0.01, min(0.25, position_size))  # Between 1% and 25%
    
    def calculate_price_targets(self, current_price: float, signal: TradingSignal, 
                              confidence: float, analysis: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop loss levels"""
        
        if signal in [TradingSignal.HOLD]:
            return None, None
        
        # Base target percentages
        if signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY]:
            target_pct = 0.05 + (confidence - 0.5) * 0.1  # 5-10% target
            stop_pct = 0.03 + (1 - confidence) * 0.02  # 3-5% stop loss
            
            price_target = current_price * (1 + target_pct)
            stop_loss = current_price * (1 - stop_pct)
            
        else:  # SELL signals
            target_pct = 0.05 + (confidence - 0.5) * 0.1  # 5-10% target
            stop_pct = 0.03 + (1 - confidence) * 0.02  # 3-5% stop loss
            
            price_target = current_price * (1 - target_pct)
            stop_loss = current_price * (1 + stop_pct)
        
        return round(price_target, 2), round(stop_loss, 2)
    
    def calculate_risk_score(self, symbol: str, signal: TradingSignal, 
                           analysis: Dict[str, Any]) -> float:
        """Calculate risk score for the trading decision"""
        
        risk_factors = []
        
        # Volatility risk (from technical analysis)
        technical_data = analysis.get('technical', {})
        indicators = technical_data.get('indicators', {})
        
        # Market condition risk
        try:
            market_condition = self.hybrid_analyst.assess_market_condition(symbol)
            if market_condition in [MarketCondition.VOLATILE, MarketCondition.DISTRESSED]:
                risk_factors.append(0.3)
            elif market_condition == MarketCondition.BEAR_MARKET:
                risk_factors.append(0.2)
        except:
            risk_factors.append(0.1)  # Default market risk
        
        # Signal consensus risk
        ensemble_data = analysis.get('ensemble', {})
        consensus_ratio = ensemble_data.get('consensus_ratio', 0.5)
        consensus_risk = (1 - consensus_ratio) * 0.2
        risk_factors.append(consensus_risk)
        
        # Confidence risk
        confidence = ensemble_data.get('confidence', 0.5)
        confidence_risk = (1 - confidence) * 0.3
        risk_factors.append(confidence_risk)
        
        # Company-specific risk
        try:
            company_profile = self.hybrid_analyst.get_company_profile(symbol)
            if company_profile.liquidity_score < 0.3:  # Low liquidity
                risk_factors.append(0.2)
            if company_profile.market_cap < 1_000_000_000:  # Small cap
                risk_factors.append(0.15)
        except:
            risk_factors.append(0.1)  # Default company risk
        
        # Calculate overall risk score (0-1, higher = riskier)
        base_risk = 0.1  # Minimum risk
        additional_risk = sum(risk_factors)
        total_risk = min(1.0, base_risk + additional_risk)
        
        return round(total_risk, 3)
    
    def execute_trading_decision(self, decision: TradingDecision, 
                               current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading decision with risk management"""
        
        execution_result = {
            'executed': False,
            'reason': '',
            'action': None,
            'quantity': 0,
            'price': 0
        }
        
        # Check if we already have a position
        current_position = current_positions.get(decision.symbol, {})
        has_position = current_position.get('position', 0) > 0
        
        # Risk management checks
        if decision.risk_score > 0.7:
            execution_result['reason'] = f"Risk score too high: {decision.risk_score:.1%}"
            return execution_result
        
        if len(current_positions) >= self.max_positions and not has_position:
            execution_result['reason'] = f"Maximum positions ({self.max_positions}) reached"
            return execution_result
        
        # Execute based on signal
        if decision.signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY]:
            if not has_position:
                # Calculate quantity based on position size
                portfolio_value = 100000  # Default portfolio value
                position_value = portfolio_value * decision.position_size
                quantity = int(position_value / current_positions.get(decision.symbol, {}).get('current_price', 100))
                
                execution_result.update({
                    'executed': True,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_positions.get(decision.symbol, {}).get('current_price', 0),
                    'reason': f"BUY signal with {decision.confidence:.1%} confidence"
                })
            else:
                execution_result['reason'] = "Already have position"
        
        elif decision.signal in [TradingSignal.STRONG_SELL, TradingSignal.SELL]:
            if has_position:
                quantity = current_position.get('position', 0)
                execution_result.update({
                    'executed': True,
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': current_positions.get(decision.symbol, {}).get('current_price', 0),
                    'reason': f"SELL signal with {decision.confidence:.1%} confidence"
                })
            else:
                execution_result['reason'] = "No position to sell"
        
        else:  # HOLD
            execution_result['reason'] = "HOLD signal - no action taken"
        
        # Log the decision
        self.trading_history.append({
            'timestamp': decision.timestamp.isoformat(),
            'symbol': decision.symbol,
            'decision': decision.signal.value,
            'confidence': decision.confidence,
            'execution': execution_result,
            'reasoning': decision.reasoning
        })
        
        return execution_result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'total_decisions': len(self.trading_history),
            'execution_rate': 0,
            'signal_performance': self.signal_performance.copy(),
            'recent_decisions': self.trading_history[-10:] if self.trading_history else [],
            'method_weights': self.calculate_adaptive_weights()
        }
        
        # Calculate execution rate
        if self.trading_history:
            executed_count = sum(1 for decision in self.trading_history 
                               if decision.get('execution', {}).get('executed', False))
            report['execution_rate'] = executed_count / len(self.trading_history)
        
        # Add hybrid analyst performance
        analyst_summary = self.hybrid_analyst.get_performance_summary()
        report['analyst_performance'] = analyst_summary
        
        return report
    
    def update_performance(self, symbol: str, method: str, was_correct: bool):
        """Update performance tracking for a specific method"""
        if method in self.signal_performance:
            self.signal_performance[method]['total'] += 1
            if was_correct:
                self.signal_performance[method]['correct'] += 1
    
    def save_state(self, filepath: str = None):
        """Save engine state to file"""
        if filepath is None:
            filepath = f"enhanced_trading_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        state = {
            'trading_history': self.trading_history,
            'signal_performance': self.signal_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        return filepath
    
    def load_state(self, filepath: str):
        """Load engine state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.trading_history = state.get('trading_history', [])
        self.signal_performance = state.get('signal_performance', self.signal_performance)

# Test the Enhanced Trading Engine
if __name__ == "__main__":
    print("=== QUANTSPHERE ENHANCED TRADING ENGINE TEST ===")
    
    # Initialize engine
    engine = EnhancedTradingEngine()
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'MSFT']
    mock_portfolio = {}
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"TESTING ENHANCED TRADING ENGINE FOR {symbol}")
        print('='*60)
        
        # Mock current price
        current_price = 150.0 + np.random.uniform(-20, 20)
        
        try:
            # Generate trading decision
            decision = engine.generate_trading_decision(symbol, current_price)
            
            if decision:
                print(f"\n📋 TRADING DECISION:")
                print(f"   Symbol: {decision.symbol}")
                print(f"   Signal: {decision.signal.value}")
                print(f"   Confidence: {decision.confidence:.1%}")
                print(f"   Position Size: {decision.position_size:.1%}")
                print(f"   Price Target: ${decision.price_target:.2f}" if decision.price_target else "   Price Target: None")
                print(f"   Stop Loss: ${decision.stop_loss:.2f}" if decision.stop_loss else "   Stop Loss: None")
                print(f"   Risk Score: {decision.risk_score:.1%}")
                print(f"   Reasoning: {decision.reasoning}")
                
                # Mock execution
                mock_positions = {symbol: {'position': 0, 'current_price': current_price}}
                execution = engine.execute_trading_decision(decision, mock_positions)
                
                print(f"\n⚡ EXECUTION RESULT:")
                print(f"   Executed: {execution['executed']}")
                print(f"   Action: {execution.get('action', 'None')}")
                print(f"   Quantity: {execution.get('quantity', 0)}")
                print(f"   Reason: {execution['reason']}")
                
            else:
                print("❌ No trading decision generated (low confidence or other constraints)")
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    # Performance report
    print(f"\n{'='*60}")
    print("PERFORMANCE REPORT")
    print('='*60)
    
    report = engine.get_performance_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Save state
    state_file = engine.save_state()
    print(f"\n💾 Engine state saved to: {state_file}")
