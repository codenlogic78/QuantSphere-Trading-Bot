"""
Hybrid AI Integration with QuantSphere Trading Platform
Based on Cao et al. (2024) - "Man + Machine" approach

This module integrates the hybrid AI system with the existing QuantSphere platform,
providing enhanced analysis capabilities to the main trading bot.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Add hybrid_ai to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hybrid_ai'))

from hybrid_ai import HybridAITradingSystem, HybridAnalysisReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantSphereHybridAI:
    """Integration class for Hybrid AI with QuantSphere platform"""
    
    def __init__(self, config: Dict = None):
        """Initialize the hybrid AI integration"""
        self.config = config or {}
        
        # Initialize hybrid AI system
        self.hybrid_system = HybridAITradingSystem(self.config)
        
        # Integration settings
        self.cache_duration = self.config.get('cache_duration', 3600)  # 1 hour
        self.analysis_cache = {}
        
        logger.info("QuantSphere Hybrid AI Integration initialized")
    
    def get_enhanced_analysis(self, symbol: str) -> Dict:
        """Get enhanced analysis for integration with QuantSphere GUI"""
        try:
            logger.info(f"Getting enhanced analysis for {symbol}")
            
            # Get hybrid analysis
            report = self.hybrid_system.analyze_stock(symbol)
            
            # Convert to QuantSphere-compatible format
            enhanced_analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'hybrid_recommendation': report.hybrid_recommendation.hybrid_recommendation,
                'confidence': report.confidence_score,
                'expected_alpha': report.expected_alpha,
                'risk_score': report.risk_score,
                
                # AI Component
                'ai_analysis': {
                    'recommendation': report.ai_analysis.ai_recommendation,
                    'confidence': report.ai_analysis.ai_confidence,
                    'technical_signal': report.ai_analysis.technical_signal,
                    'price_prediction': report.ai_analysis.price_prediction,
                    'risk_score': report.ai_analysis.risk_score,
                    'reasoning': report.ai_analysis.technical_reasoning[:3]
                },
                
                # Human-like AI Component
                'human_analysis': {
                    'recommendation': report.human_analysis.human_recommendation,
                    'confidence': report.human_analysis.human_confidence,
                    'company_assessment': report.human_analysis.company_assessment,
                    'market_sentiment': report.human_analysis.market_sentiment_interpretation,
                    'opportunities': report.human_analysis.unique_opportunities[:2],
                    'risks': report.human_analysis.hidden_risks[:2],
                    'catalysts': report.human_analysis.key_catalysts[:3]
                },
                
                # Weighting Information
                'weighting': {
                    'ai_weight': report.hybrid_recommendation.ai_weight,
                    'human_weight': report.hybrid_recommendation.human_weight,
                    'rationale': report.hybrid_recommendation.weighting_rationale,
                    'decision_narrative': report.hybrid_recommendation.decision_narrative
                },
                
                # Performance Metrics
                'performance': {
                    'processing_time': report.processing_time_seconds,
                    'data_completeness': report.data_completeness,
                    'data_freshness': report.data_freshness,
                    'risk_adjusted_score': report.hybrid_recommendation.risk_adjusted_score
                }
            }
            
            logger.info(f"Enhanced analysis completed for {symbol}")
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Error getting enhanced analysis for {symbol}: {e}")
            return self._create_fallback_analysis(symbol, str(e))
    
    def get_portfolio_recommendations(self, symbols: List[str], top_n: int = 5) -> List[Dict]:
        """Get top portfolio recommendations for QuantSphere"""
        try:
            logger.info(f"Getting portfolio recommendations for {len(symbols)} symbols")
            
            # Get top recommendations from hybrid system
            top_recs = self.hybrid_system.get_top_recommendations(symbols, top_n)
            
            # Convert to QuantSphere format
            recommendations = []
            
            for i, (symbol, report) in enumerate(top_recs, 1):
                rec = {
                    'rank': i,
                    'symbol': symbol,
                    'recommendation': report.hybrid_recommendation.hybrid_recommendation,
                    'confidence': report.confidence_score,
                    'expected_alpha': report.expected_alpha,
                    'risk_score': report.risk_score,
                    'ai_weight': report.hybrid_recommendation.ai_weight,
                    'risk_adjusted_score': report.hybrid_recommendation.risk_adjusted_score,
                    'reasoning': report.hybrid_recommendation.decision_narrative[:100] + "...",
                    'key_factors': {
                        'ai_signal': report.ai_analysis.technical_signal,
                        'human_assessment': report.human_analysis.company_assessment[:50] + "...",
                        'main_catalyst': report.human_analysis.key_catalysts[0] if report.human_analysis.key_catalysts else "Market conditions"
                    }
                }
                recommendations.append(rec)
            
            logger.info(f"Generated {len(recommendations)} portfolio recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting portfolio recommendations: {e}")
            return []
    
    def get_trading_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get trading signals for QuantSphere trading engine"""
        try:
            logger.info(f"Getting trading signals for {len(symbols)} symbols")
            
            signals = {}
            
            # Analyze portfolio
            portfolio_results = self.hybrid_system.analyze_portfolio(symbols, max_workers=2)
            
            for symbol, report in portfolio_results.items():
                # Convert hybrid recommendation to trading signal
                signal_strength = self._recommendation_to_signal_strength(
                    report.hybrid_recommendation.hybrid_recommendation,
                    report.confidence_score
                )
                
                signals[symbol] = {
                    'signal': report.hybrid_recommendation.hybrid_recommendation,
                    'strength': signal_strength,
                    'confidence': report.confidence_score,
                    'position_size': min(0.2, report.ai_analysis.max_position_size),  # Max 20%
                    'expected_return': report.expected_alpha,
                    'risk_level': 'LOW' if report.risk_score < 0.3 else 'MEDIUM' if report.risk_score < 0.7 else 'HIGH',
                    'ai_component': {
                        'technical_signal': report.ai_analysis.technical_signal,
                        'price_target': report.ai_analysis.price_prediction
                    },
                    'human_component': {
                        'qualitative_assessment': report.human_analysis.company_assessment[:100],
                        'key_catalyst': report.human_analysis.key_catalysts[0] if report.human_analysis.key_catalysts else None
                    },
                    'metadata': {
                        'ai_weight': report.hybrid_recommendation.ai_weight,
                        'analysis_time': report.timestamp.isoformat(),
                        'data_quality': report.data_completeness
                    }
                }
            
            logger.info(f"Generated trading signals for {len(signals)} symbols")
            return signals
            
        except Exception as e:
            logger.error(f"Error getting trading signals: {e}")
            return {}
    
    def get_risk_assessment(self, portfolio: Dict[str, float]) -> Dict:
        """Get comprehensive risk assessment for current portfolio"""
        try:
            logger.info("Generating hybrid AI risk assessment")
            
            symbols = list(portfolio.keys())
            weights = list(portfolio.values())
            
            # Get analysis for all positions
            portfolio_results = self.hybrid_system.analyze_portfolio(symbols, max_workers=2)
            
            # Calculate portfolio-level metrics
            total_risk_score = 0.0
            weighted_confidence = 0.0
            ai_heavy_positions = 0
            human_heavy_positions = 0
            
            position_risks = {}
            
            for symbol, weight in portfolio.items():
                if symbol in portfolio_results:
                    report = portfolio_results[symbol]
                    
                    # Position-level risk
                    position_risk = report.risk_score * weight
                    total_risk_score += position_risk
                    
                    # Confidence weighting
                    weighted_confidence += report.confidence_score * weight
                    
                    # Count AI vs Human heavy positions
                    if report.hybrid_recommendation.ai_weight > 0.7:
                        ai_heavy_positions += 1
                    elif report.hybrid_recommendation.ai_weight < 0.3:
                        human_heavy_positions += 1
                    
                    position_risks[symbol] = {
                        'individual_risk': report.risk_score,
                        'weighted_risk': position_risk,
                        'confidence': report.confidence_score,
                        'ai_weight': report.hybrid_recommendation.ai_weight,
                        'recommendation': report.hybrid_recommendation.hybrid_recommendation
                    }
            
            # Portfolio diversification score
            diversification_score = 1.0 - np.sum(np.array(weights) ** 2)  # Herfindahl index
            
            risk_assessment = {
                'overall_risk_score': total_risk_score,
                'portfolio_confidence': weighted_confidence,
                'diversification_score': diversification_score,
                'ai_heavy_positions': ai_heavy_positions,
                'human_heavy_positions': human_heavy_positions,
                'balanced_positions': len(symbols) - ai_heavy_positions - human_heavy_positions,
                'position_risks': position_risks,
                'recommendations': self._generate_risk_recommendations(
                    total_risk_score, diversification_score, ai_heavy_positions, human_heavy_positions
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Hybrid AI risk assessment completed")
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error generating risk assessment: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_market_insights(self, symbols: List[str]) -> Dict:
        """Get market insights combining AI and human-like analysis"""
        try:
            logger.info("Generating hybrid market insights")
            
            # Analyze a sample of symbols
            sample_symbols = symbols[:5] if len(symbols) > 5 else symbols
            portfolio_results = self.hybrid_system.analyze_portfolio(sample_symbols, max_workers=2)
            
            # Aggregate insights
            ai_bullish = 0
            ai_bearish = 0
            human_bullish = 0
            human_bearish = 0
            
            sector_sentiment = {}
            key_opportunities = []
            key_risks = []
            
            for symbol, report in portfolio_results.items():
                # Count AI sentiment
                if report.ai_analysis.ai_recommendation in ['BUY', 'STRONG_BUY']:
                    ai_bullish += 1
                elif report.ai_analysis.ai_recommendation in ['SELL', 'STRONG_SELL']:
                    ai_bearish += 1
                
                # Count human sentiment
                if report.human_analysis.human_recommendation in ['BUY', 'STRONG_BUY']:
                    human_bullish += 1
                elif report.human_analysis.human_recommendation in ['SELL', 'STRONG_SELL']:
                    human_bearish += 1
                
                # Collect opportunities and risks
                key_opportunities.extend(report.human_analysis.unique_opportunities[:1])
                key_risks.extend(report.human_analysis.hidden_risks[:1])
            
            market_insights = {
                'ai_sentiment': {
                    'bullish_signals': ai_bullish,
                    'bearish_signals': ai_bearish,
                    'neutral_signals': len(sample_symbols) - ai_bullish - ai_bearish,
                    'overall_bias': 'BULLISH' if ai_bullish > ai_bearish else 'BEARISH' if ai_bearish > ai_bullish else 'NEUTRAL'
                },
                'human_sentiment': {
                    'bullish_signals': human_bullish,
                    'bearish_signals': human_bearish,
                    'neutral_signals': len(sample_symbols) - human_bullish - human_bearish,
                    'overall_bias': 'BULLISH' if human_bullish > human_bearish else 'BEARISH' if human_bearish > human_bullish else 'NEUTRAL'
                },
                'key_opportunities': list(set(key_opportunities))[:5],
                'key_risks': list(set(key_risks))[:5],
                'analysis_coverage': len(portfolio_results),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Market insights generated successfully")
            return market_insights
            
        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _recommendation_to_signal_strength(self, recommendation: str, confidence: float) -> float:
        """Convert recommendation to signal strength (0-1)"""
        strength_map = {
            'STRONG_BUY': 1.0,
            'BUY': 0.7,
            'HOLD': 0.0,
            'SELL': -0.7,
            'STRONG_SELL': -1.0
        }
        
        base_strength = strength_map.get(recommendation, 0.0)
        return base_strength * confidence
    
    def _generate_risk_recommendations(self, total_risk: float, diversification: float,
                                     ai_heavy: int, human_heavy: int) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if total_risk > 0.7:
            recommendations.append("High portfolio risk detected - consider reducing position sizes")
        
        if diversification < 0.5:
            recommendations.append("Low diversification - consider adding more positions")
        
        if ai_heavy > human_heavy * 2:
            recommendations.append("Portfolio heavily weighted toward AI-favored stocks - consider adding small-cap or distressed opportunities")
        elif human_heavy > ai_heavy * 2:
            recommendations.append("Portfolio heavily weighted toward human-favored stocks - consider adding large-cap liquid positions")
        
        if not recommendations:
            recommendations.append("Portfolio risk profile appears balanced")
        
        return recommendations
    
    def _create_fallback_analysis(self, symbol: str, error: str) -> Dict:
        """Create fallback analysis when hybrid system fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'hybrid_recommendation': 'HOLD',
            'confidence': 0.5,
            'expected_alpha': 0.0,
            'risk_score': 0.5,
            'error': error,
            'ai_analysis': {
                'recommendation': 'HOLD',
                'confidence': 0.5,
                'technical_signal': 'HOLD',
                'reasoning': ['Analysis unavailable due to error']
            },
            'human_analysis': {
                'recommendation': 'HOLD',
                'confidence': 0.5,
                'company_assessment': 'Assessment unavailable',
                'opportunities': [],
                'risks': [],
                'catalysts': []
            },
            'weighting': {
                'ai_weight': 0.5,
                'human_weight': 0.5,
                'rationale': 'Default weighting due to error'
            }
        }

# Integration functions for QuantSphere GUI
def get_hybrid_analysis_for_gui(symbol: str, config: Dict = None) -> Dict:
    """Standalone function for GUI integration"""
    try:
        hybrid_ai = QuantSphereHybridAI(config)
        return hybrid_ai.get_enhanced_analysis(symbol)
    except Exception as e:
        logger.error(f"Error in GUI integration for {symbol}: {e}")
        return {'error': str(e), 'symbol': symbol}

def get_portfolio_signals_for_trading(symbols: List[str], config: Dict = None) -> Dict:
    """Standalone function for trading engine integration"""
    try:
        hybrid_ai = QuantSphereHybridAI(config)
        return hybrid_ai.get_trading_signals(symbols)
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        return {}

# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🔗 QUANTSPHERE HYBRID AI INTEGRATION TEST")
    print(f"{'='*60}")
    
    # Initialize integration
    config = {
        'openai_api_key': None,  # Using mock responses
        'cache_duration': 3600
    }
    
    integration = QuantSphereHybridAI(config)
    
    # Test enhanced analysis
    test_symbol = 'AAPL'
    print(f"\n🔍 Testing Enhanced Analysis for {test_symbol}")
    analysis = integration.get_enhanced_analysis(test_symbol)
    
    print(f"Hybrid Recommendation: {analysis['hybrid_recommendation']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"AI Weight: {analysis['weighting']['ai_weight']:.1%}")
    print(f"Expected Alpha: {analysis['expected_alpha']:.2%}")
    
    # Test portfolio recommendations
    test_portfolio = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    print(f"\n📊 Testing Portfolio Recommendations")
    recommendations = integration.get_portfolio_recommendations(test_portfolio, top_n=3)
    
    for rec in recommendations:
        print(f"{rec['rank']}. {rec['symbol']}: {rec['recommendation']} "
              f"(confidence: {rec['confidence']:.2f}, alpha: {rec['expected_alpha']:.2%})")
    
    # Test trading signals
    print(f"\n📈 Testing Trading Signals")
    signals = integration.get_trading_signals(test_portfolio[:2])  # Limit for testing
    
    for symbol, signal in signals.items():
        print(f"{symbol}: {signal['signal']} (strength: {signal['strength']:.2f}, "
              f"size: {signal['position_size']:.1%})")
    
    print(f"\n✅ Integration test completed successfully!")
    print(f"🎯 Ready for QuantSphere platform integration")
