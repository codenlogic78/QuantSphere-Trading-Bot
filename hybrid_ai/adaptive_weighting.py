"""
Adaptive Weighting System for Hybrid AI Trading
Based on Cao et al. (2024) - Dynamic weighting between AI and Human analysis

This module implements the core innovation of the research:
- Adaptive weighting based on stock characteristics
- Performance tracking and learning
- Dynamic adjustment of AI vs Human influence
- Ensemble decision making
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeightingFactors:
    """Factors that influence AI vs Human weighting"""
    market_cap_factor: float
    liquidity_factor: float
    distress_factor: float
    sector_factor: float
    volatility_factor: float
    data_quality_factor: float
    
    def calculate_ai_weight(self) -> float:
        """Calculate AI weight based on all factors"""
        # Base weight
        ai_weight = 0.5
        
        # Apply factors
        ai_weight += self.market_cap_factor * 0.2
        ai_weight += self.liquidity_factor * 0.15
        ai_weight -= self.distress_factor * 0.2  # Distress favors human
        ai_weight += self.sector_factor * 0.1
        ai_weight -= self.volatility_factor * 0.1  # High volatility favors human
        ai_weight += self.data_quality_factor * 0.15
        
        # Clamp between 0.1 and 0.9
        return max(0.1, min(0.9, ai_weight))

@dataclass
class HybridRecommendation:
    """Combined recommendation from hybrid system"""
    symbol: str
    timestamp: datetime
    
    # Individual components
    ai_recommendation: str
    ai_confidence: float
    ai_weight: float
    
    human_recommendation: str
    human_confidence: float
    human_weight: float
    
    # Hybrid result
    hybrid_recommendation: str
    hybrid_confidence: float
    hybrid_score: float
    
    # Reasoning
    weighting_rationale: str
    decision_narrative: str
    risk_assessment: str
    
    # Performance tracking
    expected_return: float
    risk_adjusted_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'ai_recommendation': self.ai_recommendation,
            'ai_confidence': self.ai_confidence,
            'ai_weight': self.ai_weight,
            'human_recommendation': self.human_recommendation,
            'human_confidence': self.human_confidence,
            'human_weight': self.human_weight,
            'hybrid_recommendation': self.hybrid_recommendation,
            'hybrid_confidence': self.hybrid_confidence,
            'hybrid_score': self.hybrid_score,
            'weighting_rationale': self.weighting_rationale,
            'decision_narrative': self.decision_narrative,
            'risk_assessment': self.risk_assessment,
            'expected_return': self.expected_return,
            'risk_adjusted_score': self.risk_adjusted_score
        }

class PerformanceTracker:
    """Track performance of different weighting strategies"""
    
    def __init__(self, data_file: str = "hybrid_performance.json"):
        self.data_file = Path(data_file)
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self) -> Dict:
        """Load historical performance data"""
        try:
            if self.data_file.exists():
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'ai_only': {'total_return': 0.0, 'win_rate': 0.5, 'trades': 0},
                    'human_only': {'total_return': 0.0, 'win_rate': 0.5, 'trades': 0},
                    'hybrid': {'total_return': 0.0, 'win_rate': 0.5, 'trades': 0},
                    'by_characteristics': {}
                }
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return {}
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def record_trade_outcome(self, recommendation: HybridRecommendation, 
                           actual_return: float, holding_period_days: int):
        """Record the outcome of a trade for learning"""
        try:
            # Update overall performance
            for method in ['ai_only', 'human_only', 'hybrid']:
                if method not in self.performance_data:
                    self.performance_data[method] = {'total_return': 0.0, 'win_rate': 0.5, 'trades': 0}
            
            # Record hybrid performance
            hybrid_perf = self.performance_data['hybrid']
            hybrid_perf['total_return'] += actual_return
            hybrid_perf['trades'] += 1
            
            # Update win rate (simplified)
            if actual_return > 0:
                hybrid_perf['win_rate'] = (hybrid_perf['win_rate'] * (hybrid_perf['trades'] - 1) + 1) / hybrid_perf['trades']
            else:
                hybrid_perf['win_rate'] = (hybrid_perf['win_rate'] * (hybrid_perf['trades'] - 1)) / hybrid_perf['trades']
            
            # Record by characteristics for learning
            char_key = f"large_cap_{recommendation.ai_weight > 0.6}"
            if char_key not in self.performance_data['by_characteristics']:
                self.performance_data['by_characteristics'][char_key] = {
                    'returns': [], 'ai_weights': [], 'outcomes': []
                }
            
            char_data = self.performance_data['by_characteristics'][char_key]
            char_data['returns'].append(actual_return)
            char_data['ai_weights'].append(recommendation.ai_weight)
            char_data['outcomes'].append(1 if actual_return > 0 else 0)
            
            # Keep only recent data (last 100 trades per category)
            for key in ['returns', 'ai_weights', 'outcomes']:
                if len(char_data[key]) > 100:
                    char_data[key] = char_data[key][-100:]
            
            self._save_performance_data()
            logger.info(f"Recorded trade outcome for {recommendation.symbol}: {actual_return:.2%}")
            
        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")
    
    def get_optimal_weights(self, stock_characteristics: Dict) -> Tuple[float, str]:
        """Get optimal AI weight based on historical performance"""
        try:
            # Default weight
            base_weight = 0.6
            rationale = "Using base weighting strategy"
            
            # Check if we have performance data for similar characteristics
            char_key = f"large_cap_{stock_characteristics.get('is_large_cap', False)}"
            
            if char_key in self.performance_data.get('by_characteristics', {}):
                char_data = self.performance_data['by_characteristics'][char_key]
                
                if len(char_data['returns']) >= 10:  # Need minimum data
                    # Analyze performance by AI weight ranges
                    returns = np.array(char_data['returns'])
                    weights = np.array(char_data['ai_weights'])
                    
                    # Find optimal weight range
                    high_ai_returns = returns[weights > 0.7].mean() if len(returns[weights > 0.7]) > 0 else 0
                    low_ai_returns = returns[weights < 0.4].mean() if len(returns[weights < 0.4]) > 0 else 0
                    
                    if high_ai_returns > low_ai_returns + 0.01:  # 1% threshold
                        base_weight = 0.75
                        rationale = f"Historical data shows AI-heavy approach performs better for this stock type (+{(high_ai_returns - low_ai_returns):.1%})"
                    elif low_ai_returns > high_ai_returns + 0.01:
                        base_weight = 0.35
                        rationale = f"Historical data shows human-heavy approach performs better for this stock type (+{(low_ai_returns - high_ai_returns):.1%})"
            
            return base_weight, rationale
            
        except Exception as e:
            logger.error(f"Error getting optimal weights: {e}")
            return 0.6, "Using default weighting due to error"

class AdaptiveWeightingSystem:
    """Main adaptive weighting system"""
    
    def __init__(self, performance_file: str = "hybrid_performance.json"):
        self.performance_tracker = PerformanceTracker(performance_file)
        
    def calculate_weighting_factors(self, symbol: str, market_data: pd.DataFrame,
                                  fundamentals: Dict, characteristics: Dict) -> WeightingFactors:
        """Calculate all weighting factors for a stock"""
        try:
            # Market cap factor (large cap favors AI)
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap > 50_000_000_000:  # $50B+
                market_cap_factor = 1.0
            elif market_cap > 10_000_000_000:  # $10B+
                market_cap_factor = 0.5
            elif market_cap > 2_000_000_000:  # $2B+
                market_cap_factor = 0.0
            else:
                market_cap_factor = -0.5
            
            # Liquidity factor (high liquidity favors AI)
            avg_volume = fundamentals.get('avg_volume', 0)
            if avg_volume > 5_000_000:  # 5M+ daily volume
                liquidity_factor = 1.0
            elif avg_volume > 1_000_000:  # 1M+ daily volume
                liquidity_factor = 0.5
            else:
                liquidity_factor = -0.5
            
            # Distress factor (distress favors human, especially for asset-light)
            distress_score = characteristics.get('distress_score', 0)
            is_asset_light = characteristics.get('is_asset_light', False)
            
            if is_asset_light and distress_score > 0.6:
                distress_factor = 1.0  # Strong human preference
            elif distress_score > 0.4:
                distress_factor = 0.5
            else:
                distress_factor = 0.0
            
            # Sector factor (tech/growth favors AI, traditional favors human)
            sector = fundamentals.get('sector', '').lower()
            if any(tech_word in sector for tech_word in ['technology', 'communication', 'software']):
                sector_factor = 0.5
            elif any(trad_word in sector for trad_word in ['utilities', 'real estate', 'materials']):
                sector_factor = -0.5
            else:
                sector_factor = 0.0
            
            # Volatility factor (high volatility favors human intuition)
            if not market_data.empty and len(market_data) > 20:
                volatility = market_data['Close'].pct_change().std()
                if volatility > 0.03:  # 3%+ daily volatility
                    volatility_factor = 1.0
                elif volatility > 0.02:  # 2%+ daily volatility
                    volatility_factor = 0.5
                else:
                    volatility_factor = 0.0
            else:
                volatility_factor = 0.0
            
            # Data quality factor (more data favors AI)
            data_quality = 1.0 if not market_data.empty and len(market_data) > 100 else 0.5
            data_quality_factor = data_quality - 0.5  # Center around 0
            
            return WeightingFactors(
                market_cap_factor=market_cap_factor,
                liquidity_factor=liquidity_factor,
                distress_factor=distress_factor,
                sector_factor=sector_factor,
                volatility_factor=volatility_factor,
                data_quality_factor=data_quality_factor
            )
            
        except Exception as e:
            logger.error(f"Error calculating weighting factors: {e}")
            return WeightingFactors(0, 0, 0, 0, 0, 0)
    
    def combine_recommendations(self, symbol: str, ai_result, human_result,
                              market_data: pd.DataFrame, fundamentals: Dict,
                              characteristics: Dict) -> HybridRecommendation:
        """Combine AI and human recommendations using adaptive weighting"""
        try:
            logger.info(f"Combining recommendations for {symbol}")
            
            # Calculate weighting factors
            factors = self.calculate_weighting_factors(symbol, market_data, fundamentals, characteristics)
            base_ai_weight = factors.calculate_ai_weight()
            
            # Get performance-based adjustment
            optimal_weight, weight_rationale = self.performance_tracker.get_optimal_weights(characteristics)
            
            # Combine base and performance-based weights
            final_ai_weight = (base_ai_weight + optimal_weight) / 2
            final_human_weight = 1 - final_ai_weight
            
            # Convert recommendations to scores
            ai_score = self._recommendation_to_score(ai_result.ai_recommendation)
            human_score = self._recommendation_to_score(human_result.human_recommendation)
            
            # Weight the scores
            weighted_ai_score = ai_score * ai_result.ai_confidence * final_ai_weight
            weighted_human_score = human_score * human_result.human_confidence * final_human_weight
            
            # Combine scores
            hybrid_score = weighted_ai_score + weighted_human_score
            hybrid_confidence = (ai_result.ai_confidence * final_ai_weight + 
                               human_result.human_confidence * final_human_weight)
            
            # Convert back to recommendation
            hybrid_recommendation = self._score_to_recommendation(hybrid_score)
            
            # Generate narratives
            weighting_rationale = self._generate_weighting_rationale(
                factors, final_ai_weight, weight_rationale
            )
            
            decision_narrative = self._generate_decision_narrative(
                ai_result, human_result, final_ai_weight, hybrid_recommendation
            )
            
            risk_assessment = self._generate_risk_assessment(
                ai_result, human_result, hybrid_score
            )
            
            # Calculate expected return (simplified)
            expected_return = hybrid_score * 0.1  # Rough conversion
            risk_adjusted_score = hybrid_score * hybrid_confidence
            
            result = HybridRecommendation(
                symbol=symbol,
                timestamp=datetime.now(),
                ai_recommendation=ai_result.ai_recommendation,
                ai_confidence=ai_result.ai_confidence,
                ai_weight=final_ai_weight,
                human_recommendation=human_result.human_recommendation,
                human_confidence=human_result.human_confidence,
                human_weight=final_human_weight,
                hybrid_recommendation=hybrid_recommendation,
                hybrid_confidence=hybrid_confidence,
                hybrid_score=hybrid_score,
                weighting_rationale=weighting_rationale,
                decision_narrative=decision_narrative,
                risk_assessment=risk_assessment,
                expected_return=expected_return,
                risk_adjusted_score=risk_adjusted_score
            )
            
            logger.info(f"Hybrid recommendation for {symbol}: {hybrid_recommendation} "
                       f"(AI: {final_ai_weight:.2f}, Human: {final_human_weight:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error combining recommendations for {symbol}: {e}")
            return self._create_default_recommendation(symbol, ai_result, human_result)
    
    def _recommendation_to_score(self, recommendation: str) -> float:
        """Convert recommendation to numerical score"""
        score_map = {
            'STRONG_BUY': 2.0,
            'BUY': 1.0,
            'HOLD': 0.0,
            'SELL': -1.0,
            'STRONG_SELL': -2.0
        }
        return score_map.get(recommendation, 0.0)
    
    def _score_to_recommendation(self, score: float) -> str:
        """Convert numerical score to recommendation"""
        if score > 1.5:
            return 'STRONG_BUY'
        elif score > 0.5:
            return 'BUY'
        elif score > -0.5:
            return 'HOLD'
        elif score > -1.5:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _generate_weighting_rationale(self, factors: WeightingFactors, 
                                    ai_weight: float, perf_rationale: str) -> str:
        """Generate explanation for weighting decision"""
        try:
            rationale_parts = []
            
            if factors.market_cap_factor > 0.5:
                rationale_parts.append("Large-cap stock favors AI analysis")
            elif factors.market_cap_factor < -0.3:
                rationale_parts.append("Small-cap stock favors human insight")
            
            if factors.liquidity_factor > 0.5:
                rationale_parts.append("High liquidity supports quantitative analysis")
            elif factors.liquidity_factor < -0.3:
                rationale_parts.append("Low liquidity requires qualitative assessment")
            
            if factors.distress_factor > 0.5:
                rationale_parts.append("Financial distress indicators favor human judgment")
            
            if factors.volatility_factor > 0.5:
                rationale_parts.append("High volatility benefits from human intuition")
            
            base_rationale = "; ".join(rationale_parts) if rationale_parts else "Balanced characteristics"
            
            return f"AI Weight: {ai_weight:.1%} - {base_rationale}. {perf_rationale}"
            
        except Exception as e:
            return f"AI Weight: {ai_weight:.1%} - Standard weighting applied"
    
    def _generate_decision_narrative(self, ai_result, human_result, 
                                   ai_weight: float, hybrid_rec: str) -> str:
        """Generate narrative explaining the hybrid decision"""
        try:
            if ai_weight > 0.7:
                return f"Decision primarily driven by AI analysis ({ai_result.ai_recommendation}) with human insights providing context. Final recommendation: {hybrid_rec}"
            elif ai_weight < 0.3:
                return f"Decision primarily driven by human analysis ({human_result.human_recommendation}) with AI providing quantitative support. Final recommendation: {hybrid_rec}"
            else:
                return f"Balanced combination of AI ({ai_result.ai_recommendation}) and human ({human_result.human_recommendation}) analysis. Final recommendation: {hybrid_rec}"
        except:
            return f"Hybrid analysis recommends {hybrid_rec}"
    
    def _generate_risk_assessment(self, ai_result, human_result, hybrid_score: float) -> str:
        """Generate risk assessment for the hybrid recommendation"""
        try:
            risk_level = "Low" if abs(hybrid_score) < 0.5 else "Medium" if abs(hybrid_score) < 1.5 else "High"
            
            ai_risk = getattr(ai_result, 'risk_score', 0.5)
            
            return f"Risk Level: {risk_level} - Combined analysis suggests {risk_level.lower()} risk profile with hybrid confidence supporting the recommendation."
        except:
            return "Risk assessment: Moderate risk profile"
    
    def _create_default_recommendation(self, symbol: str, ai_result, human_result) -> HybridRecommendation:
        """Create default recommendation when combination fails"""
        return HybridRecommendation(
            symbol=symbol,
            timestamp=datetime.now(),
            ai_recommendation=getattr(ai_result, 'ai_recommendation', 'HOLD'),
            ai_confidence=getattr(ai_result, 'ai_confidence', 0.5),
            ai_weight=0.5,
            human_recommendation=getattr(human_result, 'human_recommendation', 'HOLD'),
            human_confidence=getattr(human_result, 'human_confidence', 0.5),
            human_weight=0.5,
            hybrid_recommendation='HOLD',
            hybrid_confidence=0.5,
            hybrid_score=0.0,
            weighting_rationale="Default weighting applied due to error",
            decision_narrative="Unable to generate detailed analysis",
            risk_assessment="Standard risk profile assumed",
            expected_return=0.0,
            risk_adjusted_score=0.0
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the adaptive weighting system
    from data_collector import MultiSourceDataCollector
    from ai_analyst import AIAnalyst
    from human_ai_analyst import HumanAIAnalyst
    
    print(f"\n{'='*60}")
    print(f"Testing Adaptive Weighting System")
    print(f"{'='*60}")
    
    # Initialize components
    collector = MultiSourceDataCollector({})
    ai_analyst = AIAnalyst()
    human_analyst = HumanAIAnalyst()
    weighting_system = AdaptiveWeightingSystem()
    
    # Test with different stock types
    test_symbols = ['AAPL', 'TSLA', 'GOOGL']  # Large cap tech stocks
    
    for symbol in test_symbols:
        print(f"\n{'='*40}")
        print(f"Testing {symbol}")
        print(f"{'='*40}")
        
        # Collect data
        data = collector.collect_comprehensive_data(symbol)
        market_data = pd.DataFrame(data['market_data'])
        
        if not market_data.empty:
            # Convert index to datetime if needed
            if 'Date' in market_data.columns:
                market_data['Date'] = pd.to_datetime(market_data['Date'])
                market_data.set_index('Date', inplace=True)
            
            # Get individual analyses
            ai_result = ai_analyst.analyze_stock(
                symbol, market_data, data['fundamentals'], data['sentiment']
            )
            
            human_result = human_analyst.analyze_stock(
                symbol, market_data, data['fundamentals'], 
                data['sentiment'], data['characteristics']
            )
            
            # Combine with adaptive weighting
            hybrid_result = weighting_system.combine_recommendations(
                symbol, ai_result, human_result, market_data,
                data['fundamentals'], data['characteristics']
            )
            
            # Display results
            print(f"AI Recommendation: {hybrid_result.ai_recommendation} (confidence: {hybrid_result.ai_confidence:.2f})")
            print(f"Human Recommendation: {hybrid_result.human_recommendation} (confidence: {hybrid_result.human_confidence:.2f})")
            print(f"Hybrid Recommendation: {hybrid_result.hybrid_recommendation} (confidence: {hybrid_result.hybrid_confidence:.2f})")
            print(f"AI Weight: {hybrid_result.ai_weight:.1%}")
            print(f"Weighting Rationale: {hybrid_result.weighting_rationale}")
            print(f"Decision Narrative: {hybrid_result.decision_narrative[:100]}...")
        else:
            print(f"❌ No data available for {symbol}")
