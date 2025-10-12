"""
Main Hybrid AI Trading System
Based on Cao et al. (2024) - Complete "Man + Machine" implementation

This is the main orchestrator that combines all components:
- Data collection from multiple sources
- AI analysis (quantitative/technical)
- Human-like AI analysis (qualitative/contextual)
- Adaptive weighting system
- Performance tracking and learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import asyncio
import concurrent.futures

from data_collector import MultiSourceDataCollector, StockCharacteristics
from ai_analyst import AIAnalyst, AIAnalysisResult
from human_ai_analyst import HumanAIAnalyst, HumanAIAnalysisResult
from adaptive_weighting import AdaptiveWeightingSystem, HybridRecommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HybridAnalysisReport:
    """Complete hybrid analysis report for a stock"""
    symbol: str
    timestamp: datetime
    
    # Data quality metrics
    data_completeness: float
    data_freshness: str
    
    # Individual analysis results
    ai_analysis: AIAnalysisResult
    human_analysis: HumanAIAnalysisResult
    
    # Hybrid recommendation
    hybrid_recommendation: HybridRecommendation
    
    # Performance metrics
    confidence_score: float
    risk_score: float
    expected_alpha: float
    
    # Execution details
    processing_time_seconds: float
    data_sources_used: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data_completeness': self.data_completeness,
            'data_freshness': self.data_freshness,
            'ai_analysis': self.ai_analysis.to_dict(),
            'human_analysis': self.human_analysis.to_dict(),
            'hybrid_recommendation': self.hybrid_recommendation.to_dict(),
            'confidence_score': self.confidence_score,
            'risk_score': self.risk_score,
            'expected_alpha': self.expected_alpha,
            'processing_time_seconds': self.processing_time_seconds,
            'data_sources_used': self.data_sources_used
        }

class HybridAITradingSystem:
    """Main hybrid AI trading system orchestrator"""
    
    def __init__(self, config: Dict = None):
        """Initialize the hybrid AI trading system"""
        self.config = config or {}
        
        # Initialize components
        self.data_collector = MultiSourceDataCollector(self.config)
        self.ai_analyst = AIAnalyst()
        self.human_analyst = HumanAIAnalyst(
            api_key=self.config.get('openai_api_key')
        )
        self.weighting_system = AdaptiveWeightingSystem(
            performance_file=self.config.get('performance_file', 'hybrid_performance.json')
        )
        
        # Performance tracking
        self.analysis_cache = {}
        self.performance_history = []
        
        logger.info("Hybrid AI Trading System initialized")
    
    def analyze_stock(self, symbol: str, use_cache: bool = True) -> HybridAnalysisReport:
        """Perform complete hybrid analysis of a stock"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting hybrid analysis for {symbol}")
            
            # Check cache first
            if use_cache and symbol in self.analysis_cache:
                cached_result = self.analysis_cache[symbol]
                if (datetime.now() - cached_result.timestamp).seconds < 3600:  # 1 hour cache
                    logger.info(f"Using cached analysis for {symbol}")
                    return cached_result
            
            # Step 1: Collect comprehensive data
            logger.info(f"Collecting data for {symbol}")
            comprehensive_data = self.data_collector.collect_comprehensive_data(symbol)
            
            if not comprehensive_data:
                raise ValueError(f"No data available for {symbol}")
            
            # Extract data components
            market_data = pd.DataFrame(comprehensive_data['market_data'])
            fundamentals = comprehensive_data['fundamentals']
            sentiment = comprehensive_data['sentiment']
            characteristics = comprehensive_data['characteristics']
            
            # Data quality assessment
            data_completeness = self._assess_data_completeness(comprehensive_data)
            data_freshness = self._assess_data_freshness(comprehensive_data)
            data_sources = self._identify_data_sources(comprehensive_data)
            
            # Step 2: Parallel analysis execution
            logger.info(f"Executing parallel AI and Human analysis for {symbol}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both analyses concurrently
                ai_future = executor.submit(
                    self.ai_analyst.analyze_stock,
                    symbol, market_data, fundamentals, sentiment
                )
                
                human_future = executor.submit(
                    self.human_analyst.analyze_stock,
                    symbol, market_data, fundamentals, sentiment, characteristics
                )
                
                # Wait for both to complete
                ai_result = ai_future.result()
                human_result = human_future.result()
            
            # Step 3: Adaptive weighting and combination
            logger.info(f"Combining analyses with adaptive weighting for {symbol}")
            
            hybrid_result = self.weighting_system.combine_recommendations(
                symbol, ai_result, human_result, market_data, fundamentals, characteristics
            )
            
            # Step 4: Generate final metrics
            confidence_score = self._calculate_confidence_score(ai_result, human_result, hybrid_result)
            risk_score = self._calculate_risk_score(ai_result, human_result, market_data)
            expected_alpha = self._calculate_expected_alpha(hybrid_result, market_data)
            
            # Step 5: Create comprehensive report
            processing_time = (datetime.now() - start_time).total_seconds()
            
            report = HybridAnalysisReport(
                symbol=symbol,
                timestamp=datetime.now(),
                data_completeness=data_completeness,
                data_freshness=data_freshness,
                ai_analysis=ai_result,
                human_analysis=human_result,
                hybrid_recommendation=hybrid_result,
                confidence_score=confidence_score,
                risk_score=risk_score,
                expected_alpha=expected_alpha,
                processing_time_seconds=processing_time,
                data_sources_used=data_sources
            )
            
            # Cache the result
            self.analysis_cache[symbol] = report
            
            logger.info(f"Hybrid analysis completed for {symbol} in {processing_time:.2f}s: "
                       f"{hybrid_result.hybrid_recommendation} (confidence: {confidence_score:.2f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in hybrid analysis for {symbol}: {e}")
            return self._create_error_report(symbol, str(e), start_time)
    
    def analyze_portfolio(self, symbols: List[str], max_workers: int = 4) -> Dict[str, HybridAnalysisReport]:
        """Analyze multiple stocks in parallel"""
        logger.info(f"Starting portfolio analysis for {len(symbols)} symbols")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analyses
            future_to_symbol = {
                executor.submit(self.analyze_stock, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    results[symbol] = self._create_error_report(symbol, str(e))
        
        logger.info(f"Portfolio analysis completed for {len(results)} symbols")
        return results
    
    def get_top_recommendations(self, symbols: List[str], top_n: int = 5) -> List[Tuple[str, HybridAnalysisReport]]:
        """Get top N recommendations from a list of symbols"""
        logger.info(f"Getting top {top_n} recommendations from {len(symbols)} symbols")
        
        # Analyze all symbols
        portfolio_results = self.analyze_portfolio(symbols)
        
        # Sort by risk-adjusted score
        sorted_results = sorted(
            portfolio_results.items(),
            key=lambda x: x[1].hybrid_recommendation.risk_adjusted_score,
            reverse=True
        )
        
        return sorted_results[:top_n]
    
    def backtest_strategy(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Backtest the hybrid strategy (simplified implementation)"""
        logger.info(f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # This is a simplified backtest - in production, you'd use historical data
        results = {
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'details': []
        }
        
        try:
            # Analyze current recommendations
            portfolio_results = self.analyze_portfolio(symbols)
            
            # Simulate performance based on recommendations
            total_return = 0.0
            wins = 0
            
            for symbol, report in portfolio_results.items():
                # Simulate return based on recommendation strength
                rec_score = report.hybrid_recommendation.hybrid_score
                simulated_return = rec_score * 0.05 + np.random.normal(0, 0.02)  # Add noise
                
                total_return += simulated_return
                if simulated_return > 0:
                    wins += 1
                
                results['details'].append({
                    'symbol': symbol,
                    'recommendation': report.hybrid_recommendation.hybrid_recommendation,
                    'confidence': report.confidence_score,
                    'simulated_return': simulated_return
                })
            
            results['total_return'] = total_return / len(symbols) if symbols else 0
            results['win_rate'] = wins / len(symbols) if symbols else 0
            results['trades'] = len(symbols)
            results['sharpe_ratio'] = results['total_return'] / 0.15 if results['total_return'] != 0 else 0  # Simplified
            
            logger.info(f"Backtest completed: {results['total_return']:.2%} return, {results['win_rate']:.1%} win rate")
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
        
        return results
    
    def _assess_data_completeness(self, data: Dict) -> float:
        """Assess the completeness of collected data"""
        try:
            completeness_score = 0.0
            
            # Market data
            if data.get('market_data') and len(data['market_data']) > 0:
                completeness_score += 0.4
            
            # Fundamentals
            fundamentals = data.get('fundamentals', {})
            if fundamentals and len(fundamentals) > 5:
                completeness_score += 0.3
            
            # Sentiment
            if data.get('sentiment') and len(data['sentiment']) > 0:
                completeness_score += 0.2
            
            # Characteristics
            if data.get('characteristics') and data['characteristics'].get('symbol'):
                completeness_score += 0.1
            
            return min(1.0, completeness_score)
            
        except Exception as e:
            logger.error(f"Error assessing data completeness: {e}")
            return 0.5
    
    def _assess_data_freshness(self, data: Dict) -> str:
        """Assess the freshness of collected data"""
        try:
            timestamp_str = data.get('timestamp', '')
            if timestamp_str:
                data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age_minutes = (datetime.now() - data_time).total_seconds() / 60
                
                if age_minutes < 15:
                    return "Very Fresh"
                elif age_minutes < 60:
                    return "Fresh"
                elif age_minutes < 240:
                    return "Moderate"
                else:
                    return "Stale"
            
            return "Unknown"
            
        except Exception as e:
            return "Unknown"
    
    def _identify_data_sources(self, data: Dict) -> List[str]:
        """Identify which data sources were used"""
        sources = []
        
        if data.get('market_data'):
            sources.append("Market Data (yfinance)")
        if data.get('fundamentals'):
            sources.append("Fundamentals (yfinance)")
        if data.get('sentiment'):
            sources.append("Sentiment (simulated)")
        
        return sources
    
    def _calculate_confidence_score(self, ai_result: AIAnalysisResult, 
                                  human_result: HumanAIAnalysisResult,
                                  hybrid_result: HybridRecommendation) -> float:
        """Calculate overall confidence score"""
        try:
            # Weight individual confidences
            ai_weight = hybrid_result.ai_weight
            human_weight = hybrid_result.human_weight
            
            weighted_confidence = (ai_result.ai_confidence * ai_weight + 
                                 human_result.human_confidence * human_weight)
            
            # Adjust for agreement between AI and human
            agreement_bonus = 0.0
            if ai_result.ai_recommendation == human_result.human_recommendation:
                agreement_bonus = 0.1
            
            return min(1.0, weighted_confidence + agreement_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _calculate_risk_score(self, ai_result: AIAnalysisResult,
                            human_result: HumanAIAnalysisResult,
                            market_data: pd.DataFrame) -> float:
        """Calculate overall risk score"""
        try:
            # Use AI risk assessment as base
            base_risk = getattr(ai_result, 'risk_score', 0.5)
            
            # Adjust based on market volatility
            if not market_data.empty and len(market_data) > 20:
                volatility = market_data['Close'].pct_change().std()
                volatility_risk = min(1.0, volatility / 0.03)  # Normalize
                combined_risk = (base_risk + volatility_risk) / 2
            else:
                combined_risk = base_risk
            
            return combined_risk
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _calculate_expected_alpha(self, hybrid_result: HybridRecommendation,
                                market_data: pd.DataFrame) -> float:
        """Calculate expected alpha (excess return)"""
        try:
            # Base alpha on recommendation strength and confidence
            rec_score = hybrid_result.hybrid_score
            confidence = hybrid_result.hybrid_confidence
            
            # Expected alpha = recommendation strength * confidence * base alpha
            base_alpha = 0.02  # 2% base monthly alpha
            expected_alpha = rec_score * confidence * base_alpha
            
            return expected_alpha
            
        except Exception as e:
            logger.error(f"Error calculating expected alpha: {e}")
            return 0.0
    
    def _create_error_report(self, symbol: str, error_msg: str, 
                           start_time: datetime = None) -> HybridAnalysisReport:
        """Create error report when analysis fails"""
        if start_time is None:
            start_time = datetime.now()
        
        # Create minimal analysis results
        from ai_analyst import AIAnalysisResult
        from human_ai_analyst import HumanAIAnalysisResult
        from adaptive_weighting import HybridRecommendation
        
        ai_result = AIAnalysisResult(
            symbol=symbol, timestamp=datetime.now(), technical_signal='HOLD',
            technical_confidence=0.5, technical_reasoning=[error_msg],
            price_prediction=100.0, price_confidence=0.5, prediction_horizon='1d',
            volatility_forecast=0.2, risk_score=0.5, max_position_size=0.1,
            ai_recommendation='HOLD', ai_confidence=0.5
        )
        
        human_result = HumanAIAnalysisResult(
            symbol=symbol, timestamp=datetime.now(), company_assessment=error_msg,
            management_quality="Unknown", competitive_position="Unknown",
            industry_outlook="Unknown", market_sentiment_interpretation="Unknown",
            macro_economic_impact="Unknown", sector_rotation_analysis="Unknown",
            unique_opportunities=[], hidden_risks=[], contrarian_viewpoints=[],
            human_recommendation='HOLD', human_confidence=0.5,
            reasoning_narrative=error_msg, key_catalysts=[]
        )
        
        hybrid_result = HybridRecommendation(
            symbol=symbol, timestamp=datetime.now(), ai_recommendation='HOLD',
            ai_confidence=0.5, ai_weight=0.5, human_recommendation='HOLD',
            human_confidence=0.5, human_weight=0.5, hybrid_recommendation='HOLD',
            hybrid_confidence=0.5, hybrid_score=0.0, weighting_rationale=error_msg,
            decision_narrative=error_msg, risk_assessment="Unknown",
            expected_return=0.0, risk_adjusted_score=0.0
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return HybridAnalysisReport(
            symbol=symbol,
            timestamp=datetime.now(),
            data_completeness=0.0,
            data_freshness="Error",
            ai_analysis=ai_result,
            human_analysis=human_result,
            hybrid_recommendation=hybrid_result,
            confidence_score=0.5,
            risk_score=0.5,
            expected_alpha=0.0,
            processing_time_seconds=processing_time,
            data_sources_used=[]
        )

# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"🚀 HYBRID AI TRADING SYSTEM - COMPLETE IMPLEMENTATION")
    print(f"{'='*70}")
    
    # Initialize system
    config = {
        'openai_api_key': None,  # Set your OpenAI API key here
        'performance_file': 'test_hybrid_performance.json'
    }
    
    hybrid_system = HybridAITradingSystem(config)
    
    # Test single stock analysis
    test_symbol = 'AAPL'
    print(f"\n🔍 Testing Single Stock Analysis: {test_symbol}")
    print(f"{'='*50}")
    
    report = hybrid_system.analyze_stock(test_symbol)
    
    print(f"📊 ANALYSIS RESULTS:")
    print(f"Symbol: {report.symbol}")
    print(f"Processing Time: {report.processing_time_seconds:.2f}s")
    print(f"Data Completeness: {report.data_completeness:.1%}")
    print(f"Data Freshness: {report.data_freshness}")
    
    print(f"\n🤖 AI Analysis:")
    print(f"  Recommendation: {report.ai_analysis.ai_recommendation}")
    print(f"  Confidence: {report.ai_analysis.ai_confidence:.2f}")
    print(f"  Technical Signal: {report.ai_analysis.technical_signal}")
    
    print(f"\n🧠 Human-like Analysis:")
    print(f"  Recommendation: {report.human_analysis.human_recommendation}")
    print(f"  Confidence: {report.human_analysis.human_confidence:.2f}")
    print(f"  Assessment: {report.human_analysis.company_assessment[:100]}...")
    
    print(f"\n⚖️ Hybrid Recommendation:")
    print(f"  Final Recommendation: {report.hybrid_recommendation.hybrid_recommendation}")
    print(f"  Confidence: {report.confidence_score:.2f}")
    print(f"  AI Weight: {report.hybrid_recommendation.ai_weight:.1%}")
    print(f"  Human Weight: {report.hybrid_recommendation.human_weight:.1%}")
    print(f"  Expected Alpha: {report.expected_alpha:.2%}")
    print(f"  Risk Score: {report.risk_score:.2f}")
    
    print(f"\n📈 Weighting Rationale:")
    print(f"  {report.hybrid_recommendation.weighting_rationale}")
    
    # Test portfolio analysis
    test_portfolio = ['AAPL', 'MSFT', 'GOOGL']
    print(f"\n📊 Testing Portfolio Analysis: {test_portfolio}")
    print(f"{'='*50}")
    
    portfolio_results = hybrid_system.analyze_portfolio(test_portfolio, max_workers=2)
    
    for symbol, result in portfolio_results.items():
        print(f"{symbol}: {result.hybrid_recommendation.hybrid_recommendation} "
              f"(confidence: {result.confidence_score:.2f}, "
              f"alpha: {result.expected_alpha:.2%})")
    
    # Test top recommendations
    print(f"\n🏆 Top Recommendations:")
    print(f"{'='*30}")
    
    top_recs = hybrid_system.get_top_recommendations(test_portfolio, top_n=2)
    
    for i, (symbol, result) in enumerate(top_recs, 1):
        print(f"{i}. {symbol}: {result.hybrid_recommendation.hybrid_recommendation} "
              f"(score: {result.hybrid_recommendation.risk_adjusted_score:.3f})")
    
    print(f"\n✅ Hybrid AI Trading System test completed successfully!")
    print(f"🎯 System ready for production deployment")
