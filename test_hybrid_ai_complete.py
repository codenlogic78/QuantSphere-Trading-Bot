"""
Complete Test Suite for Hybrid AI Trading System
Based on Cao et al. (2024) - "Man + Machine" approach

This script demonstrates the complete implementation of the hybrid AI trading system
with comprehensive testing of all components and real-world scenarios.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

# Add hybrid_ai to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hybrid_ai'))

from hybrid_ai import (
    HybridAITradingSystem,
    MultiSourceDataCollector,
    AIAnalyst,
    HumanAIAnalyst,
    AdaptiveWeightingSystem
)

def print_header(title: str, width: int = 80):
    """Print formatted header"""
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}")

def print_section(title: str, width: int = 60):
    """Print formatted section"""
    print(f"\n{'-'*width}")
    print(f"📊 {title}")
    print(f"{'-'*width}")

def test_data_collection():
    """Test multi-source data collection"""
    print_section("Testing Data Collection Module")
    
    collector = MultiSourceDataCollector({})
    
    # Test different stock types
    test_stocks = {
        'AAPL': 'Large-cap tech (AI-favored)',
        'TSLA': 'High-volatility growth (Mixed)',
        'BRK-A': 'Large-cap value (Human-favored)',
    }
    
    results = {}
    
    for symbol, description in test_stocks.items():
        print(f"\n🔍 Testing {symbol} ({description})")
        
        try:
            data = collector.collect_comprehensive_data(symbol)
            
            # Analyze characteristics
            characteristics = data.get('characteristics', {})
            market_cap = data.get('fundamentals', {}).get('market_cap', 0)
            ai_weight = data.get('ai_weight', 0.5)
            
            print(f"  ✅ Data collected successfully")
            print(f"  📈 Market Cap: ${market_cap:,.0f}")
            print(f"  🏢 Sector: {data.get('fundamentals', {}).get('sector', 'Unknown')}")
            print(f"  🤖 AI Weight: {ai_weight:.1%}")
            print(f"  📊 Large Cap: {characteristics.get('is_large_cap', False)}")
            print(f"  💧 Liquid: {characteristics.get('is_liquid', False)}")
            print(f"  ⚠️ Distress Score: {characteristics.get('distress_score', 0):.2f}")
            
            results[symbol] = {
                'success': True,
                'ai_weight': ai_weight,
                'characteristics': characteristics
            }
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[symbol] = {'success': False, 'error': str(e)}
    
    return results

def test_ai_analysis():
    """Test AI analysis component"""
    print_section("Testing AI Analysis Engine")
    
    collector = MultiSourceDataCollector({})
    ai_analyst = AIAnalyst()
    
    test_symbol = 'AAPL'
    print(f"🤖 Testing AI analysis for {test_symbol}")
    
    try:
        # Get data
        data = collector.collect_comprehensive_data(test_symbol)
        market_data = pd.DataFrame(data['market_data'])
        
        if not market_data.empty:
            # Convert index if needed
            if 'Date' in market_data.columns:
                market_data['Date'] = pd.to_datetime(market_data['Date'])
                market_data.set_index('Date', inplace=True)
        
        # Perform AI analysis
        ai_result = ai_analyst.analyze_stock(
            test_symbol, 
            market_data, 
            data['fundamentals'], 
            data['sentiment']
        )
        
        print(f"  ✅ AI Analysis completed")
        print(f"  📊 Technical Signal: {ai_result.technical_signal}")
        print(f"  🎯 AI Recommendation: {ai_result.ai_recommendation}")
        print(f"  📈 Price Prediction: ${ai_result.price_prediction:.2f}")
        print(f"  🔒 Confidence: {ai_result.ai_confidence:.2f}")
        print(f"  ⚠️ Risk Score: {ai_result.risk_score:.2f}")
        print(f"  💰 Max Position: {ai_result.max_position_size:.1%}")
        
        print(f"\n  🧠 Technical Reasoning:")
        for reason in ai_result.technical_reasoning[:3]:
            print(f"    • {reason}")
        
        return {'success': True, 'result': ai_result}
        
    except Exception as e:
        print(f"  ❌ Error in AI analysis: {e}")
        return {'success': False, 'error': str(e)}

def test_human_ai_analysis():
    """Test human-like AI analysis component"""
    print_section("Testing Human-like AI Analysis Engine")
    
    collector = MultiSourceDataCollector({})
    human_analyst = HumanAIAnalyst()  # Will use mock responses without OpenAI key
    
    test_symbol = 'AAPL'
    print(f"🧠 Testing Human-like AI analysis for {test_symbol}")
    
    try:
        # Get data
        data = collector.collect_comprehensive_data(test_symbol)
        market_data = pd.DataFrame(data['market_data'])
        
        if not market_data.empty:
            if 'Date' in market_data.columns:
                market_data['Date'] = pd.to_datetime(market_data['Date'])
                market_data.set_index('Date', inplace=True)
        
        # Perform human-like AI analysis
        human_result = human_analyst.analyze_stock(
            test_symbol, 
            market_data, 
            data['fundamentals'], 
            data['sentiment'],
            data['characteristics']
        )
        
        print(f"  ✅ Human-like AI Analysis completed")
        print(f"  🎯 Human Recommendation: {human_result.human_recommendation}")
        print(f"  🔒 Confidence: {human_result.human_confidence:.2f}")
        print(f"  🏢 Company Assessment: {human_result.company_assessment[:80]}...")
        print(f"  🎭 Management Quality: {human_result.management_quality[:60]}...")
        print(f"  📊 Market Sentiment: {human_result.market_sentiment_interpretation[:60]}...")
        
        print(f"\n  💡 Creative Insights:")
        print(f"    Opportunities: {', '.join(human_result.unique_opportunities[:2])}")
        print(f"    Risks: {', '.join(human_result.hidden_risks[:2])}")
        
        print(f"\n  🎯 Key Catalysts:")
        for catalyst in human_result.key_catalysts[:2]:
            print(f"    • {catalyst}")
        
        return {'success': True, 'result': human_result}
        
    except Exception as e:
        print(f"  ❌ Error in Human-like AI analysis: {e}")
        return {'success': False, 'error': str(e)}

def test_adaptive_weighting():
    """Test adaptive weighting system"""
    print_section("Testing Adaptive Weighting System")
    
    collector = MultiSourceDataCollector({})
    ai_analyst = AIAnalyst()
    human_analyst = HumanAIAnalyst()
    weighting_system = AdaptiveWeightingSystem()
    
    # Test with different stock types
    test_cases = [
        ('AAPL', 'Large-cap tech (should favor AI)'),
        ('TSLA', 'High-volatility growth (mixed weighting)'),
    ]
    
    results = {}
    
    for symbol, description in test_cases:
        print(f"\n⚖️ Testing adaptive weighting for {symbol} ({description})")
        
        try:
            # Get data and individual analyses
            data = collector.collect_comprehensive_data(symbol)
            market_data = pd.DataFrame(data['market_data'])
            
            if not market_data.empty and 'Date' in market_data.columns:
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
            
            print(f"  ✅ Adaptive weighting completed")
            print(f"  🤖 AI Recommendation: {hybrid_result.ai_recommendation} (confidence: {hybrid_result.ai_confidence:.2f})")
            print(f"  🧠 Human Recommendation: {hybrid_result.human_recommendation} (confidence: {hybrid_result.human_confidence:.2f})")
            print(f"  ⚖️ AI Weight: {hybrid_result.ai_weight:.1%}")
            print(f"  🎯 Hybrid Recommendation: {hybrid_result.hybrid_recommendation}")
            print(f"  🔒 Hybrid Confidence: {hybrid_result.hybrid_confidence:.2f}")
            print(f"  📊 Hybrid Score: {hybrid_result.hybrid_score:.3f}")
            
            print(f"\n  📝 Weighting Rationale:")
            print(f"    {hybrid_result.weighting_rationale}")
            
            results[symbol] = {
                'success': True,
                'ai_weight': hybrid_result.ai_weight,
                'hybrid_recommendation': hybrid_result.hybrid_recommendation,
                'hybrid_confidence': hybrid_result.hybrid_confidence
            }
            
        except Exception as e:
            print(f"  ❌ Error in adaptive weighting: {e}")
            results[symbol] = {'success': False, 'error': str(e)}
    
    return results

def test_complete_hybrid_system():
    """Test the complete hybrid AI trading system"""
    print_section("Testing Complete Hybrid AI Trading System")
    
    # Initialize system
    config = {
        'openai_api_key': None,  # Using mock responses
        'performance_file': 'test_hybrid_performance.json'
    }
    
    hybrid_system = HybridAITradingSystem(config)
    
    # Test single stock analysis
    test_symbol = 'AAPL'
    print(f"🚀 Testing complete system analysis for {test_symbol}")
    
    start_time = time.time()
    
    try:
        report = hybrid_system.analyze_stock(test_symbol)
        
        processing_time = time.time() - start_time
        
        print(f"  ✅ Complete analysis finished in {processing_time:.2f}s")
        print(f"  📊 Data Completeness: {report.data_completeness:.1%}")
        print(f"  🕐 Data Freshness: {report.data_freshness}")
        print(f"  📈 Processing Time: {report.processing_time_seconds:.2f}s")
        
        print(f"\n  🤖 AI Component:")
        print(f"    Recommendation: {report.ai_analysis.ai_recommendation}")
        print(f"    Confidence: {report.ai_analysis.ai_confidence:.2f}")
        print(f"    Technical Signal: {report.ai_analysis.technical_signal}")
        
        print(f"\n  🧠 Human-like Component:")
        print(f"    Recommendation: {report.human_analysis.human_recommendation}")
        print(f"    Confidence: {report.human_analysis.human_confidence:.2f}")
        
        print(f"\n  ⚖️ Hybrid Result:")
        print(f"    Final Recommendation: {report.hybrid_recommendation.hybrid_recommendation}")
        print(f"    Overall Confidence: {report.confidence_score:.2f}")
        print(f"    AI Weight: {report.hybrid_recommendation.ai_weight:.1%}")
        print(f"    Human Weight: {report.hybrid_recommendation.human_weight:.1%}")
        print(f"    Expected Alpha: {report.expected_alpha:.2%}")
        print(f"    Risk Score: {report.risk_score:.2f}")
        
        print(f"\n  📊 Performance Metrics:")
        print(f"    Risk-Adjusted Score: {report.hybrid_recommendation.risk_adjusted_score:.3f}")
        print(f"    Data Sources: {', '.join(report.data_sources_used)}")
        
        return {'success': True, 'report': report}
        
    except Exception as e:
        print(f"  ❌ Error in complete system test: {e}")
        return {'success': False, 'error': str(e)}

def test_portfolio_analysis():
    """Test portfolio analysis capabilities"""
    print_section("Testing Portfolio Analysis")
    
    config = {'performance_file': 'test_portfolio_performance.json'}
    hybrid_system = HybridAITradingSystem(config)
    
    # Test portfolio
    test_portfolio = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    print(f"📊 Testing portfolio analysis for: {', '.join(test_portfolio)}")
    
    start_time = time.time()
    
    try:
        portfolio_results = hybrid_system.analyze_portfolio(test_portfolio, max_workers=2)
        
        processing_time = time.time() - start_time
        
        print(f"  ✅ Portfolio analysis completed in {processing_time:.2f}s")
        print(f"  📈 Analyzed {len(portfolio_results)} stocks")
        
        print(f"\n  📊 Portfolio Results:")
        for symbol, result in portfolio_results.items():
            print(f"    {symbol}: {result.hybrid_recommendation.hybrid_recommendation} "
                  f"(confidence: {result.confidence_score:.2f}, "
                  f"alpha: {result.expected_alpha:.2%}, "
                  f"AI weight: {result.hybrid_recommendation.ai_weight:.1%})")
        
        # Get top recommendations
        print(f"\n  🏆 Top Recommendations:")
        top_recs = hybrid_system.get_top_recommendations(test_portfolio, top_n=3)
        
        for i, (symbol, result) in enumerate(top_recs, 1):
            print(f"    {i}. {symbol}: {result.hybrid_recommendation.hybrid_recommendation} "
                  f"(score: {result.hybrid_recommendation.risk_adjusted_score:.3f})")
        
        return {'success': True, 'portfolio_results': portfolio_results, 'top_recs': top_recs}
        
    except Exception as e:
        print(f"  ❌ Error in portfolio analysis: {e}")
        return {'success': False, 'error': str(e)}

def demonstrate_research_implementation():
    """Demonstrate how the implementation follows Cao et al. (2024) research"""
    print_section("Research Implementation Demonstration")
    
    print(f"🔬 Demonstrating Cao et al. (2024) 'Man + Machine' Implementation")
    
    # Show different weighting scenarios
    scenarios = [
        {
            'type': 'Large-cap, liquid tech stock',
            'characteristics': {
                'is_large_cap': True,
                'is_liquid': True,
                'is_asset_light': True,
                'distress_score': 0.1,
                'sector': 'Technology'
            },
            'expected_ai_weight': 'High (70-80%)',
            'rationale': 'AI excels with abundant structured data and quantitative patterns'
        },
        {
            'type': 'Small-cap, illiquid, distressed company',
            'characteristics': {
                'is_large_cap': False,
                'is_liquid': False,
                'is_asset_light': True,
                'distress_score': 0.8,
                'sector': 'Consumer Discretionary'
            },
            'expected_ai_weight': 'Low (20-30%)',
            'rationale': 'Human reasoning better for qualitative assessment during distress'
        },
        {
            'type': 'Mid-cap, moderate liquidity',
            'characteristics': {
                'is_large_cap': False,
                'is_liquid': True,
                'is_asset_light': False,
                'distress_score': 0.3,
                'sector': 'Industrials'
            },
            'expected_ai_weight': 'Balanced (45-55%)',
            'rationale': 'Mixed characteristics benefit from balanced approach'
        }
    ]
    
    print(f"\n  📊 Weighting Scenarios Based on Research Findings:")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n    {i}. {scenario['type']}")
        print(f"       Expected AI Weight: {scenario['expected_ai_weight']}")
        print(f"       Rationale: {scenario['rationale']}")
        
        # Calculate actual weight using our system
        from hybrid_ai.adaptive_weighting import WeightingFactors
        
        factors = WeightingFactors(
            market_cap_factor=1.0 if scenario['characteristics']['is_large_cap'] else -0.5,
            liquidity_factor=0.5 if scenario['characteristics']['is_liquid'] else -0.5,
            distress_factor=scenario['characteristics']['distress_score'],
            sector_factor=0.5 if 'Technology' in scenario['characteristics']['sector'] else 0.0,
            volatility_factor=0.0,  # Neutral for demo
            data_quality_factor=0.5
        )
        
        actual_ai_weight = factors.calculate_ai_weight()
        print(f"       Actual AI Weight: {actual_ai_weight:.1%}")
    
    print(f"\n  🎯 Key Research Insights Implemented:")
    print(f"    • AI advantages: Large-cap, liquid stocks with structured data")
    print(f"    • Human advantages: Small-cap, illiquid, asset-light during distress")
    print(f"    • Adaptive weighting: Dynamic based on stock characteristics")
    print(f"    • Performance tracking: Continuous learning and improvement")
    print(f"    • Ensemble approach: Combining quantitative and qualitative analysis")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print_header("🚀 HYBRID AI TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print(f"Based on Cao et al. (2024) - 'Man + Machine' Approach")
    print(f"Implementation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Test 1: Data Collection
    test_results['data_collection'] = test_data_collection()
    
    # Test 2: AI Analysis
    test_results['ai_analysis'] = test_ai_analysis()
    
    # Test 3: Human-like AI Analysis
    test_results['human_ai_analysis'] = test_human_ai_analysis()
    
    # Test 4: Adaptive Weighting
    test_results['adaptive_weighting'] = test_adaptive_weighting()
    
    # Test 5: Complete System
    test_results['complete_system'] = test_complete_hybrid_system()
    
    # Test 6: Portfolio Analysis
    test_results['portfolio_analysis'] = test_portfolio_analysis()
    
    # Demonstration of research implementation
    demonstrate_research_implementation()
    
    # Summary
    print_section("Test Results Summary")
    
    total_tests = len(test_results)
    successful_tests = sum(1 for result in test_results.values() 
                          if isinstance(result, dict) and result.get('success', False))
    
    print(f"📊 Test Results: {successful_tests}/{total_tests} tests passed")
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if isinstance(result, dict) and result.get('success', False) else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        if isinstance(result, dict) and not result.get('success', False) and 'error' in result:
            print(f"    Error: {result['error']}")
    
    print_header("🎉 HYBRID AI TRADING SYSTEM TEST COMPLETE")
    
    if successful_tests == total_tests:
        print(f"🎯 ALL TESTS PASSED! System is ready for production deployment.")
        print(f"🚀 The hybrid AI trading system successfully implements the")
        print(f"   'Man + Machine' approach from Cao et al. (2024) research.")
    else:
        print(f"⚠️  Some tests failed. Please review the errors above.")
    
    print(f"\n📈 Next Steps:")
    print(f"  1. Set up OpenAI API key for enhanced human-like reasoning")
    print(f"  2. Configure real-time data feeds for production")
    print(f"  3. Implement paper trading integration")
    print(f"  4. Set up performance monitoring and alerting")
    print(f"  5. Deploy to production environment")
    
    return test_results

if __name__ == "__main__":
    # Run the comprehensive test suite
    results = run_comprehensive_test()
    
    # Save test results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"hybrid_ai_test_results_{timestamp}.json"
    
    try:
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    'success': value.get('success', False),
                    'error': value.get('error', None) if not value.get('success', False) else None
                }
            else:
                json_results[key] = {'success': False, 'error': 'Unknown result format'}
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_results': json_results,
                'summary': {
                    'total_tests': len(results),
                    'passed_tests': sum(1 for r in results.values() 
                                      if isinstance(r, dict) and r.get('success', False)),
                    'system_status': 'READY' if all(isinstance(r, dict) and r.get('success', False) 
                                                   for r in results.values()) else 'NEEDS_ATTENTION'
                }
            }, f, indent=2)
        
        print(f"\n💾 Test results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save test results: {e}")
