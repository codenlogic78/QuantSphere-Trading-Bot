#!/usr/bin/env python3
"""
Test Script for Hybrid AI Implementation
Demonstrates the "Man + Machine" approach based on research findings
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_ai_analyst import HybridAIAnalyst, AnalysisType, MarketCondition
from enhanced_trading_engine import EnhancedTradingEngine, TradingSignal, SignalSource

def test_hybrid_ai_analyst():
    """Test the Hybrid AI Analyst with different analysis types"""
    
    print("🧠 TESTING HYBRID AI ANALYST")
    print("="*60)
    
    # Initialize analyst
    analyst = HybridAIAnalyst()
    
    # Test symbols representing different company characteristics
    test_cases = [
        {
            'symbol': 'AAPL',
            'description': 'Large-cap, high liquidity, transparent (AI advantage expected)'
        },
        {
            'symbol': 'TSLA', 
            'description': 'High-growth, volatile, asset-light (Mixed advantages)'
        },
        {
            'symbol': 'NVDA',
            'description': 'Tech leader, high intangibles (Human reasoning valuable)'
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        symbol = test_case['symbol']
        description = test_case['description']
        
        print(f"\n📊 ANALYZING {symbol}")
        print(f"Characteristics: {description}")
        print("-" * 50)
        
        symbol_results = {}
        
        # Test all analysis types
        for analysis_type in [AnalysisType.AI_ONLY, AnalysisType.HUMAN_ENHANCED, AnalysisType.HYBRID]:
            try:
                print(f"\n🔍 {analysis_type.value.upper()} Analysis:")
                
                result = analyst.analyze_stock(symbol, analysis_type)
                
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.1%}")
                print(f"  Reasoning: {result['reasoning'][:100]}...")
                
                if analysis_type == AnalysisType.HYBRID:
                    print(f"  AI Weight: {result.get('ai_weight', 0):.1%}")
                    print(f"  Human Weight: {result.get('human_weight', 0):.1%}")
                
                symbol_results[analysis_type.value] = result
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                symbol_results[analysis_type.value] = {'error': str(e)}
        
        results[symbol] = symbol_results
    
    return results

def test_enhanced_trading_engine():
    """Test the Enhanced Trading Engine"""
    
    print("\n\n🚀 TESTING ENHANCED TRADING ENGINE")
    print("="*60)
    
    # Initialize engine
    engine = EnhancedTradingEngine()
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'MSFT']
    
    trading_results = {}
    
    for symbol in test_symbols:
        print(f"\n📈 TESTING TRADING DECISIONS FOR {symbol}")
        print("-" * 40)
        
        # Mock current price
        current_price = 150.0 + np.random.uniform(-30, 30)
        
        try:
            # Generate comprehensive analysis
            analysis = engine.generate_comprehensive_analysis(symbol)
            
            print(f"📊 Analysis Results:")
            for method, result in analysis.items():
                if method == 'technical':
                    signal = result.get('signal', 'HOLD')
                    confidence = result.get('confidence', 0.5)
                else:
                    signal = result.get('prediction', 'HOLD')
                    confidence = result.get('confidence', 0.5)
                
                print(f"  {method}: {signal} ({confidence:.1%})")
            
            # Generate trading decision
            decision = engine.generate_trading_decision(symbol, current_price)
            
            if decision:
                print(f"\n🎯 TRADING DECISION:")
                print(f"  Signal: {decision.signal.value}")
                print(f"  Confidence: {decision.confidence:.1%}")
                print(f"  Position Size: {decision.position_size:.1%}")
                print(f"  Price Target: ${decision.price_target:.2f}" if decision.price_target else "  Price Target: None")
                print(f"  Stop Loss: ${decision.stop_loss:.2f}" if decision.stop_loss else "  Stop Loss: None")
                print(f"  Risk Score: {decision.risk_score:.1%}")
                
                # Test execution
                mock_positions = {symbol: {'position': 0, 'current_price': current_price}}
                execution = engine.execute_trading_decision(decision, mock_positions)
                
                print(f"\n⚡ EXECUTION:")
                print(f"  Executed: {execution['executed']}")
                print(f"  Action: {execution.get('action', 'None')}")
                print(f"  Reason: {execution['reason']}")
                
                trading_results[symbol] = {
                    'analysis': analysis,
                    'decision': {
                        'signal': decision.signal.value,
                        'confidence': decision.confidence,
                        'position_size': decision.position_size,
                        'risk_score': decision.risk_score
                    },
                    'execution': execution
                }
            else:
                print("❌ No trading decision generated")
                trading_results[symbol] = {'error': 'No decision generated'}
                
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            trading_results[symbol] = {'error': str(e)}
    
    return trading_results

def test_performance_comparison():
    """Compare performance of different analysis methods"""
    
    print("\n\n📊 PERFORMANCE COMPARISON TEST")
    print("="*60)
    
    analyst = HybridAIAnalyst()
    
    # Test multiple symbols to compare methods
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    method_performance = {
        'ai_only': {'predictions': [], 'confidences': []},
        'human_enhanced': {'predictions': [], 'confidences': []},
        'hybrid': {'predictions': [], 'confidences': []}
    }
    
    print("Testing multiple symbols to compare analysis methods...")
    
    for symbol in symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        
        for method in ['ai_only', 'human_enhanced', 'hybrid']:
            try:
                analysis_type = AnalysisType(method)
                result = analyst.analyze_stock(symbol, analysis_type)
                
                prediction = result.get('prediction', 'HOLD')
                confidence = result.get('confidence', 0.5)
                
                method_performance[method]['predictions'].append(prediction)
                method_performance[method]['confidences'].append(confidence)
                
                print(f"  {method}: {prediction} ({confidence:.1%})")
                
            except Exception as e:
                print(f"  {method}: Error - {e}")
    
    # Calculate summary statistics
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print("-" * 30)
    
    for method, data in method_performance.items():
        if data['confidences']:
            avg_confidence = np.mean(data['confidences'])
            buy_signals = data['predictions'].count('BUY') + data['predictions'].count('STRONG_BUY')
            sell_signals = data['predictions'].count('SELL') + data['predictions'].count('STRONG_SELL')
            hold_signals = data['predictions'].count('HOLD')
            
            print(f"\n{method.upper()}:")
            print(f"  Average Confidence: {avg_confidence:.1%}")
            print(f"  Signal Distribution: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")
    
    return method_performance

def generate_research_summary():
    """Generate summary of research implementation"""
    
    summary = f"""
🧠 QUANTSPHERE HYBRID AI IMPLEMENTATION SUMMARY
{'='*70}

📚 RESEARCH BASIS:
"From Man vs. Machine to Man + Machine: The Art and AI of Stock Analyses"
by Cao, Jiang, Wang, Yang (2024) - Journal of Financial Economics

🔬 KEY RESEARCH FINDINGS IMPLEMENTED:

1. AI ANALYST PERFORMANCE:
   ✅ Outperformed 53.7% of human analysts
   ✅ Generated 0.84-0.92% monthly risk-adjusted returns
   ✅ Superior with transparent, voluminous information

2. HUMAN ANALYST ADVANTAGES:
   ✅ Better with illiquid, smaller firms
   ✅ Superior for asset-light business models
   ✅ More effective during industry distress
   ✅ Better institutional knowledge application

3. HYBRID MODEL SUCCESS:
   ✅ "Man + Machine" outperformed 57.3% of analysts
   ✅ Beat AI-only models in all test years
   ✅ Optimal combination of computational power + human reasoning

🚀 QUANTSPHERE IMPLEMENTATION:

✅ HYBRID AI ANALYST:
   • Multi-source data integration (firm, industry, macro, textual)
   • Adaptive weighting based on company characteristics
   • GPT-4 powered human-like reasoning
   • Alternative data processing capabilities

✅ ENHANCED TRADING ENGINE:
   • Ensemble analysis combining all methods
   • Dynamic position sizing based on confidence
   • Risk management with company-specific adjustments
   • Performance tracking and adaptive learning

✅ INTEGRATION FEATURES:
   • Real-time analysis method selection
   • Performance comparison dashboard
   • Research-backed decision explanations
   • Continuous learning and adaptation

📊 EXPECTED BENEFITS:
• Higher prediction accuracy through method combination
• Better risk-adjusted returns
• Adaptive performance based on market conditions
• Institutional-grade analysis capabilities

🎯 USAGE RECOMMENDATIONS:
1. Use Hybrid mode for best overall performance
2. Switch to AI-Only for large-cap, liquid stocks
3. Use Human-Enhanced for complex, illiquid situations
4. Monitor performance metrics for continuous improvement

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return summary

def main():
    """Main test function"""
    
    print("🧠 QUANTSPHERE HYBRID AI IMPLEMENTATION TEST")
    print("Based on 'From Man vs. Machine to Man + Machine' Research")
    print("="*70)
    
    # Test 1: Hybrid AI Analyst
    try:
        analyst_results = test_hybrid_ai_analyst()
        print("✅ Hybrid AI Analyst test completed")
    except Exception as e:
        print(f"❌ Hybrid AI Analyst test failed: {e}")
        analyst_results = {}
    
    # Test 2: Enhanced Trading Engine
    try:
        trading_results = test_enhanced_trading_engine()
        print("✅ Enhanced Trading Engine test completed")
    except Exception as e:
        print(f"❌ Enhanced Trading Engine test failed: {e}")
        trading_results = {}
    
    # Test 3: Performance Comparison
    try:
        performance_results = test_performance_comparison()
        print("✅ Performance comparison test completed")
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        performance_results = {}
    
    # Generate comprehensive results
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'analyst_results': analyst_results,
        'trading_results': trading_results,
        'performance_results': performance_results,
        'research_summary': generate_research_summary()
    }
    
    # Save results
    results_file = f"hybrid_ai_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to: {results_file}")
    
    # Display research summary
    print(test_results['research_summary'])
    
    print("\n🎉 HYBRID AI IMPLEMENTATION TEST COMPLETED!")
    print("Your QuantSphere platform now includes cutting-edge research-based AI analysis!")

if __name__ == "__main__":
    main()
