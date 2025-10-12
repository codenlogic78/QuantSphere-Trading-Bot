"""
Demo Script: QuantSphere Hybrid AI Integration
Shows how to use the new hybrid AI features in your QuantSphere platform
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from hybrid_ai_integration import QuantSphereHybridAI
    print("✅ Hybrid AI integration loaded successfully!")
except ImportError as e:
    print(f"❌ Error loading Hybrid AI: {e}")
    exit(1)

def demo_single_stock_analysis():
    """Demo: Analyze a single stock with hybrid AI"""
    print(f"\n{'='*60}")
    print(f"🔍 DEMO: Single Stock Hybrid AI Analysis")
    print(f"{'='*60}")
    
    # Initialize hybrid AI
    config = {
        'openai_api_key': None,  # Using mock responses for demo
        'performance_file': 'demo_hybrid_performance.json'
    }
    
    hybrid_ai = QuantSphereHybridAI(config)
    
    # Analyze AAPL
    symbol = 'AAPL'
    print(f"🧠 Analyzing {symbol} with Hybrid AI...")
    
    try:
        analysis = hybrid_ai.get_enhanced_analysis(symbol)
        
        print(f"\n📊 RESULTS FOR {symbol}:")
        print(f"Hybrid Recommendation: {analysis['hybrid_recommendation']}")
        print(f"Overall Confidence: {analysis['confidence']:.2f}")
        print(f"Expected Alpha: {analysis['expected_alpha']:.2%}")
        print(f"Risk Score: {analysis['risk_score']:.2f}")
        
        print(f"\n🤖 AI Component:")
        print(f"  Recommendation: {analysis['ai_analysis']['recommendation']}")
        print(f"  Technical Signal: {analysis['ai_analysis']['technical_signal']}")
        print(f"  Price Prediction: ${analysis['ai_analysis']['price_prediction']:.2f}")
        
        print(f"\n🧠 Human-like Component:")
        print(f"  Recommendation: {analysis['human_analysis']['recommendation']}")
        print(f"  Company Assessment: {analysis['human_analysis']['company_assessment'][:100]}...")
        
        print(f"\n⚖️ Weighting:")
        print(f"  AI Weight: {analysis['weighting']['ai_weight']:.1%}")
        print(f"  Human Weight: {analysis['weighting']['human_weight']:.1%}")
        print(f"  Rationale: {analysis['weighting']['rationale'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return False

def demo_portfolio_recommendations():
    """Demo: Get portfolio recommendations"""
    print(f"\n{'='*60}")
    print(f"📊 DEMO: Portfolio Recommendations")
    print(f"{'='*60}")
    
    config = {'performance_file': 'demo_portfolio_performance.json'}
    hybrid_ai = QuantSphereHybridAI(config)
    
    # Sample portfolio
    portfolio = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    print(f"📈 Analyzing portfolio: {', '.join(portfolio)}")
    
    try:
        recommendations = hybrid_ai.get_portfolio_recommendations(portfolio, top_n=3)
        
        print(f"\n🏆 TOP RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['symbol']}: {rec['recommendation']} "
                  f"(confidence: {rec['confidence']:.2f}, "
                  f"alpha: {rec['expected_alpha']:.2%}, "
                  f"AI weight: {rec['ai_weight']:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        return False

def demo_market_insights():
    """Demo: Get market insights"""
    print(f"\n{'='*60}")
    print(f"🌍 DEMO: Market Insights")
    print(f"{'='*60}")
    
    config = {'performance_file': 'demo_insights_performance.json'}
    hybrid_ai = QuantSphereHybridAI(config)
    
    # Sample stocks for market analysis
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print(f"🔍 Analyzing market sentiment for: {', '.join(symbols)}")
    
    try:
        insights = hybrid_ai.get_market_insights(symbols)
        
        print(f"\n🤖 AI SENTIMENT:")
        print(f"  Bullish: {insights['ai_sentiment']['bullish_signals']}")
        print(f"  Bearish: {insights['ai_sentiment']['bearish_signals']}")
        print(f"  Overall Bias: {insights['ai_sentiment']['overall_bias']}")
        
        print(f"\n🧠 HUMAN-LIKE SENTIMENT:")
        print(f"  Bullish: {insights['human_sentiment']['bullish_signals']}")
        print(f"  Bearish: {insights['human_sentiment']['bearish_signals']}")
        print(f"  Overall Bias: {insights['human_sentiment']['overall_bias']}")
        
        print(f"\n💡 KEY OPPORTUNITIES:")
        for opp in insights['key_opportunities'][:3]:
            print(f"  • {opp}")
        
        print(f"\n⚠️ KEY RISKS:")
        for risk in insights['key_risks'][:3]:
            print(f"  • {risk}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting market insights: {e}")
        return False

def demo_trading_signals():
    """Demo: Get trading signals for automated trading"""
    print(f"\n{'='*60}")
    print(f"📈 DEMO: Trading Signals for Automation")
    print(f"{'='*60}")
    
    config = {'performance_file': 'demo_trading_performance.json'}
    hybrid_ai = QuantSphereHybridAI(config)
    
    # Sample symbols for trading
    symbols = ['AAPL', 'MSFT']
    print(f"⚡ Generating trading signals for: {', '.join(symbols)}")
    
    try:
        signals = hybrid_ai.get_trading_signals(symbols)
        
        print(f"\n📊 TRADING SIGNALS:")
        for symbol, signal in signals.items():
            print(f"\n{symbol}:")
            print(f"  Signal: {signal['signal']}")
            print(f"  Strength: {signal['strength']:.2f}")
            print(f"  Confidence: {signal['confidence']:.2f}")
            print(f"  Position Size: {signal['position_size']:.1%}")
            print(f"  Expected Return: {signal['expected_return']:.2%}")
            print(f"  Risk Level: {signal['risk_level']}")
            print(f"  AI Weight: {signal['metadata']['ai_weight']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting trading signals: {e}")
        return False

def demo_gui_integration():
    """Demo: Show how to integrate with QuantSphere GUI"""
    print(f"\n{'='*60}")
    print(f"🖥️ DEMO: GUI Integration Examples")
    print(f"{'='*60}")
    
    print(f"""
🎯 HOW TO USE HYBRID AI IN YOUR QUANTSPHERE GUI:

1. 📊 ANALYZE SELECTED STOCK:
   - Select a stock in your QuantSphere table
   - Click "🔍 Analyze Selected Stock" button
   - View comprehensive hybrid AI analysis in popup window

2. 📈 PORTFOLIO RECOMMENDATIONS:
   - Click "📊 Portfolio Recommendations" button
   - See top-ranked stocks based on hybrid AI analysis
   - View confidence scores, expected alpha, and AI weighting

3. 🌍 MARKET INSIGHTS:
   - Click "🌍 Market Insights" button
   - Get market sentiment from both AI and human-like analysis
   - See key opportunities and risks identified

4. 🤖 ENHANCED CHAT:
   - Your existing AI chat now has access to hybrid analysis
   - Ask questions about specific stocks for enhanced insights
   - Get research-backed explanations for recommendations

5. ⚡ AUTOMATED TRADING:
   - Hybrid AI signals can enhance your existing trading logic
   - Higher confidence recommendations get priority
   - Adaptive weighting based on stock characteristics
    """)

def run_complete_demo():
    """Run the complete demo suite"""
    print(f"\n{'='*80}")
    print(f"🚀 QUANTSPHERE HYBRID AI INTEGRATION - COMPLETE DEMO")
    print(f"{'='*80}")
    print(f"Based on Cao et al. (2024) 'Man + Machine' Research")
    print(f"Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all demos
    results.append(("Single Stock Analysis", demo_single_stock_analysis()))
    results.append(("Portfolio Recommendations", demo_portfolio_recommendations()))
    results.append(("Market Insights", demo_market_insights()))
    results.append(("Trading Signals", demo_trading_signals()))
    
    # Show GUI integration info
    demo_gui_integration()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 DEMO RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for demo_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} demos successful")
    
    if passed == total:
        print(f"\n🎉 ALL DEMOS PASSED!")
        print(f"✅ Hybrid AI integration is working perfectly!")
        print(f"🚀 Your QuantSphere platform now has cutting-edge AI capabilities!")
    else:
        print(f"\n⚠️ Some demos had issues - check the error messages above")
    
    print(f"\n📚 NEXT STEPS:")
    print(f"1. Run your QuantSphere GUI: python bot.py")
    print(f"2. Add some stocks to your portfolio")
    print(f"3. Try the new Hybrid AI buttons!")
    print(f"4. Set up OpenAI API key for enhanced human-like analysis")
    print(f"5. Monitor performance and enjoy the improved trading insights!")

if __name__ == "__main__":
    run_complete_demo()
