#!/usr/bin/env python3
"""
QuantSphere Hybrid AI Analyst
Implementation based on "From Man vs. Machine to Man + Machine" research
Combines AI computational power with human-like reasoning capabilities
"""

import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import openai
import os
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class AnalysisType(Enum):
    AI_ONLY = "ai_only"
    HUMAN_ENHANCED = "human_enhanced"
    HYBRID = "hybrid"

class MarketCondition(Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    DISTRESSED = "distressed"
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"

@dataclass
class CompanyProfile:
    """Company characteristics that influence AI vs Human advantage"""
    symbol: str
    market_cap: float
    liquidity_score: float  # Higher = more liquid
    intangible_ratio: float  # Higher = more asset-light
    industry: str
    information_transparency: float  # Higher = more transparent
    
class HybridAIAnalyst:
    """
    Hybrid AI Analyst implementing research findings from Cao et al. (2024)
    Combines machine learning with human-like reasoning patterns
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.analysis_history = []
        self.performance_metrics = {
            'ai_only': {'correct_predictions': 0, 'total_predictions': 0},
            'human_enhanced': {'correct_predictions': 0, 'total_predictions': 0},
            'hybrid': {'correct_predictions': 0, 'total_predictions': 0}
        }
        
        # Research-based thresholds
        self.liquidity_threshold = 0.5  # Below this, humans have advantage
        self.intangible_threshold = 0.3  # Above this, humans have advantage
        self.information_volume_threshold = 100  # Above this, AI has advantage
        
    def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Get company characteristics to determine AI vs Human advantage"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Calculate liquidity score (simplified)
            avg_volume = info.get('averageVolume', 0)
            market_cap = info.get('marketCap', 0)
            liquidity_score = min(1.0, (avg_volume * 252) / max(market_cap, 1)) if market_cap > 0 else 0
            
            # Calculate intangible ratio
            total_assets = info.get('totalAssets', 1)
            tangible_assets = info.get('tangibleBookValue', total_assets)
            intangible_ratio = max(0, (total_assets - tangible_assets) / total_assets) if total_assets > 0 else 0
            
            # Information transparency (simplified - based on company size and reporting)
            transparency = min(1.0, market_cap / 10_000_000_000) if market_cap > 0 else 0  # Larger = more transparent
            
            return CompanyProfile(
                symbol=symbol,
                market_cap=market_cap,
                liquidity_score=liquidity_score,
                intangible_ratio=intangible_ratio,
                industry=info.get('industry', 'Unknown'),
                information_transparency=transparency
            )
            
        except Exception as e:
            print(f"Error getting company profile for {symbol}: {e}")
            # Return default profile
            return CompanyProfile(
                symbol=symbol,
                market_cap=1_000_000_000,
                liquidity_score=0.5,
                intangible_ratio=0.2,
                industry='Unknown',
                information_transparency=0.5
            )
    
    def assess_market_condition(self, symbol: str) -> MarketCondition:
        """Assess current market condition for the industry/stock"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if len(hist) < 30:
                return MarketCondition.NORMAL
            
            # Calculate volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate trend
            recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 4  # Annualized
            
            # Classify market condition
            if volatility > 0.4:  # High volatility
                return MarketCondition.VOLATILE
            elif recent_return < -0.2:  # Down more than 20% annualized
                return MarketCondition.BEAR_MARKET
            elif recent_return > 0.3:  # Up more than 30% annualized
                return MarketCondition.BULL_MARKET
            else:
                return MarketCondition.NORMAL
                
        except Exception as e:
            print(f"Error assessing market condition for {symbol}: {e}")
            return MarketCondition.NORMAL
    
    def collect_multi_source_data(self, symbol: str) -> Dict[str, Any]:
        """Collect data from multiple sources as per research methodology"""
        data = {}
        
        try:
            ticker = yf.Ticker(symbol)
            
            # 1. Firm-level data
            info = ticker.info
            hist = ticker.history(period="1y")
            
            data['firm_level'] = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margins': info.get('profitMargins', 0),
                'current_ratio': info.get('currentRatio', 0),
                'price_performance_1y': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) if len(hist) > 0 else 0
            }
            
            # 2. Industry-level data (simplified)
            industry = info.get('industry', 'Unknown')
            sector = info.get('sector', 'Unknown')
            
            data['industry_level'] = {
                'industry': industry,
                'sector': sector,
                'industry_pe': info.get('trailingPE', 15),  # Simplified
                'sector_performance': 0  # Would need sector index data
            }
            
            # 3. Macro-economic indicators (simplified)
            data['macro_economic'] = {
                'market_trend': self.assess_market_condition(symbol).value,
                'volatility_regime': 'normal',  # Would need VIX data
                'interest_rate_environment': 'neutral'  # Would need Fed data
            }
            
            # 4. Textual information (news sentiment)
            data['textual'] = self.get_news_sentiment(symbol)
            
            # 5. Alternative data (mock - would need real sources)
            data['alternative'] = {
                'social_sentiment': np.random.uniform(-1, 1),  # Mock social media sentiment
                'satellite_data': np.random.uniform(0, 1),     # Mock satellite/foot traffic data
                'web_search_trends': np.random.uniform(0, 1)   # Mock search trend data
            }
            
        except Exception as e:
            print(f"Error collecting multi-source data for {symbol}: {e}")
            data = self.get_default_data(symbol)
        
        return data
    
    def get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get news sentiment analysis (simplified implementation)"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return {'sentiment_score': 0.0, 'news_volume': 0}
            
            sentiments = []
            for article in news[:10]:  # Analyze last 10 articles
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = f"{title} {summary}"
                
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            
            return {
                'sentiment_score': np.mean(sentiments) if sentiments else 0.0,
                'news_volume': len(news)
            }
            
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {e}")
            return {'sentiment_score': 0.0, 'news_volume': 0}
    
    def get_default_data(self, symbol: str) -> Dict[str, Any]:
        """Return default data structure when real data unavailable"""
        return {
            'firm_level': {
                'market_cap': 1_000_000_000,
                'pe_ratio': 15,
                'pb_ratio': 2,
                'debt_to_equity': 0.5,
                'roe': 0.15,
                'revenue_growth': 0.05,
                'profit_margins': 0.1,
                'current_ratio': 1.5,
                'price_performance_1y': 0.1
            },
            'industry_level': {
                'industry': 'Technology',
                'sector': 'Technology',
                'industry_pe': 20,
                'sector_performance': 0.12
            },
            'macro_economic': {
                'market_trend': 'normal',
                'volatility_regime': 'normal',
                'interest_rate_environment': 'neutral'
            },
            'textual': {
                'sentiment_score': 0.0,
                'news_volume': 5
            },
            'alternative': {
                'social_sentiment': 0.0,
                'satellite_data': 0.5,
                'web_search_trends': 0.5
            }
        }
    
    def ai_only_analysis(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pure AI analysis using machine learning approach"""
        try:
            # Extract numerical features for ML model
            features = []
            
            # Firm-level features
            firm_data = data['firm_level']
            features.extend([
                firm_data.get('pe_ratio', 15),
                firm_data.get('pb_ratio', 2),
                firm_data.get('debt_to_equity', 0.5),
                firm_data.get('roe', 0.15),
                firm_data.get('revenue_growth', 0.05),
                firm_data.get('profit_margins', 0.1),
                firm_data.get('current_ratio', 1.5),
                firm_data.get('price_performance_1y', 0.1)
            ])
            
            # Textual features
            textual_data = data['textual']
            features.extend([
                textual_data.get('sentiment_score', 0),
                textual_data.get('news_volume', 5) / 100  # Normalize
            ])
            
            # Alternative data features
            alt_data = data['alternative']
            features.extend([
                alt_data.get('social_sentiment', 0),
                alt_data.get('satellite_data', 0.5),
                alt_data.get('web_search_trends', 0.5)
            ])
            
            # Simple ML-like scoring (in real implementation, use trained models)
            feature_weights = [0.1, 0.08, -0.05, 0.15, 0.12, 0.1, 0.05, 0.2, 0.1, 0.02, 0.05, 0.03, 0.02]
            
            if len(features) != len(feature_weights):
                feature_weights = feature_weights[:len(features)]
            
            ai_score = sum(f * w for f, w in zip(features, feature_weights))
            
            # Convert to prediction
            if ai_score > 0.1:
                prediction = "BUY"
                confidence = min(0.9, 0.5 + ai_score)
            elif ai_score < -0.1:
                prediction = "SELL"
                confidence = min(0.9, 0.5 + abs(ai_score))
            else:
                prediction = "HOLD"
                confidence = 0.6
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'ai_score': ai_score,
                'reasoning': f"AI model computed score of {ai_score:.3f} based on quantitative features",
                'method': 'machine_learning'
            }
            
        except Exception as e:
            return {
                'prediction': 'HOLD',
                'confidence': 0.5,
                'ai_score': 0,
                'reasoning': f"AI analysis error: {str(e)}",
                'method': 'fallback'
            }
    
    def human_enhanced_analysis(self, symbol: str, data: Dict[str, Any], company_profile: CompanyProfile) -> Dict[str, Any]:
        """Human-enhanced analysis using GPT for reasoning"""
        try:
            if not self.openai_api_key:
                return self.fallback_human_analysis(symbol, data, company_profile)
            
            # Create human-like reasoning prompt
            prompt = f"""
            You are an experienced stock analyst with deep institutional knowledge. Analyze {symbol} using human judgment and reasoning.
            
            Company Profile:
            - Market Cap: ${company_profile.market_cap:,.0f}
            - Liquidity Score: {company_profile.liquidity_score:.2f}
            - Intangible Assets Ratio: {company_profile.intangible_ratio:.2f}
            - Industry: {company_profile.industry}
            - Information Transparency: {company_profile.information_transparency:.2f}
            
            Financial Data:
            - P/E Ratio: {data['firm_level'].get('pe_ratio', 'N/A')}
            - P/B Ratio: {data['firm_level'].get('pb_ratio', 'N/A')}
            - ROE: {data['firm_level'].get('roe', 'N/A')}
            - Revenue Growth: {data['firm_level'].get('revenue_growth', 'N/A')}
            - Debt/Equity: {data['firm_level'].get('debt_to_equity', 'N/A')}
            
            Market Context:
            - News Sentiment: {data['textual'].get('sentiment_score', 0):.2f}
            - Market Trend: {data['macro_economic'].get('market_trend', 'normal')}
            
            Focus on:
            1. Qualitative factors that machines miss
            2. Industry-specific knowledge and context
            3. Management quality and strategic positioning
            4. Competitive moats and intangible assets
            5. Market timing and cyclical considerations
            
            Provide your analysis in this JSON format:
            {{
                "prediction": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "detailed explanation",
                "key_factors": ["factor1", "factor2", "factor3"],
                "risks": ["risk1", "risk2"],
                "opportunities": ["opp1", "opp2"]
            }}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                api_key=self.openai_api_key,
                temperature=0.3
            )
            
            # Parse response
            try:
                result = json.loads(response['choices'][0]['message']['content'])
                result['method'] = 'human_reasoning'
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                content = response['choices'][0]['message']['content']
                return {
                    'prediction': 'HOLD',
                    'confidence': 0.6,
                    'reasoning': content,
                    'method': 'human_reasoning_text',
                    'key_factors': [],
                    'risks': [],
                    'opportunities': []
                }
                
        except Exception as e:
            return self.fallback_human_analysis(symbol, data, company_profile)
    
    def fallback_human_analysis(self, symbol: str, data: Dict[str, Any], company_profile: CompanyProfile) -> Dict[str, Any]:
        """Fallback human-like analysis when OpenAI is unavailable"""
        reasoning_factors = []
        score = 0
        
        # Analyze based on research findings about human advantages
        
        # 1. Liquidity consideration (humans better with illiquid stocks)
        if company_profile.liquidity_score < self.liquidity_threshold:
            reasoning_factors.append("Low liquidity stock - requires human judgment")
            score += 0.1
        
        # 2. Intangible assets (humans better with asset-light companies)
        if company_profile.intangible_ratio > self.intangible_threshold:
            reasoning_factors.append("High intangible assets - human analysis advantage")
            score += 0.15
        
        # 3. Industry distress (humans better during distress)
        market_condition = self.assess_market_condition(symbol)
        if market_condition in [MarketCondition.DISTRESSED, MarketCondition.BEAR_MARKET]:
            reasoning_factors.append("Market distress - human experience valuable")
            score += 0.2
        
        # 4. Qualitative factors
        sentiment = data['textual'].get('sentiment_score', 0)
        if abs(sentiment) > 0.3:  # Strong sentiment
            reasoning_factors.append(f"Strong sentiment ({sentiment:.2f}) requires interpretation")
            score += sentiment * 0.1
        
        # 5. Financial health assessment
        roe = data['firm_level'].get('roe', 0.15)
        debt_ratio = data['firm_level'].get('debt_to_equity', 0.5)
        
        if roe > 0.2 and debt_ratio < 0.3:
            reasoning_factors.append("Strong fundamentals with low debt")
            score += 0.1
        elif roe < 0.05 or debt_ratio > 1.0:
            reasoning_factors.append("Weak fundamentals or high debt")
            score -= 0.15
        
        # Generate prediction
        if score > 0.15:
            prediction = "BUY"
            confidence = min(0.85, 0.6 + score)
        elif score < -0.1:
            prediction = "SELL"
            confidence = min(0.85, 0.6 + abs(score))
        else:
            prediction = "HOLD"
            confidence = 0.65
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': "; ".join(reasoning_factors) if reasoning_factors else "Balanced analysis suggests neutral stance",
            'method': 'human_heuristics',
            'key_factors': reasoning_factors,
            'risks': ["Market volatility", "Sector rotation"],
            'opportunities': ["Value realization", "Market recovery"]
        }
    
    def hybrid_analysis(self, symbol: str, ai_result: Dict[str, Any], human_result: Dict[str, Any], 
                       company_profile: CompanyProfile) -> Dict[str, Any]:
        """Combine AI and human analysis based on research findings"""
        
        # Determine optimal weighting based on company characteristics
        ai_weight = self.calculate_ai_weight(company_profile)
        human_weight = 1 - ai_weight
        
        # Combine predictions with weighted approach
        prediction_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # AI contribution
        ai_pred = ai_result['prediction']
        ai_conf = ai_result['confidence']
        prediction_scores[ai_pred] += ai_weight * ai_conf
        
        # Human contribution
        human_pred = human_result['prediction']
        human_conf = human_result['confidence']
        prediction_scores[human_pred] += human_weight * human_conf
        
        # Final prediction
        final_prediction = max(prediction_scores, key=prediction_scores.get)
        final_confidence = prediction_scores[final_prediction]
        
        # Calculate ensemble score
        ai_score = ai_result.get('ai_score', 0)
        human_score = (1 if human_pred == 'BUY' else -1 if human_pred == 'SELL' else 0) * human_conf
        ensemble_score = ai_weight * ai_score + human_weight * human_score
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'ensemble_score': ensemble_score,
            'ai_weight': ai_weight,
            'human_weight': human_weight,
            'ai_contribution': {
                'prediction': ai_pred,
                'confidence': ai_conf,
                'reasoning': ai_result.get('reasoning', '')
            },
            'human_contribution': {
                'prediction': human_pred,
                'confidence': human_conf,
                'reasoning': human_result.get('reasoning', ''),
                'key_factors': human_result.get('key_factors', [])
            },
            'method': 'hybrid_ensemble',
            'reasoning': f"Hybrid analysis (AI: {ai_weight:.1%}, Human: {human_weight:.1%}) - {final_prediction} with {final_confidence:.1%} confidence"
        }
    
    def calculate_ai_weight(self, company_profile: CompanyProfile) -> float:
        """Calculate optimal AI weight based on research findings"""
        ai_weight = 0.5  # Base weight
        
        # AI advantages: high liquidity, transparent information, large companies
        if company_profile.liquidity_score > self.liquidity_threshold:
            ai_weight += 0.2
        
        if company_profile.information_transparency > 0.7:
            ai_weight += 0.15
        
        if company_profile.market_cap > 10_000_000_000:  # Large cap
            ai_weight += 0.1
        
        # Human advantages: low liquidity, high intangibles, small companies
        if company_profile.liquidity_score < self.liquidity_threshold:
            ai_weight -= 0.2
        
        if company_profile.intangible_ratio > self.intangible_threshold:
            ai_weight -= 0.15
        
        if company_profile.market_cap < 1_000_000_000:  # Small cap
            ai_weight -= 0.1
        
        # Market condition adjustment
        # During distress, humans have advantage (research finding)
        market_condition = self.assess_market_condition(company_profile.symbol)
        if market_condition in [MarketCondition.DISTRESSED, MarketCondition.VOLATILE]:
            ai_weight -= 0.15
        
        # Ensure weight stays in valid range
        return max(0.1, min(0.9, ai_weight))
    
    def analyze_stock(self, symbol: str, analysis_type: AnalysisType = AnalysisType.HYBRID) -> Dict[str, Any]:
        """Main analysis function implementing research methodology"""
        
        print(f"\n🔍 Analyzing {symbol} using {analysis_type.value} approach...")
        
        # Step 1: Get company profile
        company_profile = self.get_company_profile(symbol)
        
        # Step 2: Collect multi-source data
        data = self.collect_multi_source_data(symbol)
        
        # Step 3: Perform analysis based on type
        if analysis_type == AnalysisType.AI_ONLY:
            result = self.ai_only_analysis(symbol, data)
            
        elif analysis_type == AnalysisType.HUMAN_ENHANCED:
            result = self.human_enhanced_analysis(symbol, data, company_profile)
            
        else:  # HYBRID
            ai_result = self.ai_only_analysis(symbol, data)
            human_result = self.human_enhanced_analysis(symbol, data, company_profile)
            result = self.hybrid_analysis(symbol, ai_result, human_result, company_profile)
        
        # Add metadata
        result.update({
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type.value,
            'company_profile': {
                'market_cap': company_profile.market_cap,
                'liquidity_score': company_profile.liquidity_score,
                'intangible_ratio': company_profile.intangible_ratio,
                'industry': company_profile.industry
            }
        })
        
        # Store for performance tracking
        self.analysis_history.append(result)
        
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of different analysis types"""
        summary = {
            'total_analyses': len(self.analysis_history),
            'by_type': {},
            'recent_analyses': self.analysis_history[-10:] if self.analysis_history else []
        }
        
        # Group by analysis type
        for analysis in self.analysis_history:
            analysis_type = analysis.get('analysis_type', 'unknown')
            if analysis_type not in summary['by_type']:
                summary['by_type'][analysis_type] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'predictions': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                }
            
            summary['by_type'][analysis_type]['count'] += 1
            summary['by_type'][analysis_type]['avg_confidence'] += analysis.get('confidence', 0)
            pred = analysis.get('prediction', 'HOLD')
            summary['by_type'][analysis_type]['predictions'][pred] += 1
        
        # Calculate averages
        for analysis_type in summary['by_type']:
            count = summary['by_type'][analysis_type]['count']
            if count > 0:
                summary['by_type'][analysis_type]['avg_confidence'] /= count
        
        return summary

# Test the Hybrid AI Analyst
if __name__ == "__main__":
    print("=== QUANTSPHERE HYBRID AI ANALYST TEST ===")
    
    # Initialize analyst
    analyst = HybridAIAnalyst()
    
    # Test symbols representing different company types
    test_symbols = ['AAPL', 'TSLA', 'NVDA']  # Large cap, high-growth, tech
    
    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"ANALYZING {symbol}")
        print('='*50)
        
        # Test all three analysis types
        for analysis_type in [AnalysisType.AI_ONLY, AnalysisType.HUMAN_ENHANCED, AnalysisType.HYBRID]:
            try:
                result = analyst.analyze_stock(symbol, analysis_type)
                
                print(f"\n{analysis_type.value.upper()} ANALYSIS:")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Reasoning: {result['reasoning']}")
                
                if analysis_type == AnalysisType.HYBRID:
                    print(f"AI Weight: {result['ai_weight']:.1%}")
                    print(f"Human Weight: {result['human_weight']:.1%}")
                
            except Exception as e:
                print(f"Error in {analysis_type.value} analysis: {e}")
    
    # Performance summary
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print('='*50)
    
    summary = analyst.get_performance_summary()
    print(json.dumps(summary, indent=2, default=str))
