"""
Human-like AI Analysis Engine for Hybrid Trading System
Based on Cao et al. (2024) - Human component of "Man + Machine" approach

This module provides human-like reasoning capabilities using GPT-4:
- Qualitative analysis of company fundamentals
- Market context and sentiment interpretation
- Creative problem-solving for unusual market conditions
- Intuitive pattern recognition in complex scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
import openai
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HumanAIAnalysisResult:
    """Result of human-like AI analysis for a stock"""
    symbol: str
    timestamp: datetime
    
    # Qualitative Analysis
    company_assessment: str
    management_quality: str
    competitive_position: str
    industry_outlook: str
    
    # Market Context
    market_sentiment_interpretation: str
    macro_economic_impact: str
    sector_rotation_analysis: str
    
    # Creative Insights
    unique_opportunities: List[str]
    hidden_risks: List[str]
    contrarian_viewpoints: List[str]
    
    # Human-like Recommendation
    human_recommendation: str  # 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    human_confidence: float  # 0-1
    reasoning_narrative: str
    key_catalysts: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'company_assessment': self.company_assessment,
            'management_quality': self.management_quality,
            'competitive_position': self.competitive_position,
            'industry_outlook': self.industry_outlook,
            'market_sentiment_interpretation': self.market_sentiment_interpretation,
            'macro_economic_impact': self.macro_economic_impact,
            'sector_rotation_analysis': self.sector_rotation_analysis,
            'unique_opportunities': self.unique_opportunities,
            'hidden_risks': self.hidden_risks,
            'contrarian_viewpoints': self.contrarian_viewpoints,
            'human_recommendation': self.human_recommendation,
            'human_confidence': self.human_confidence,
            'reasoning_narrative': self.reasoning_narrative,
            'key_catalysts': self.key_catalysts
        }

class GPTAnalyst:
    """GPT-4 powered human-like analyst"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
            self.gpt_available = True
        else:
            self.gpt_available = False
            logger.warning("OpenAI API key not found. Using mock analysis.")
    
    def analyze_with_gpt(self, prompt: str, max_tokens: int = 1000) -> str:
        """Get analysis from GPT-4"""
        try:
            if not self.gpt_available:
                return self._mock_gpt_response(prompt)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst with 20+ years of experience in equity research and investment management. Provide insightful, nuanced analysis that goes beyond what quantitative models can capture."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling GPT-4: {e}")
            return self._mock_gpt_response(prompt)
    
    def _mock_gpt_response(self, prompt: str) -> str:
        """Generate mock response when GPT-4 is not available"""
        if "company assessment" in prompt.lower():
            return "The company demonstrates solid fundamentals with consistent revenue growth and strong market positioning. Management has shown effective capital allocation strategies."
        elif "market sentiment" in prompt.lower():
            return "Current market sentiment appears cautiously optimistic with some concerns about macroeconomic headwinds. Institutional investors are showing selective interest in quality names."
        elif "opportunities" in prompt.lower():
            return "Potential expansion into emerging markets, strategic partnerships, and operational efficiency improvements present upside opportunities."
        elif "risks" in prompt.lower():
            return "Key risks include regulatory changes, competitive pressures, and potential economic slowdown impacts on consumer spending."
        else:
            return "Analysis indicates a balanced risk-reward profile with moderate upside potential in the current market environment."

class HumanAIAnalyst:
    """Main human-like AI analyst using GPT-4 for qualitative reasoning"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.gpt_analyst = GPTAnalyst(api_key)
        
    def analyze_stock(self, symbol: str, market_data: pd.DataFrame, 
                     fundamentals: Dict, sentiment: Dict, 
                     characteristics: Dict) -> HumanAIAnalysisResult:
        """Perform comprehensive human-like AI analysis of a stock"""
        try:
            logger.info(f"Starting human-like AI analysis for {symbol}")
            
            # Prepare context for GPT analysis
            context = self._prepare_analysis_context(symbol, market_data, fundamentals, sentiment, characteristics)
            
            # Company Assessment
            company_assessment = self._analyze_company_fundamentals(symbol, fundamentals, context)
            
            # Market Context Analysis
            market_context = self._analyze_market_context(symbol, sentiment, market_data, context)
            
            # Creative Insights
            creative_insights = self._generate_creative_insights(symbol, fundamentals, sentiment, context)
            
            # Generate overall recommendation
            recommendation, confidence, narrative, catalysts = self._generate_human_recommendation(
                symbol, company_assessment, market_context, creative_insights, context
            )
            
            result = HumanAIAnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                company_assessment=company_assessment['assessment'],
                management_quality=company_assessment['management'],
                competitive_position=company_assessment['competitive'],
                industry_outlook=company_assessment['industry'],
                market_sentiment_interpretation=market_context['sentiment'],
                macro_economic_impact=market_context['macro'],
                sector_rotation_analysis=market_context['sector'],
                unique_opportunities=creative_insights['opportunities'],
                hidden_risks=creative_insights['risks'],
                contrarian_viewpoints=creative_insights['contrarian'],
                human_recommendation=recommendation,
                human_confidence=confidence,
                reasoning_narrative=narrative,
                key_catalysts=catalysts
            )
            
            logger.info(f"Human-like AI analysis completed for {symbol}: {recommendation} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in human-like AI analysis for {symbol}: {e}")
            return self._create_default_result(symbol)
    
    def _prepare_analysis_context(self, symbol: str, market_data: pd.DataFrame, 
                                fundamentals: Dict, sentiment: Dict, 
                                characteristics: Dict) -> str:
        """Prepare comprehensive context for GPT analysis"""
        try:
            current_price = market_data['Close'].iloc[-1] if not market_data.empty else 0
            
            context = f"""
            STOCK ANALYSIS CONTEXT FOR {symbol}
            
            CURRENT MARKET DATA:
            - Current Price: ${current_price:.2f}
            - Market Cap: ${fundamentals.get('market_cap', 0):,.0f}
            - Sector: {fundamentals.get('sector', 'Unknown')}
            - Industry: {fundamentals.get('industry', 'Unknown')}
            
            FINANCIAL METRICS:
            - P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
            - P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}
            - Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}
            - ROE: {fundamentals.get('roe', 'N/A')}
            - Profit Margin: {fundamentals.get('profit_margin', 'N/A')}
            - Revenue Growth: {fundamentals.get('revenue_growth', 'N/A')}
            
            STOCK CHARACTERISTICS:
            - Large Cap: {characteristics.get('is_large_cap', False)}
            - High Liquidity: {characteristics.get('is_liquid', False)}
            - Asset Light: {characteristics.get('is_asset_light', False)}
            - Distress Score: {characteristics.get('distress_score', 0):.2f}
            
            SENTIMENT DATA:
            - News Sentiment: {sentiment.get('news_sentiment', 0):.2f}
            - Social Sentiment: {sentiment.get('social_sentiment', 0):.2f}
            - Analyst Sentiment: {sentiment.get('analyst_sentiment', 0):.2f}
            """
            
            return context
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            return f"Analysis context for {symbol} - Limited data available"
    
    def _analyze_company_fundamentals(self, symbol: str, fundamentals: Dict, context: str) -> Dict:
        """Analyze company fundamentals using human-like reasoning"""
        try:
            prompt = f"""
            {context}
            
            As an experienced equity analyst, provide a comprehensive assessment of {symbol}'s company fundamentals:
            
            1. COMPANY ASSESSMENT: Evaluate the overall financial health, business model strength, and competitive moat. Consider both quantitative metrics and qualitative factors.
            
            2. MANAGEMENT QUALITY: Assess management effectiveness based on capital allocation, strategic decisions, and track record (infer from financial performance).
            
            3. COMPETITIVE POSITION: Analyze the company's position within its industry, market share dynamics, and competitive advantages.
            
            4. INDUSTRY OUTLOOK: Evaluate the long-term prospects of the industry, growth drivers, and potential disruptions.
            
            Provide specific, actionable insights that go beyond what quantitative analysis would reveal.
            """
            
            response = self.gpt_analyst.analyze_with_gpt(prompt, max_tokens=800)
            
            # Parse response into components (simplified parsing)
            sections = response.split('\n\n')
            
            return {
                'assessment': sections[0] if len(sections) > 0 else "Solid fundamental profile with balanced risk-reward characteristics.",
                'management': sections[1] if len(sections) > 1 else "Management appears competent based on financial performance metrics.",
                'competitive': sections[2] if len(sections) > 2 else "Company maintains competitive position within its sector.",
                'industry': sections[3] if len(sections) > 3 else "Industry shows stable long-term growth prospects."
            }
            
        except Exception as e:
            logger.error(f"Error analyzing company fundamentals: {e}")
            return {
                'assessment': "Unable to complete detailed fundamental analysis.",
                'management': "Management assessment unavailable.",
                'competitive': "Competitive analysis unavailable.",
                'industry': "Industry outlook assessment unavailable."
            }
    
    def _analyze_market_context(self, symbol: str, sentiment: Dict, 
                              market_data: pd.DataFrame, context: str) -> Dict:
        """Analyze market context and sentiment"""
        try:
            recent_volatility = market_data['Close'].pct_change().tail(20).std() if not market_data.empty else 0.02
            
            prompt = f"""
            {context}
            
            Recent volatility: {recent_volatility:.3f}
            
            As a seasoned market strategist, analyze the current market context for {symbol}:
            
            1. MARKET SENTIMENT INTERPRETATION: How should we interpret the current sentiment readings in the context of broader market conditions?
            
            2. MACROECONOMIC IMPACT: What macroeconomic factors are most likely to impact this stock, and how?
            
            3. SECTOR ROTATION ANALYSIS: Is this sector currently in favor or out of favor? What rotation dynamics should we consider?
            
            Focus on nuanced insights that consider market psychology, institutional behavior, and macro trends.
            """
            
            response = self.gpt_analyst.analyze_with_gpt(prompt, max_tokens=600)
            
            sections = response.split('\n\n')
            
            return {
                'sentiment': sections[0] if len(sections) > 0 else "Market sentiment appears neutral with mixed signals.",
                'macro': sections[1] if len(sections) > 1 else "Macroeconomic environment presents balanced risks and opportunities.",
                'sector': sections[2] if len(sections) > 2 else "Sector positioning appears stable in current market rotation."
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return {
                'sentiment': "Market context analysis unavailable.",
                'macro': "Macroeconomic analysis unavailable.",
                'sector': "Sector analysis unavailable."
            }
    
    def _generate_creative_insights(self, symbol: str, fundamentals: Dict, 
                                  sentiment: Dict, context: str) -> Dict:
        """Generate creative insights and contrarian viewpoints"""
        try:
            prompt = f"""
            {context}
            
            As a contrarian investor and creative thinker, identify non-obvious insights for {symbol}:
            
            1. UNIQUE OPPORTUNITIES: What opportunities might the market be overlooking? Think beyond consensus views.
            
            2. HIDDEN RISKS: What risks are not adequately reflected in current valuations or sentiment?
            
            3. CONTRARIAN VIEWPOINTS: What would a contrarian investor argue about this stock? Challenge conventional wisdom.
            
            Be specific and provide reasoning for each insight. Focus on what others might miss.
            """
            
            response = self.gpt_analyst.analyze_with_gpt(prompt, max_tokens=600)
            
            # Simple parsing to extract lists
            opportunities = ["Potential market expansion opportunities", "Operational efficiency improvements"]
            risks = ["Regulatory headwinds", "Competitive pressure intensification"]
            contrarian = ["Market may be underestimating long-term growth potential"]
            
            try:
                # Try to extract actual insights from response
                lines = response.split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if 'opportunities' in line.lower():
                        current_section = 'opportunities'
                        opportunities = []
                    elif 'risks' in line.lower():
                        current_section = 'risks'
                        risks = []
                    elif 'contrarian' in line.lower():
                        current_section = 'contrarian'
                        contrarian = []
                    elif line.startswith('-') or line.startswith('•'):
                        if current_section == 'opportunities':
                            opportunities.append(line[1:].strip())
                        elif current_section == 'risks':
                            risks.append(line[1:].strip())
                        elif current_section == 'contrarian':
                            contrarian.append(line[1:].strip())
            except:
                pass  # Use default values
            
            return {
                'opportunities': opportunities[:3],  # Limit to 3 items
                'risks': risks[:3],
                'contrarian': contrarian[:3]
            }
            
        except Exception as e:
            logger.error(f"Error generating creative insights: {e}")
            return {
                'opportunities': ["Limited insight generation available"],
                'risks': ["Standard market risks apply"],
                'contrarian': ["No contrarian viewpoints identified"]
            }
    
    def _generate_human_recommendation(self, symbol: str, company_assessment: Dict,
                                     market_context: Dict, creative_insights: Dict,
                                     context: str) -> Tuple[str, float, str, List[str]]:
        """Generate overall human-like recommendation"""
        try:
            prompt = f"""
            {context}
            
            ANALYSIS SUMMARY:
            Company Assessment: {company_assessment['assessment'][:200]}...
            Market Context: {market_context['sentiment'][:200]}...
            Key Opportunities: {', '.join(creative_insights['opportunities'][:2])}
            Key Risks: {', '.join(creative_insights['risks'][:2])}
            
            As a senior portfolio manager, provide your final investment recommendation for {symbol}:
            
            1. RECOMMENDATION: Choose from STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
            2. CONFIDENCE: Rate your confidence from 0.1 to 0.9
            3. REASONING: Provide a clear narrative explaining your decision
            4. CATALYSTS: List 2-3 key catalysts that could drive the stock
            
            Format your response as:
            RECOMMENDATION: [choice]
            CONFIDENCE: [number]
            REASONING: [narrative]
            CATALYSTS: [list]
            """
            
            response = self.gpt_analyst.analyze_with_gpt(prompt, max_tokens=400)
            
            # Parse response
            recommendation = 'HOLD'
            confidence = 0.6
            narrative = "Balanced risk-reward profile with moderate upside potential."
            catalysts = ["Earnings growth", "Market expansion", "Operational improvements"]
            
            try:
                lines = response.split('\n')
                for line in lines:
                    if line.startswith('RECOMMENDATION:'):
                        rec = line.split(':')[1].strip().upper()
                        if rec in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
                            recommendation = rec
                    elif line.startswith('CONFIDENCE:'):
                        try:
                            confidence = float(line.split(':')[1].strip())
                            confidence = max(0.1, min(0.9, confidence))
                        except:
                            pass
                    elif line.startswith('REASONING:'):
                        narrative = line.split(':', 1)[1].strip()
                    elif line.startswith('CATALYSTS:'):
                        catalyst_text = line.split(':', 1)[1].strip()
                        catalysts = [c.strip() for c in catalyst_text.split(',')][:3]
            except:
                pass  # Use defaults
            
            return recommendation, confidence, narrative, catalysts
            
        except Exception as e:
            logger.error(f"Error generating human recommendation: {e}")
            return 'HOLD', 0.6, "Analysis unavailable", ["Market conditions", "Company performance"]
    
    def _create_default_result(self, symbol: str) -> HumanAIAnalysisResult:
        """Create default result when analysis fails"""
        return HumanAIAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            company_assessment="Analysis unavailable",
            management_quality="Assessment unavailable",
            competitive_position="Analysis unavailable",
            industry_outlook="Outlook unavailable",
            market_sentiment_interpretation="Sentiment analysis unavailable",
            macro_economic_impact="Impact analysis unavailable",
            sector_rotation_analysis="Rotation analysis unavailable",
            unique_opportunities=["Analysis unavailable"],
            hidden_risks=["Analysis unavailable"],
            contrarian_viewpoints=["Analysis unavailable"],
            human_recommendation='HOLD',
            human_confidence=0.5,
            reasoning_narrative="Insufficient data for comprehensive analysis",
            key_catalysts=["Market conditions", "Company performance"]
        )

# Example usage and testing
if __name__ == "__main__":
    # Test the human-like AI analyst
    from data_collector import MultiSourceDataCollector
    
    config = {}
    collector = MultiSourceDataCollector(config)
    human_ai_analyst = HumanAIAnalyst()
    
    # Test with a symbol
    symbol = 'AAPL'
    print(f"\n{'='*60}")
    print(f"Testing Human-like AI Analysis for {symbol}")
    print(f"{'='*60}")
    
    # Collect data
    data = collector.collect_comprehensive_data(symbol)
    market_data = pd.DataFrame(data['market_data'])
    
    if not market_data.empty:
        # Convert index to datetime if needed
        if 'Date' in market_data.columns:
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            market_data.set_index('Date', inplace=True)
        
        # Perform human-like AI analysis
        human_result = human_ai_analyst.analyze_stock(
            symbol, 
            market_data, 
            data['fundamentals'], 
            data['sentiment'],
            data['characteristics']
        )
        
        # Display results
        print(f"\n🧠 HUMAN-LIKE AI ANALYSIS RESULTS:")
        print(f"Company Assessment: {human_result.company_assessment[:100]}...")
        print(f"Competitive Position: {human_result.competitive_position[:100]}...")
        print(f"Market Sentiment: {human_result.market_sentiment_interpretation[:100]}...")
        print(f"Recommendation: {human_result.human_recommendation} (confidence: {human_result.human_confidence:.2f})")
        
        print(f"\n💡 Creative Insights:")
        print(f"Opportunities: {', '.join(human_result.unique_opportunities[:2])}")
        print(f"Risks: {', '.join(human_result.hidden_risks[:2])}")
        print(f"Contrarian Views: {', '.join(human_result.contrarian_viewpoints[:1])}")
        
        print(f"\n🎯 Key Catalysts:")
        for catalyst in human_result.key_catalysts[:3]:
            print(f"  • {catalyst}")
    else:
        print("❌ No market data available for analysis")
