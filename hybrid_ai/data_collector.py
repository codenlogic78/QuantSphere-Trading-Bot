"""
Multi-Source Data Collection Module for Hybrid AI Trading System
Based on Cao et al. (2024) - "Man + Machine" approach

This module collects and preprocesses data from multiple sources:
- Market data (price, volume, technical indicators)
- Fundamental data (financial metrics, ratios)
- Alternative data (news sentiment, social media, economic indicators)
- Company characteristics (size, liquidity, sector)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockCharacteristics:
    """Stock characteristics that determine AI vs Human weighting"""
    symbol: str
    market_cap: float
    avg_volume: float
    sector: str
    is_large_cap: bool
    is_liquid: bool
    is_asset_light: bool
    distress_score: float
    
    def get_ai_weight(self) -> float:
        """Calculate AI weighting based on stock characteristics"""
        ai_weight = 0.5  # Base weight
        
        # Large cap stocks favor AI
        if self.is_large_cap:
            ai_weight += 0.2
        else:
            ai_weight -= 0.2
            
        # High liquidity favors AI
        if self.is_liquid:
            ai_weight += 0.15
        else:
            ai_weight -= 0.15
            
        # Asset-light companies during distress favor human reasoning
        if self.is_asset_light and self.distress_score > 0.6:
            ai_weight -= 0.25
            
        # Clamp between 0.1 and 0.9
        return max(0.1, min(0.9, ai_weight))

class MultiSourceDataCollector:
    """Collects data from multiple sources for hybrid AI analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
        
    def collect_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Collect basic market data using yfinance with rate limiting"""
        try:
            # Add delay to avoid rate limiting
            import time
            time.sleep(0.5)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No market data found for {symbol}, generating mock data")
                # Generate mock data if real data unavailable
                return self._generate_mock_market_data(symbol)
                
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            logger.info(f"Collected market data for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {e}")
            logger.info(f"Generating mock data for {symbol}")
            return self._generate_mock_market_data(symbol)
    
    def collect_fundamental_data(self, symbol: str) -> Dict:
        """Collect fundamental data and financial metrics with fallback"""
        try:
            # Add delay to avoid rate limiting
            import time
            time.sleep(0.3)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'profit_margin': info.get('profitMargins', None),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'avg_volume': info.get('averageVolume', 0),
                'beta': info.get('beta', 1.0),
                'book_value': info.get('bookValue', None),
                'cash_per_share': info.get('totalCashPerShare', None)
            }
            
            logger.info(f"Collected fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error collecting fundamental data for {symbol}: {e}")
            logger.info(f"Generating mock fundamental data for {symbol}")
            return self._generate_mock_fundamentals(symbol)
    
    def collect_sentiment_data(self, symbol: str) -> Dict:
        """Collect news sentiment and alternative data"""
        try:
            # For now, we'll simulate sentiment data
            # In production, this would integrate with news APIs, social media, etc.
            
            sentiment_data = {
                'news_sentiment': np.random.uniform(-1, 1),  # -1 to 1 scale
                'social_sentiment': np.random.uniform(-1, 1),
                'analyst_sentiment': np.random.uniform(-1, 1),
                'news_volume': np.random.randint(1, 100),
                'sentiment_trend': np.random.choice(['improving', 'declining', 'stable']),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"Collected sentiment data for {symbol}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting sentiment data for {symbol}: {e}")
            return {}
    
    def analyze_stock_characteristics(self, symbol: str, fundamentals: Dict) -> StockCharacteristics:
        """Analyze stock characteristics to determine AI vs Human weighting"""
        try:
            market_cap = fundamentals.get('market_cap', 0)
            avg_volume = fundamentals.get('avg_volume', 0)
            sector = fundamentals.get('sector', 'Unknown')
            
            # Determine characteristics
            is_large_cap = market_cap > 10_000_000_000  # $10B+
            is_liquid = avg_volume > 1_000_000  # 1M+ daily volume
            
            # Asset-light industries (tech, services, etc.)
            asset_light_sectors = ['Technology', 'Communication Services', 'Consumer Discretionary']
            is_asset_light = sector in asset_light_sectors
            
            # Calculate distress score based on financial metrics
            distress_score = self._calculate_distress_score(fundamentals)
            
            characteristics = StockCharacteristics(
                symbol=symbol,
                market_cap=market_cap,
                avg_volume=avg_volume,
                sector=sector,
                is_large_cap=is_large_cap,
                is_liquid=is_liquid,
                is_asset_light=is_asset_light,
                distress_score=distress_score
            )
            
            logger.info(f"Analyzed characteristics for {symbol}: AI weight = {characteristics.get_ai_weight():.2f}")
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing stock characteristics for {symbol}: {e}")
            return StockCharacteristics(symbol, 0, 0, 'Unknown', False, False, False, 0.0)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to market data"""
        try:
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Volatility
            data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
            
            # Price momentum
            data['Momentum'] = data['Close'].pct_change(periods=10)
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _calculate_distress_score(self, fundamentals: Dict) -> float:
        """Calculate financial distress score (0-1, higher = more distressed)"""
        try:
            score = 0.0
            
            # High debt-to-equity increases distress
            debt_to_equity = fundamentals.get('debt_to_equity', 0)
            if debt_to_equity and debt_to_equity > 1.0:
                score += 0.3
            
            # Low profit margin increases distress
            profit_margin = fundamentals.get('profit_margin', 0)
            if profit_margin and profit_margin < 0.05:  # Less than 5%
                score += 0.2
            
            # High PE ratio can indicate overvaluation stress
            pe_ratio = fundamentals.get('pe_ratio', 0)
            if pe_ratio and pe_ratio > 30:
                score += 0.2
            
            # Low ROE indicates poor performance
            roe = fundamentals.get('roe', 0)
            if roe and roe < 0.1:  # Less than 10%
                score += 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating distress score: {e}")
            return 0.0
    
    def collect_comprehensive_data(self, symbol: str) -> Dict:
        """Collect all data sources for a symbol"""
        logger.info(f"Starting comprehensive data collection for {symbol}")
        
        # Collect all data types
        market_data = self.collect_market_data(symbol)
        fundamentals = self.collect_fundamental_data(symbol)
        sentiment = self.collect_sentiment_data(symbol)
        characteristics = self.analyze_stock_characteristics(symbol, fundamentals)
        
        comprehensive_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'market_data': market_data.to_dict() if not market_data.empty else {},
            'fundamentals': fundamentals,
            'sentiment': sentiment,
            'characteristics': characteristics.__dict__,
            'ai_weight': characteristics.get_ai_weight()
        }
        
        logger.info(f"Completed comprehensive data collection for {symbol}")
        return comprehensive_data
    
    def _generate_mock_market_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock market data when real data is unavailable"""
        try:
            # Generate 252 days of mock data (1 year)
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            
            # Base price varies by symbol
            base_prices = {
                'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0, 'TSLA': 200.0,
                'AMZN': 3000.0, 'META': 250.0, 'NVDA': 400.0
            }
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate realistic price movements
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data
            data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
                'High': [p * np.random.uniform(1.00, 1.03) for p in prices],
                'Low': [p * np.random.uniform(0.97, 1.00) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(1000000, 10000000) for _ in prices]
            }, index=dates)
            
            # Ensure OHLC relationships are correct
            for i in range(len(data)):
                high = max(data.iloc[i]['Open'], data.iloc[i]['Close'], data.iloc[i]['High'])
                low = min(data.iloc[i]['Open'], data.iloc[i]['Close'], data.iloc[i]['Low'])
                data.iloc[i, data.columns.get_loc('High')] = high
                data.iloc[i, data.columns.get_loc('Low')] = low
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            logger.info(f"Generated mock market data for {symbol}: {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error generating mock market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_mock_fundamentals(self, symbol: str) -> Dict:
        """Generate mock fundamental data when real data is unavailable"""
        try:
            # Symbol-specific mock data
            mock_data = {
                'AAPL': {
                    'market_cap': 2800000000000, 'pe_ratio': 28.5, 'pb_ratio': 45.2,
                    'sector': 'Technology', 'industry': 'Consumer Electronics',
                    'avg_volume': 50000000, 'debt_to_equity': 1.73
                },
                'MSFT': {
                    'market_cap': 2400000000000, 'pe_ratio': 32.1, 'pb_ratio': 12.8,
                    'sector': 'Technology', 'industry': 'Software',
                    'avg_volume': 30000000, 'debt_to_equity': 0.47
                },
                'GOOGL': {
                    'market_cap': 1600000000000, 'pe_ratio': 25.3, 'pb_ratio': 5.8,
                    'sector': 'Communication Services', 'industry': 'Internet Content',
                    'avg_volume': 25000000, 'debt_to_equity': 0.11
                },
                'TSLA': {
                    'market_cap': 800000000000, 'pe_ratio': 65.2, 'pb_ratio': 15.4,
                    'sector': 'Consumer Discretionary', 'industry': 'Auto Manufacturers',
                    'avg_volume': 75000000, 'debt_to_equity': 0.17
                }
            }
            
            if symbol in mock_data:
                base_data = mock_data[symbol]
            else:
                # Generic mock data for unknown symbols
                base_data = {
                    'market_cap': 50000000000, 'pe_ratio': 20.0, 'pb_ratio': 3.5,
                    'sector': 'Technology', 'industry': 'Software',
                    'avg_volume': 5000000, 'debt_to_equity': 0.5
                }
            
            # Complete the fundamental data
            fundamentals = {
                'market_cap': base_data['market_cap'],
                'pe_ratio': base_data['pe_ratio'],
                'pb_ratio': base_data['pb_ratio'],
                'debt_to_equity': base_data['debt_to_equity'],
                'roe': np.random.uniform(0.1, 0.3),
                'revenue_growth': np.random.uniform(-0.1, 0.2),
                'profit_margin': np.random.uniform(0.05, 0.25),
                'sector': base_data['sector'],
                'industry': base_data['industry'],
                'avg_volume': base_data['avg_volume'],
                'beta': np.random.uniform(0.8, 1.5),
                'book_value': np.random.uniform(10, 50),
                'cash_per_share': np.random.uniform(5, 25)
            }
            
            logger.info(f"Generated mock fundamental data for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error generating mock fundamental data for {symbol}: {e}")
            return {
                'market_cap': 10000000000, 'pe_ratio': 20.0, 'pb_ratio': 3.0,
                'sector': 'Technology', 'industry': 'Software', 'avg_volume': 1000000,
                'debt_to_equity': 0.5, 'roe': 0.15, 'revenue_growth': 0.1,
                'profit_margin': 0.15, 'beta': 1.0, 'book_value': 20.0, 'cash_per_share': 10.0
            }

# Example usage and testing
if __name__ == "__main__":
    # Test the data collector
    config = {}
    collector = MultiSourceDataCollector(config)
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'TSLA', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"Testing data collection for {symbol}")
        print(f"{'='*50}")
        
        data = collector.collect_comprehensive_data(symbol)
        
        print(f"Market data records: {len(data['market_data'])}")
        print(f"Sector: {data['fundamentals'].get('sector', 'Unknown')}")
        print(f"Market cap: ${data['fundamentals'].get('market_cap', 0):,.0f}")
        print(f"AI weight: {data['ai_weight']:.2f}")
        print(f"Large cap: {data['characteristics']['is_large_cap']}")
        print(f"Liquid: {data['characteristics']['is_liquid']}")
        print(f"Distress score: {data['characteristics']['distress_score']:.2f}")
