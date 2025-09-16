#!/usr/bin/env python3
"""
QuantSphere AI Trading Platform - Real-Time Data Feed
Live market data integration for real-time charts and trading
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import json
from typing import Dict, List, Optional
import websocket
import os
from dotenv import load_dotenv

load_dotenv()

class RealTimeDataFeed:
    """Real-time market data feed with multiple data sources"""
    
    def __init__(self):
        self.data_cache = {}
        self.subscribers = []
        self.is_running = False
        self.update_thread = None
        
        # API configurations
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.iex_token = os.getenv("IEX_CLOUD_TOKEN", "")
        
        # Data sources priority
        self.data_sources = [
            self.get_alpha_vantage_data,
            self.get_polygon_data,
            self.get_iex_data,
            self.get_yahoo_finance_data,
            self.generate_realistic_mock_data  # Fallback
        ]
    
    def start_feed(self, symbols: List[str], update_interval: int = 5):
        """Start real-time data feed for given symbols"""
        self.symbols = symbols
        self.update_interval = update_interval
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        print(f"📡 Real-time data feed started for {symbols}")
        print(f"🔄 Update interval: {update_interval} seconds")
    
    def stop_feed(self):
        """Stop real-time data feed"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        print("📡 Real-time data feed stopped")
    
    def subscribe(self, callback_function):
        """Subscribe to data updates"""
        self.subscribers.append(callback_function)
    
    def unsubscribe(self, callback_function):
        """Unsubscribe from data updates"""
        if callback_function in self.subscribers:
            self.subscribers.remove(callback_function)
    
    def _update_loop(self):
        """Main update loop for real-time data"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # Try each data source until one works
                    data = None
                    for source in self.data_sources:
                        try:
                            data = source(symbol)
                            if data is not None:
                                break
                        except Exception as e:
                            print(f"⚠️ Data source failed for {symbol}: {e}")
                            continue
                    
                    if data is not None:
                        # Update cache
                        self.data_cache[symbol] = {
                            'data': data,
                            'timestamp': datetime.now(),
                            'source': source.__name__
                        }
                        
                        # Notify subscribers
                        for callback in self.subscribers:
                            try:
                                callback(symbol, data)
                            except Exception as e:
                                print(f"⚠️ Subscriber callback failed: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"❌ Update loop error: {e}")
                time.sleep(self.update_interval)
    
    def get_alpha_vantage_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from Alpha Vantage API"""
        if self.alpha_vantage_key == "demo":
            return None
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '5min',
            'apikey': self.alpha_vantage_key,
            'outputsize': 'compact'
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'Time Series (5min)' in data:
            df = pd.DataFrame(data['Time Series (5min)']).T
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            return df.sort_index()
        
        return None
    
    def get_polygon_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from Polygon.io API"""
        if not self.polygon_key:
            return None
        
        # Get last 100 5-minute bars
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {'apikey': self.polygon_key}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'results' in data and data['results']:
            df_data = []
            for bar in data['results']:
                df_data.append({
                    'timestamp': pd.to_datetime(bar['t'], unit='ms'),
                    'Open': bar['o'],
                    'High': bar['h'],
                    'Low': bar['l'],
                    'Close': bar['c'],
                    'Volume': bar['v']
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
        
        return None
    
    def get_iex_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from IEX Cloud API"""
        if not self.iex_token:
            return None
        
        url = f"https://cloud.iexapis.com/stable/stock/{symbol}/intraday-prices"
        params = {
            'token': self.iex_token,
            'chartInterval': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data and isinstance(data, list):
            df_data = []
            for bar in data:
                if bar['open'] is not None:
                    df_data.append({
                        'timestamp': pd.to_datetime(f"{bar['date']} {bar['minute']}", format='%Y-%m-%d %H:%M'),
                        'Open': bar['open'],
                        'High': bar['high'],
                        'Low': bar['low'],
                        'Close': bar['close'],
                        'Volume': bar['volume'] or 0
                    })
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                return df.sort_index()
        
        return None
    
    def get_yahoo_finance_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get data from Yahoo Finance (free, no API key required)"""
        try:
            import yfinance as yf
            
            # Get 5-day 5-minute data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="5m")
            
            if not df.empty:
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                return df
            
        except ImportError:
            print("📦 yfinance not installed. Install with: pip install yfinance")
        except Exception as e:
            print(f"⚠️ Yahoo Finance error: {e}")
        
        return None
    
    def generate_realistic_mock_data(self, symbol: str) -> pd.DataFrame:
        """Generate realistic mock data as fallback"""
        # Get base price for symbol
        base_prices = {
            'AAPL': 175, 'MSFT': 350, 'GOOGL': 140, 'AMZN': 145, 'TSLA': 250,
            'NVDA': 450, 'META': 320, 'NFLX': 400, 'AMD': 110, 'CRM': 220
        }
        
        base_price = base_prices.get(symbol, 150)
        
        # Generate last 100 5-minute bars
        end_time = datetime.now()
        times = [end_time - timedelta(minutes=5*i) for i in range(100, 0, -1)]
        
        # Realistic price simulation
        np.random.seed(hash(symbol + str(end_time.date())) % 2**32)
        
        prices = [base_price]
        for i in range(1, 100):
            # Add intraday patterns
            time_factor = np.sin(i * 2 * np.pi / 78) * 0.002  # Intraday cycle
            volatility = 0.001 + 0.002 * abs(np.random.normal(0, 1))  # Volatility clustering
            
            change = np.random.normal(time_factor, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.8))  # Floor price
        
        # Generate OHLCV
        data = []
        for i, (time, close) in enumerate(zip(times, prices)):
            vol = abs(np.random.normal(0, 0.01))
            high = close * (1 + vol * np.random.uniform(0.2, 0.8))
            low = close * (1 - vol * np.random.uniform(0.2, 0.8))
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.002)) if i > 0 else close
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            volume = int(np.random.lognormal(12, 0.8))  # Realistic volume
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data, index=times)
        return df.sort_index()
    
    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get latest cached data for symbol"""
        if symbol in self.data_cache:
            cache_entry = self.data_cache[symbol]
            # Check if data is recent (within 2 minutes)
            if (datetime.now() - cache_entry['timestamp']).seconds < 120:
                return cache_entry
        
        return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        data = self.get_latest_data(symbol)
        if data and not data['data'].empty:
            return float(data['data']['Close'].iloc[-1])
        return None
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (simplified)"""
        now = datetime.now()
        # US market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close

# Real-time chart updater
class RealTimeChartUpdater:
    """Updates charts with real-time data"""
    
    def __init__(self, chart_visualizer):
        self.chart_visualizer = chart_visualizer
        self.data_feed = RealTimeDataFeed()
        self.data_feed.subscribe(self.on_data_update)
    
    def start_updates(self, symbols: List[str]):
        """Start real-time chart updates"""
        self.data_feed.start_feed(symbols, update_interval=10)  # Update every 10 seconds
    
    def stop_updates(self):
        """Stop real-time chart updates"""
        self.data_feed.stop_feed()
    
    def on_data_update(self, symbol: str, data: pd.DataFrame):
        """Handle data updates"""
        try:
            # Update chart if it's showing this symbol
            if hasattr(self.chart_visualizer, 'symbol_var'):
                current_symbol = self.chart_visualizer.symbol_var.get()
                if current_symbol == symbol:
                    # Update the chart with new data
                    self.chart_visualizer.update_chart_with_data(data)
                    print(f"📊 Chart updated for {symbol} at {datetime.now().strftime('%H:%M:%S')}")
        
        except Exception as e:
            print(f"⚠️ Chart update error: {e}")

# Test the real-time data feed
if __name__ == "__main__":
    print("🚀 Testing QuantSphere Real-Time Data Feed")
    print("=" * 50)
    
    # Initialize data feed
    feed = RealTimeDataFeed()
    
    # Test callback function
    def on_update(symbol, data):
        current_price = data['Close'].iloc[-1]
        timestamp = data.index[-1]
        print(f"📈 {symbol}: ${current_price:.2f} at {timestamp.strftime('%H:%M:%S')}")
    
    # Subscribe to updates
    feed.subscribe(on_update)
    
    # Start feed for test symbols
    test_symbols = ['AAPL', 'MSFT']
    feed.start_feed(test_symbols, update_interval=5)
    
    try:
        # Run for 30 seconds
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping data feed...")
    finally:
        feed.stop_feed()
    
    print("✅ Real-time data feed test completed!")
