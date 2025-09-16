#!/usr/bin/env python3
"""
QuantSphere AI Trading Platform - Advanced Chart Visualizer
Comprehensive charting system with candlestick charts, technical indicators, and portfolio analytics
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

class QuantSphereChartVisualizer:
    """Advanced charting system for QuantSphere trading platform"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.fig = None
        self.canvas = None
        
        # Chart configuration
        self.chart_config = {
            'figsize': (14, 10),
            'dpi': 100,
            'facecolor': '#1e1e1e',
            'edgecolor': 'white'
        }
        
        # Color scheme
        self.colors = {
            'up': '#00ff88',      # Green for bullish
            'down': '#ff4444',    # Red for bearish
            'volume': '#4488ff',  # Blue for volume
            'ma_fast': '#ffaa00', # Orange for fast MA
            'ma_slow': '#ff00ff', # Magenta for slow MA
            'rsi': '#00ffff',     # Cyan for RSI
            'macd': '#ffff00',    # Yellow for MACD
            'bb_upper': '#888888', # Gray for Bollinger upper
            'bb_lower': '#888888', # Gray for Bollinger lower
            'grid': '#333333'     # Dark gray for grid
        }
    
    def create_main_chart_window(self):
        """Create main charting window"""
        if self.parent is None:
            self.chart_window = tk.Tk()
            self.chart_window.title("QuantSphere - Advanced Charts")
            self.chart_window.geometry("1400x900")
            self.chart_window.configure(bg='#1e1e1e')
        else:
            self.chart_window = tk.Toplevel(self.parent)
            self.chart_window.title("QuantSphere - Advanced Charts")
            self.chart_window.geometry("1400x900")
            self.chart_window.configure(bg='#1e1e1e')
        
        # Create notebook for different chart types
        self.notebook = ttk.Notebook(self.chart_window)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_price_chart_tab()
        self.create_portfolio_tab()
        self.create_risk_tab()
        self.create_performance_tab()
        
        return self.chart_window
    
    def create_price_chart_tab(self):
        """Create price chart tab with technical indicators"""
        price_frame = ttk.Frame(self.notebook)
        self.notebook.add(price_frame, text="Price Charts")
        
        # Controls frame
        controls_frame = ttk.Frame(price_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Symbol:").pack(side='left', padx=5)
        self.symbol_var = tk.StringVar(value="AAPL")
        symbol_entry = ttk.Entry(controls_frame, textvariable=self.symbol_var, width=10)
        symbol_entry.pack(side='left', padx=5)
        
        ttk.Button(controls_frame, text="Update Chart", 
                  command=self.update_price_chart).pack(side='left', padx=10)
        
        # Chart frame
        chart_frame = ttk.Frame(price_frame)
        chart_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create initial chart
        self.create_price_charts(chart_frame)
    
    def create_portfolio_tab(self):
        """Create portfolio performance tab"""
        portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(portfolio_frame, text="Portfolio")
        
        # Create portfolio charts
        self.create_portfolio_charts(portfolio_frame)
    
    def create_risk_tab(self):
        """Create risk analysis tab"""
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="Risk Analysis")
        
        # Create risk charts
        self.create_risk_charts(risk_frame)
    
    def create_performance_tab(self):
        """Create performance analytics tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")
        
        # Create performance charts
        self.create_performance_charts(perf_frame)
    
    def get_real_time_data(self, symbol='AAPL'):
        """Get real-time or realistic data for charting"""
        try:
            # Try to get real data from Alpha Vantage
            from realtime_data_feed import RealTimeDataFeed
            feed = RealTimeDataFeed()
            
            # First try Alpha Vantage
            data = feed.get_alpha_vantage_data(symbol)
            if data is not None and not data.empty:
                print(f"📊 Using real Alpha Vantage data for {symbol}")
                return data
            
            # Fallback to Yahoo Finance
            data = feed.get_yahoo_finance_data(symbol)
            if data is not None and not data.empty:
                print(f"📊 Using Yahoo Finance data for {symbol}")
                return data
            
            # Final fallback to realistic mock data
            print(f"📊 Using realistic mock data for {symbol}")
            return feed.generate_realistic_mock_data(symbol)
            
        except Exception as e:
            print(f"⚠️ Data retrieval error: {e}")
            # Generate fallback data
            return self.generate_sample_data(symbol)
    
    def generate_sample_data(self, symbol='AAPL', days=30):
        """Generate realistic sample OHLCV data (fallback)"""
        np.random.seed(42)  # For consistent demo data
        
        # Base prices for different symbols
        base_prices = {
            'AAPL': 175, 'MSFT': 350, 'GOOGL': 140, 'AMZN': 145, 'TSLA': 250,
            'NVDA': 450, 'META': 320, 'NFLX': 400, 'AMD': 110, 'CRM': 220
        }
        
        base_price = base_prices.get(symbol, 150)
        
        # Generate timestamps (5-minute intervals)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='5T')
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.002, len(timestamps))  # 0.2% volatility
        
        # Add intraday patterns
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Market open volatility
            if 9 <= hour <= 10:
                returns[i] *= 1.5
            # Lunch lull
            elif 12 <= hour <= 13:
                returns[i] *= 0.5
            # Close volatility
            elif 15 <= hour <= 16:
                returns[i] *= 1.3
        
        # Generate price series
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, 0.001))
            
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.lognormal(12, 0.8))  # Realistic volume
            
            data.append({
                'timestamp': ts,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for charting"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        return df
    
    def create_price_charts(self, parent_frame):
        """Create comprehensive price charts with technical indicators"""
        # Get real-time data
        symbol = self.symbol_var.get()
        df = self.get_real_time_data(symbol)
        df = self.calculate_technical_indicators(df)
        
        # Create figure with subplots
        self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), 
                                                          facecolor='#1e1e1e',
                                                          gridspec_kw={'height_ratios': [3, 1], 
                                                                      'width_ratios': [3, 1]})
        
        # Main price chart (top-left)
        self.plot_candlesticks(ax1, df)
        self.plot_moving_averages(ax1, df)
        self.plot_bollinger_bands(ax1, df)
        
        ax1.set_title(f'{symbol} - Price Chart with Technical Indicators', 
                     color='white', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, color=self.colors['grid'])
        ax1.legend(loc='upper left', fancybox=True, framealpha=0.9)
        
        # Volume chart (top-right)
        self.plot_volume(ax2, df)
        ax2.set_title('Volume Analysis', color='white', fontsize=12)
        ax2.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # RSI chart (bottom-left)
        self.plot_rsi(ax3, df)
        ax3.set_title('RSI (14)', color='white', fontsize=12)
        ax3.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # MACD chart (bottom-right)
        self.plot_macd(ax4, df)
        ax4.set_title('MACD', color='white', fontsize=12)
        ax4.grid(True, alpha=0.3, color=self.colors['grid'])
        
        # Format dates
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Embed in tkinter
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_candlesticks(self, ax, df):
        """Plot candlestick chart"""
        # Prepare data
        up = df[df['Close'] >= df['Open']]
        down = df[df['Close'] < df['Open']]
        
        # Plot up candles
        ax.bar(up.index, up['Close'] - up['Open'], bottom=up['Open'],
               color=self.colors['up'], alpha=0.8, width=0.8, label='Bullish')
        ax.bar(up.index, up['High'] - up['Close'], bottom=up['Close'],
               color=self.colors['up'], alpha=0.4, width=0.1)
        ax.bar(up.index, up['Open'] - up['Low'], bottom=up['Low'],
               color=self.colors['up'], alpha=0.4, width=0.1)
        
        # Plot down candles
        ax.bar(down.index, down['Open'] - down['Close'], bottom=down['Close'],
               color=self.colors['down'], alpha=0.8, width=0.8, label='Bearish')
        ax.bar(down.index, down['High'] - down['Open'], bottom=down['Open'],
               color=self.colors['down'], alpha=0.4, width=0.1)
        ax.bar(down.index, down['Close'] - down['Low'], bottom=down['Low'],
               color=self.colors['down'], alpha=0.4, width=0.1)
    
    def plot_moving_averages(self, ax, df):
        """Plot moving averages"""
        ax.plot(df.index, df['SMA_20'], color=self.colors['ma_fast'], 
                linewidth=2, label='SMA 20', alpha=0.8)
        ax.plot(df.index, df['SMA_50'], color=self.colors['ma_slow'], 
                linewidth=2, label='SMA 50', alpha=0.8)
    
    def plot_bollinger_bands(self, ax, df):
        """Plot Bollinger Bands"""
        ax.plot(df.index, df['BB_Upper'], color=self.colors['bb_upper'], 
                linewidth=1, alpha=0.6, linestyle='--', label='BB Upper')
        ax.plot(df.index, df['BB_Lower'], color=self.colors['bb_lower'], 
                linewidth=1, alpha=0.6, linestyle='--', label='BB Lower')
        ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], 
                       alpha=0.1, color='gray')
    
    def plot_volume(self, ax, df):
        """Plot volume with moving average"""
        colors = [self.colors['up'] if close >= open_price else self.colors['down'] 
                 for close, open_price in zip(df['Close'], df['Open'])]
        
        ax.bar(df.index, df['Volume'], color=colors, alpha=0.7, width=0.8)
        ax.plot(df.index, df['Volume_SMA'], color='white', linewidth=2, 
                label='Volume SMA', alpha=0.8)
        ax.legend()
    
    def plot_rsi(self, ax, df):
        """Plot RSI with overbought/oversold levels"""
        ax.plot(df.index, df['RSI'], color=self.colors['rsi'], linewidth=2)
        ax.axhline(y=70, color=self.colors['down'], linestyle='--', alpha=0.7, label='Overbought')
        ax.axhline(y=30, color=self.colors['up'], linestyle='--', alpha=0.7, label='Oversold')
        ax.axhline(y=50, color='white', linestyle='-', alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend()
    
    def plot_macd(self, ax, df):
        """Plot MACD with signal line and histogram"""
        ax.plot(df.index, df['MACD'], color=self.colors['macd'], linewidth=2, label='MACD')
        ax.plot(df.index, df['MACD_Signal'], color='white', linewidth=2, label='Signal')
        
        # MACD histogram
        colors = [self.colors['up'] if val >= 0 else self.colors['down'] 
                 for val in df['MACD_Histogram']]
        ax.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.6, width=0.8)
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.3)
        ax.legend()
    
    def create_portfolio_charts(self, parent_frame):
        """Create portfolio performance charts"""
        # Generate sample portfolio data
        portfolio_data = self.generate_portfolio_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), 
                                                    facecolor='#1e1e1e')
        
        # Portfolio value over time
        ax1.plot(portfolio_data['dates'], portfolio_data['portfolio_value'], 
                color=self.colors['up'], linewidth=3, label='Portfolio Value')
        ax1.fill_between(portfolio_data['dates'], portfolio_data['portfolio_value'], 
                        alpha=0.3, color=self.colors['up'])
        ax1.set_title('Portfolio Value Over Time', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Daily returns distribution
        ax2.hist(portfolio_data['daily_returns'], bins=30, color=self.colors['volume'], 
                alpha=0.7, edgecolor='white')
        ax2.set_title('Daily Returns Distribution', color='white', fontweight='bold')
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax3.fill_between(portfolio_data['dates'], portfolio_data['drawdown'], 
                        color=self.colors['down'], alpha=0.7, label='Drawdown')
        ax3.set_title('Portfolio Drawdown', color='white', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Asset allocation pie chart
        ax4.pie(portfolio_data['allocation_values'], labels=portfolio_data['allocation_labels'],
               colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'],
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Asset Allocation', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_risk_charts(self, parent_frame):
        """Create risk analysis charts"""
        # Generate risk data
        risk_data = self.generate_risk_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), 
                                                    facecolor='#1e1e1e')
        
        # VaR over time
        ax1.plot(risk_data['dates'], risk_data['var_95'], color=self.colors['down'], 
                linewidth=2, label='VaR 95%')
        ax1.plot(risk_data['dates'], risk_data['var_99'], color=self.colors['ma_slow'], 
                linewidth=2, label='VaR 99%')
        ax1.set_title('Value at Risk Over Time', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Correlation heatmap
        corr_matrix = risk_data['correlation_matrix']
        im = ax2.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_title('Asset Correlation Matrix', color='white', fontweight='bold')
        ax2.set_xticks(range(len(risk_data['symbols'])))
        ax2.set_yticks(range(len(risk_data['symbols'])))
        ax2.set_xticklabels(risk_data['symbols'])
        ax2.set_yticklabels(risk_data['symbols'])
        plt.colorbar(im, ax=ax2)
        
        # Risk contribution by asset
        ax3.bar(risk_data['symbols'], risk_data['risk_contribution'], 
               color=[self.colors['up'], self.colors['down'], self.colors['volume'], 
                     self.colors['ma_fast'], self.colors['rsi']])
        ax3.set_title('Risk Contribution by Asset', color='white', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Beta over time
        ax4.plot(risk_data['dates'], risk_data['portfolio_beta'], 
                color=self.colors['macd'], linewidth=2, label='Portfolio Beta')
        ax4.axhline(y=1.0, color='white', linestyle='--', alpha=0.7, label='Market Beta')
        ax4.set_title('Portfolio Beta Over Time', color='white', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_performance_charts(self, parent_frame):
        """Create performance analytics charts"""
        # Generate performance data
        perf_data = self.generate_performance_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), 
                                                    facecolor='#1e1e1e')
        
        # Cumulative returns comparison
        ax1.plot(perf_data['dates'], perf_data['portfolio_cumret'], 
                color=self.colors['up'], linewidth=3, label='Portfolio')
        ax1.plot(perf_data['dates'], perf_data['benchmark_cumret'], 
                color=self.colors['volume'], linewidth=2, label='S&P 500')
        ax1.set_title('Cumulative Returns vs Benchmark', color='white', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Rolling Sharpe ratio
        ax2.plot(perf_data['dates'], perf_data['rolling_sharpe'], 
                color=self.colors['macd'], linewidth=2)
        ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.7, label='Good Sharpe')
        ax2.set_title('Rolling Sharpe Ratio (252d)', color='white', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Monthly returns heatmap
        monthly_returns = perf_data['monthly_returns']
        im = ax3.imshow(monthly_returns, cmap='RdYlGn', aspect='auto')
        ax3.set_title('Monthly Returns Heatmap', color='white', fontweight='bold')
        ax3.set_xticks(range(12))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.set_yticks(range(len(perf_data['years'])))
        ax3.set_yticklabels(perf_data['years'])
        plt.colorbar(im, ax=ax3)
        
        # Performance metrics bar chart
        metrics = ['Sharpe', 'Sortino', 'Calmar', 'Max DD', 'Volatility']
        values = perf_data['performance_metrics']
        colors = [self.colors['up'], self.colors['volume'], self.colors['macd'], 
                 self.colors['down'], self.colors['ma_fast']]
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
        ax4.set_title('Performance Metrics', color='white', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', color='white')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def generate_portfolio_data(self):
        """Generate sample portfolio data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Portfolio value simulation
        initial_value = 100000
        daily_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual return, 15% vol
        portfolio_values = [initial_value]
        
        for ret in daily_returns[1:]:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = [(val - max_val) / max_val for val, max_val in zip(portfolio_values, running_max)]
        
        return {
            'dates': dates,
            'portfolio_value': portfolio_values,
            'daily_returns': daily_returns,
            'drawdown': drawdown,
            'allocation_labels': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'Cash'],
            'allocation_values': [25, 20, 30, 15, 10]
        }
    
    def generate_risk_data(self):
        """Generate sample risk data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # VaR simulation
        var_95 = np.random.uniform(1000, 5000, len(dates))
        var_99 = var_95 * 1.5
        
        # Correlation matrix
        np.random.seed(42)
        corr_matrix = np.random.uniform(0.3, 0.9, (5, 5))
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        
        # Portfolio beta
        portfolio_beta = 1.0 + np.random.normal(0, 0.2, len(dates))
        
        return {
            'dates': dates,
            'var_95': var_95,
            'var_99': var_99,
            'correlation_matrix': corr_matrix,
            'symbols': symbols,
            'risk_contribution': [0.3, 0.25, 0.2, 0.15, 0.1],
            'portfolio_beta': portfolio_beta
        }
    
    def generate_performance_data(self):
        """Generate sample performance data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Portfolio and benchmark returns
        portfolio_returns = np.random.normal(0.0008, 0.015, len(dates))
        benchmark_returns = np.random.normal(0.0005, 0.012, len(dates))
        
        portfolio_cumret = np.cumprod(1 + portfolio_returns)
        benchmark_cumret = np.cumprod(1 + benchmark_returns)
        
        # Rolling Sharpe ratio
        rolling_sharpe = []
        for i in range(252, len(portfolio_returns)):
            returns_window = portfolio_returns[i-252:i]
            sharpe = np.mean(returns_window) / np.std(returns_window) * np.sqrt(252)
            rolling_sharpe.append(sharpe)
        
        rolling_sharpe = [0] * 252 + rolling_sharpe
        
        # Monthly returns heatmap (simplified)
        years = ['2023', '2024']
        monthly_returns = np.random.normal(0.02, 0.05, (2, 12))
        
        return {
            'dates': dates,
            'portfolio_cumret': portfolio_cumret,
            'benchmark_cumret': benchmark_cumret,
            'rolling_sharpe': rolling_sharpe,
            'monthly_returns': monthly_returns,
            'years': years,
            'performance_metrics': [1.8, 2.1, 1.5, -0.12, 0.18]  # Sharpe, Sortino, Calmar, Max DD, Vol
        }
    
    def update_price_chart(self):
        """Update price chart with new symbol"""
        if hasattr(self, 'canvas') and self.canvas:
            # Get the parent frame
            parent_frame = self.canvas.get_tk_widget().master
            # Recreate the chart
            self.create_candlestick_chart(parent_frame)

# Standalone chart application
if __name__ == "__main__":
    root = tk.Tk()
    visualizer = QuantSphereChartVisualizer()
    chart_window = visualizer.create_main_chart_window()
    root.withdraw()  # Hide the root window
    chart_window.mainloop()
