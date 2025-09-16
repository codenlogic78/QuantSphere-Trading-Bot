#!/usr/bin/env python3
"""
QuantSphere AI Trading Platform - Enhanced Dashboard
Advanced portfolio analytics and visualization dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time

class EnhancedDashboard:
    """Enhanced dashboard with advanced analytics and visualizations"""
    
    def __init__(self, parent_frame, equities_data, trade_history):
        self.parent_frame = parent_frame
        self.equities_data = equities_data
        self.trade_history = trade_history
        
        # Create dashboard frame
        self.dashboard_frame = ttk.Frame(parent_frame)
        self.dashboard_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.dashboard_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_performance_tab()
        self.create_analytics_tab()
        self.create_risk_tab()
        self.create_alerts_tab()
        
        # Start real-time updates
        self.update_thread = threading.Thread(target=self.update_dashboard, daemon=True)
        self.update_thread.start()
    
    def create_performance_tab(self):
        """Create performance analytics tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance")
        
        # Performance metrics frame
        metrics_frame = ttk.LabelFrame(perf_frame, text="Portfolio Metrics")
        metrics_frame.pack(fill='x', padx=5, pady=5)
        
        # Create metrics labels
        self.total_value_label = ttk.Label(metrics_frame, text="Total Value: $0.00", font=('Arial', 12, 'bold'))
        self.total_value_label.grid(row=0, column=0, sticky='w', padx=10, pady=5)
        
        self.daily_pl_label = ttk.Label(metrics_frame, text="Daily P&L: $0.00")
        self.daily_pl_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        self.total_pl_label = ttk.Label(metrics_frame, text="Total P&L: $0.00")
        self.total_pl_label.grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        self.win_rate_label = ttk.Label(metrics_frame, text="Win Rate: 0%")
        self.win_rate_label.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        self.sharpe_ratio_label = ttk.Label(metrics_frame, text="Sharpe Ratio: 0.00")
        self.sharpe_ratio_label.grid(row=2, column=0, sticky='w', padx=10, pady=5)
        
        self.max_drawdown_label = ttk.Label(metrics_frame, text="Max Drawdown: 0%")
        self.max_drawdown_label.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        
        # Performance chart
        chart_frame = ttk.LabelFrame(perf_frame, text="Portfolio Performance Chart")
        chart_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.create_performance_chart(chart_frame)
    
    def create_analytics_tab(self):
        """Create analytics tab with detailed statistics"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="Analytics")
        
        # Position analysis
        pos_frame = ttk.LabelFrame(analytics_frame, text="Position Analysis")
        pos_frame.pack(fill='x', padx=5, pady=5)
        
        # Create treeview for position details
        columns = ("Symbol", "Shares", "Entry", "Current", "P&L", "P&L%", "Weight", "Risk Score")
        self.pos_tree = ttk.Treeview(pos_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.pos_tree.heading(col, text=col)
            self.pos_tree.column(col, width=80)
        
        self.pos_tree.pack(fill='x', padx=5, pady=5)
        
        # Trade history analysis
        history_frame = ttk.LabelFrame(analytics_frame, text="Recent Trades")
        history_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        trade_columns = ("Date", "Symbol", "Action", "Qty", "Price", "P&L", "Reason")
        self.trade_tree = ttk.Treeview(history_frame, columns=trade_columns, show='headings', height=10)
        
        for col in trade_columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=90)
        
        self.trade_tree.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_risk_tab(self):
        """Create risk management tab"""
        risk_frame = ttk.Frame(self.notebook)
        self.notebook.add(risk_frame, text="Risk Management")
        
        # Risk metrics
        risk_metrics_frame = ttk.LabelFrame(risk_frame, text="Risk Metrics")
        risk_metrics_frame.pack(fill='x', padx=5, pady=5)
        
        self.portfolio_beta_label = ttk.Label(risk_metrics_frame, text="Portfolio Beta: 0.00")
        self.portfolio_beta_label.grid(row=0, column=0, sticky='w', padx=10, pady=5)
        
        self.var_label = ttk.Label(risk_metrics_frame, text="Value at Risk (95%): $0.00")
        self.var_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        self.volatility_label = ttk.Label(risk_metrics_frame, text="Portfolio Volatility: 0%")
        self.volatility_label.grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        self.correlation_label = ttk.Label(risk_frame, text="Avg Correlation: 0.00")
        self.correlation_label.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Risk controls
        controls_frame = ttk.LabelFrame(risk_frame, text="Risk Controls")
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Max Position Size (%):").grid(row=0, column=0, padx=5, pady=5)
        self.max_position_var = tk.StringVar(value="20")
        ttk.Entry(controls_frame, textvariable=self.max_position_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Max Portfolio Risk (%):").grid(row=0, column=2, padx=5, pady=5)
        self.max_risk_var = tk.StringVar(value="15")
        ttk.Entry(controls_frame, textvariable=self.max_risk_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Update Risk Limits", command=self.update_risk_limits).grid(row=0, column=4, padx=10, pady=5)
        
        # Risk alerts
        alerts_frame = ttk.LabelFrame(risk_frame, text="Risk Alerts")
        alerts_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.risk_alerts_text = tk.Text(alerts_frame, height=10, state=tk.DISABLED)
        self.risk_alerts_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_alerts_tab(self):
        """Create alerts and notifications tab"""
        alerts_frame = ttk.Frame(self.notebook)
        self.notebook.add(alerts_frame, text="Alerts")
        
        # Alert configuration
        config_frame = ttk.LabelFrame(alerts_frame, text="Alert Configuration")
        config_frame.pack(fill='x', padx=5, pady=5)
        
        # Price alerts
        ttk.Label(config_frame, text="Price Change Alert (%):").grid(row=0, column=0, padx=5, pady=5)
        self.price_alert_var = tk.StringVar(value="5")
        ttk.Entry(config_frame, textvariable=self.price_alert_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # Volume alerts
        ttk.Label(config_frame, text="Volume Alert Multiplier:").grid(row=0, column=2, padx=5, pady=5)
        self.volume_alert_var = tk.StringVar(value="2")
        ttk.Entry(config_frame, textvariable=self.volume_alert_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(config_frame, text="Save Alert Settings", command=self.save_alert_settings).grid(row=0, column=4, padx=10, pady=5)
        
        # Active alerts display
        active_alerts_frame = ttk.LabelFrame(alerts_frame, text="Active Alerts")
        active_alerts_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.alerts_text = tk.Text(active_alerts_frame, height=15, state=tk.DISABLED)
        self.alerts_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_performance_chart(self, parent):
        """Create performance chart"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.fig.suptitle('Portfolio Performance')
        
        # Portfolio value chart
        self.ax1.set_title('Portfolio Value Over Time')
        self.ax1.set_ylabel('Value ($)')
        self.ax1.grid(True, alpha=0.3)
        
        # P&L chart
        self.ax2.set_title('Daily P&L')
        self.ax2.set_ylabel('P&L ($)')
        self.ax2.set_xlabel('Date')
        self.ax2.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        total_value = 0
        total_pl = 0
        daily_pl = 0
        
        for symbol, data in self.equities_data.items():
            if data['position'] > 0:
                current_price = data.get('current_price', data['entry_price'])
                position_value = data['position'] * current_price
                total_value += position_value
                
                pl = (current_price - data['entry_price']) * data['position']
                total_pl += pl
        
        # Calculate win rate
        if self.trade_history:
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pl', 0) > 0)
            win_rate = (winning_trades / len(self.trade_history)) * 100
        else:
            win_rate = 0
        
        # Calculate Sharpe ratio (simplified)
        if self.trade_history:
            returns = [trade.get('pl', 0) for trade in self.trade_history[-30:]]  # Last 30 trades
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Update labels
        self.total_value_label.config(text=f"Total Value: ${total_value:,.2f}")
        self.daily_pl_label.config(text=f"Daily P&L: ${daily_pl:,.2f}")
        self.total_pl_label.config(text=f"Total P&L: ${total_pl:,.2f}")
        self.win_rate_label.config(text=f"Win Rate: {win_rate:.1f}%")
        self.sharpe_ratio_label.config(text=f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Calculate max drawdown
        if self.trade_history:
            cumulative_pl = np.cumsum([trade.get('pl', 0) for trade in self.trade_history])
            running_max = np.maximum.accumulate(cumulative_pl)
            drawdown = (cumulative_pl - running_max) / running_max * 100
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        self.max_drawdown_label.config(text=f"Max Drawdown: {max_drawdown:.1f}%")
    
    def update_position_analysis(self):
        """Update position analysis table"""
        # Clear existing items
        for item in self.pos_tree.get_children():
            self.pos_tree.delete(item)
        
        total_value = sum(data['position'] * data.get('current_price', data['entry_price']) 
                         for data in self.equities_data.values() if data['position'] > 0)
        
        for symbol, data in self.equities_data.items():
            if data['position'] > 0:
                current_price = data.get('current_price', data['entry_price'])
                position_value = data['position'] * current_price
                pl = (current_price - data['entry_price']) * data['position']
                pl_pct = (pl / (data['entry_price'] * data['position'])) * 100
                weight = (position_value / total_value) * 100 if total_value > 0 else 0
                
                # Simple risk score based on volatility and position size
                risk_score = min(10, weight / 5 + abs(pl_pct) / 10)
                
                self.pos_tree.insert("", "end", values=(
                    symbol,
                    data['position'],
                    f"${data['entry_price']:.2f}",
                    f"${current_price:.2f}",
                    f"${pl:.2f}",
                    f"{pl_pct:.1f}%",
                    f"{weight:.1f}%",
                    f"{risk_score:.1f}"
                ))
    
    def update_trade_history(self):
        """Update trade history table"""
        # Clear existing items
        for item in self.trade_tree.get_children():
            self.trade_tree.delete(item)
        
        # Show last 20 trades
        recent_trades = self.trade_history[-20:] if len(self.trade_history) > 20 else self.trade_history
        
        for trade in reversed(recent_trades):
            self.trade_tree.insert("", "end", values=(
                trade.get('timestamp', ''),
                trade.get('symbol', ''),
                trade.get('action', ''),
                trade.get('quantity', 0),
                f"${trade.get('price', 0):.2f}",
                f"${trade.get('pl', 0):.2f}",
                trade.get('reason', '')[:30]  # Truncate reason
            ))
    
    def update_risk_metrics(self):
        """Update risk management metrics"""
        # Calculate portfolio volatility (simplified)
        if len(self.trade_history) > 10:
            returns = [trade.get('pl', 0) for trade in self.trade_history[-30:]]
            volatility = np.std(returns) * np.sqrt(252) if returns else 0  # Annualized
        else:
            volatility = 0
        
        # Calculate Value at Risk (95% confidence)
        if len(self.trade_history) > 20:
            returns = [trade.get('pl', 0) for trade in self.trade_history[-50:]]
            var_95 = np.percentile(returns, 5) if returns else 0  # 5th percentile
        else:
            var_95 = 0
        
        self.volatility_label.config(text=f"Portfolio Volatility: {volatility:.1f}%")
        self.var_label.config(text=f"Value at Risk (95%): ${var_95:.2f}")
        
        # Check for risk alerts
        self.check_risk_alerts()
    
    def check_risk_alerts(self):
        """Check for risk alerts and update display"""
        alerts = []
        
        # Check position concentration
        total_value = sum(data['position'] * data.get('current_price', data['entry_price']) 
                         for data in self.equities_data.values() if data['position'] > 0)
        
        max_position_pct = float(self.max_position_var.get())
        
        for symbol, data in self.equities_data.items():
            if data['position'] > 0:
                current_price = data.get('current_price', data['entry_price'])
                position_value = data['position'] * current_price
                weight = (position_value / total_value) * 100 if total_value > 0 else 0
                
                if weight > max_position_pct:
                    alerts.append(f"⚠️ {symbol}: Position size {weight:.1f}% exceeds limit {max_position_pct}%")
                
                # Check for large losses
                pl_pct = ((current_price - data['entry_price']) / data['entry_price']) * 100
                if pl_pct < -10:  # 10% loss threshold
                    alerts.append(f"🔴 {symbol}: Large loss {pl_pct:.1f}%")
        
        # Update risk alerts display
        self.risk_alerts_text.config(state=tk.NORMAL)
        self.risk_alerts_text.delete(1.0, tk.END)
        
        if alerts:
            for alert in alerts:
                self.risk_alerts_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {alert}\n")
        else:
            self.risk_alerts_text.insert(tk.END, "No active risk alerts.\n")
        
        self.risk_alerts_text.config(state=tk.DISABLED)
    
    def update_performance_chart(self):
        """Update performance charts"""
        if not self.trade_history:
            return
        
        # Prepare data
        dates = [datetime.strptime(trade['timestamp'], '%Y-%m-%d %H:%M:%S') for trade in self.trade_history]
        cumulative_pl = np.cumsum([trade.get('pl', 0) for trade in self.trade_history])
        daily_pl = [trade.get('pl', 0) for trade in self.trade_history]
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Portfolio value chart
        self.ax1.plot(dates, cumulative_pl, 'b-', linewidth=2)
        self.ax1.set_title('Cumulative P&L Over Time')
        self.ax1.set_ylabel('Cumulative P&L ($)')
        self.ax1.grid(True, alpha=0.3)
        
        # Daily P&L chart
        colors = ['green' if pl >= 0 else 'red' for pl in daily_pl]
        self.ax2.bar(dates, daily_pl, color=colors, alpha=0.7)
        self.ax2.set_title('Daily P&L')
        self.ax2.set_ylabel('P&L ($)')
        self.ax2.set_xlabel('Date')
        self.ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        self.fig.autofmt_xdate()
        
        # Refresh canvas
        self.canvas.draw()
    
    def update_risk_limits(self):
        """Update risk management limits"""
        try:
            max_pos = float(self.max_position_var.get())
            max_risk = float(self.max_risk_var.get())
            
            # Save to configuration
            config = {
                'max_position_pct': max_pos,
                'max_portfolio_risk': max_risk,
                'updated': datetime.now().isoformat()
            }
            
            with open('risk_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", "Risk limits updated successfully!")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
    
    def save_alert_settings(self):
        """Save alert configuration"""
        try:
            price_alert = float(self.price_alert_var.get())
            volume_alert = float(self.volume_alert_var.get())
            
            config = {
                'price_change_alert': price_alert,
                'volume_alert_multiplier': volume_alert,
                'updated': datetime.now().isoformat()
            }
            
            with open('alert_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", "Alert settings saved successfully!")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
    
    def update_dashboard(self):
        """Main dashboard update loop"""
        while True:
            try:
                self.update_performance_metrics()
                self.update_position_analysis()
                self.update_trade_history()
                self.update_risk_metrics()
                self.update_performance_chart()
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"Dashboard update error: {e}")
                time.sleep(30)  # Wait longer if there's an error

if __name__ == "__main__":
    # Test the enhanced dashboard
    root = tk.Tk()
    root.title("QuantSphere Enhanced Dashboard Test")
    root.geometry("1200x800")
    
    # Mock data for testing
    mock_equities = {
        'AAPL': {'position': 100, 'entry_price': 150.0, 'current_price': 155.0},
        'MSFT': {'position': 50, 'entry_price': 300.0, 'current_price': 295.0}
    }
    
    mock_trades = [
        {'timestamp': '2024-01-01 10:00:00', 'symbol': 'AAPL', 'action': 'BUY', 'quantity': 100, 'price': 150.0, 'pl': 0},
        {'timestamp': '2024-01-02 11:00:00', 'symbol': 'MSFT', 'action': 'BUY', 'quantity': 50, 'price': 300.0, 'pl': 0},
        {'timestamp': '2024-01-03 14:00:00', 'symbol': 'AAPL', 'action': 'SELL', 'quantity': 50, 'price': 155.0, 'pl': 250.0}
    ]
    
    dashboard = EnhancedDashboard(root, mock_equities, mock_trades)
    root.mainloop()
