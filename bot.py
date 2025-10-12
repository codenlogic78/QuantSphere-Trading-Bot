import tkinter as tk
from tkinter import ttk, messagebox
import json
import time
import threading
import random
import alpaca_trade_api as tradeapi
import openai
import os
from dotenv import load_dotenv
from technical_indicators import TechnicalIndicators

# Import Hybrid AI Integration
try:
    from hybrid_ai_integration import QuantSphereHybridAI, get_hybrid_analysis_for_gui
    HYBRID_AI_AVAILABLE = True
    print("🧠 Hybrid AI system loaded successfully!")
except ImportError as e:
    print(f"⚠️ Hybrid AI not available: {e}")
    HYBRID_AI_AVAILABLE = False

# Load environment variables
load_dotenv()

DATA_FILE = "quantsphere_portfolio.json"

# Configure your Alpaca API credentials here
key = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets/"
api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")

def fetch_portfolio():
    positions = api.list_positions()
    portfolio = []
    for pos in positions:
        portfolio.append({
            'symbol':pos.symbol,
            'qty':pos.qty,
            'entry_price':pos.avg_entry_price,
            'current_price':pos.current_price,
            'unrealized_pl':pos.unrealized_pl,
            'side': 'buy'
        })
    return portfolio

def fetch_open_orders():
    orders = api.list_orders(status='open')
    open_orders = []
    for order in orders:
        open_orders.append({
            'symbol':order.symbol,
            'qty':order.qty,
            'limit_price':order.limit_price,
            'side': 'buy'
        })

def fetch_real_price(symbol):
    """Fetch real-time price from Alpaca API"""
    try:
        latest_trade = api.get_latest_trade(symbol)
        return {"price": float(latest_trade.price)}
    except Exception as e:
        print(f"Error fetching real price for {symbol}: {e}")
        # Fallback to mock data if API fails
        import random
        return {"price": round(100 + random.uniform(-10, 10), 2)}

def fetch_mock_api(symbol):
    """Legacy function - now uses real price data"""
    return fetch_real_price(symbol)

def chatgpt_response(message):
    portfolio_data = fetch_portfolio()
    open_orders = fetch_open_orders()

    pre_prompt = f"""
    You are QuantSphere AI, an advanced portfolio manager responsible for analyzing trading portfolios.
    Your tasks are the following:
    1.) Evaluate risk exposures of my current holdings
    2.) Analyze my open limit orders and their potential impact
    3.) provide insights into portfolio health, diversification, trade adj. etc.
    4.) Speculate on the market outlook based on current market conditions
    5.) Identify potential market risks and suggest risk management strategies

    Here is my portfolio: {portfolio_data}

    Here are my open orders {open_orders}

    Overall, answer the following question with priority having that background: {message}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system", "content":pre_prompt}],
            api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Analysis Error: {str(e)}. Please check your OpenAI API key and try again."

class QuantSphereGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("QuantSphere AI Trading Platform")
        self.equities = self.load_equities()
        self.system_running = False

        self.form_frame = tk.Frame(root)
        self.form_frame.pack(pady=10)

        # Form to add a new equity to QuantSphere platform
        tk.Label(self.form_frame, text="Symbol:").grid(row=0, column=0)
        self.symbol_entry = tk.Entry(self.form_frame)
        self.symbol_entry.grid(row=0, column=1)

        tk.Label(self.form_frame, text="Levels:").grid(row=0, column=2)
        self.levels_entry = tk.Entry(self.form_frame)
        self.levels_entry.grid(row=0, column=3)

        tk.Label(self.form_frame, text="Drawdown%:").grid(row=0, column=4)
        self.drawdown_entry = tk.Entry(self.form_frame)
        self.drawdown_entry.grid(row=0, column=5)

        tk.Label(self.form_frame, text="Stop Loss%:").grid(row=1, column=0)
        self.stop_loss_entry = tk.Entry(self.form_frame)
        self.stop_loss_entry.grid(row=1, column=1)

        tk.Label(self.form_frame, text="Take Profit%:").grid(row=1, column=2)
        self.take_profit_entry = tk.Entry(self.form_frame)
        self.take_profit_entry.grid(row=1, column=3)

        self.add_button = tk.Button(self.form_frame, text="Add Equity", command=self.add_equity)
        self.add_button.grid(row=0, column=6)

        # Table to track the traded equities
        self.tree = ttk.Treeview(root, columns=("Symbol", "Position", "Entry Price", "Current Price", "P&L", "RSI", "Signal", "Stop Loss", "Take Profit", "Status"), show='headings')
        for col in ["Symbol", "Position", "Entry Price", "Current Price", "P&L", "RSI", "Signal", "Stop Loss", "Take Profit", "Status"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=85)
        self.tree.pack(pady=10)

        # Buttons to control the bot
        self.toggle_system_button = tk.Button(root, text="Toggle Selected System", command=self.toggle_selected_system)
        self.toggle_system_button.pack(pady=5)

        self.remove_button = tk.Button(root, text="Remove Selected Equity", command=self.remove_selected_equity)
        self.remove_button.pack(pady=5)

        # Charts button
        self.charts_button = tk.Button(root, text="Open Advanced Charts", command=self.open_charts)
        self.charts_button.pack(pady=5)
        
        # Hybrid AI Controls
        if HYBRID_AI_AVAILABLE:
            self.hybrid_frame = tk.Frame(root)
            self.hybrid_frame.pack(pady=10)
            
            tk.Label(self.hybrid_frame, text="🧠 Hybrid AI Analysis", font=('Arial', 12, 'bold')).pack()
            
            self.hybrid_buttons_frame = tk.Frame(self.hybrid_frame)
            self.hybrid_buttons_frame.pack(pady=5)
            
            self.hybrid_analysis_button = tk.Button(self.hybrid_buttons_frame, text="🔍 Analyze Selected Stock", 
                                                   command=self.show_hybrid_analysis, bg='lightblue')
            self.hybrid_analysis_button.pack(side=tk.LEFT, padx=5)
            
            self.portfolio_recommendations_button = tk.Button(self.hybrid_buttons_frame, text="📊 Portfolio Recommendations", 
                                                            command=self.show_portfolio_recommendations, bg='lightgreen')
            self.portfolio_recommendations_button.pack(side=tk.LEFT, padx=5)
            
            self.market_insights_button = tk.Button(self.hybrid_buttons_frame, text="🌍 Market Insights", 
                                                   command=self.show_market_insights, bg='lightyellow')
            self.market_insights_button.pack(side=tk.LEFT, padx=5)

        # AI Component
        self.chat_frame = tk.Frame(root)
        self.chat_frame.pack(pady=10)

        self.chat_input = tk.Entry(self.chat_frame, width=50)
        self.chat_input.grid(row=0, column=0, padx=5)

        self.send_button = tk.Button(self.chat_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1)

        self.chat_output = tk.Text(root, height=5, width=60, state=tk.DISABLED)
        self.chat_output.pack()

        # Portfolio Performance Dashboard
        self.performance_frame = tk.Frame(root)
        self.performance_frame.pack(pady=10, fill='x')
        
        tk.Label(self.performance_frame, text="Portfolio Performance", font=('Arial', 12, 'bold')).pack()
        
        self.performance_info = tk.Frame(self.performance_frame)
        self.performance_info.pack(fill='x', padx=10)
        
        # Performance metrics labels
        self.total_value_label = tk.Label(self.performance_info, text="Total Portfolio Value: $0.00", font=('Arial', 10))
        self.total_value_label.grid(row=0, column=0, sticky='w', padx=5)
        
        self.total_pl_label = tk.Label(self.performance_info, text="Total P&L: $0.00", font=('Arial', 10))
        self.total_pl_label.grid(row=0, column=1, sticky='w', padx=5)
        
        self.active_positions_label = tk.Label(self.performance_info, text="Active Positions: 0", font=('Arial', 10))
        self.active_positions_label.grid(row=1, column=0, sticky='w', padx=5)
        
        self.win_rate_label = tk.Label(self.performance_info, text="Win Rate: 0%", font=('Arial', 10))
        self.win_rate_label.grid(row=1, column=1, sticky='w', padx=5)

        # Initialize technical indicators
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize Hybrid AI system
        if HYBRID_AI_AVAILABLE:
            hybrid_config = {
                'openai_api_key': os.getenv("OPENAI_API_KEY"),
                'performance_file': 'quantsphere_hybrid_performance.json'
            }
            self.hybrid_ai = QuantSphereHybridAI(hybrid_config)
            print("🧠 Hybrid AI integration initialized")
        else:
            self.hybrid_ai = None
        
        # Load saved data
        self.trade_history = self.load_trade_history()
        self.refresh_table()
        self.update_performance_metrics()

        # Auto-refreshing
        self.running = True
        self.auto_update_thread = threading.Thread(target=self.auto_update, daemon=True)
        self.auto_update_thread.start()


    def add_equity(self):
        symbol = self.symbol_entry.get().upper()
        levels = self.levels_entry.get()
        drawdown = self.drawdown_entry.get()
        stop_loss = self.stop_loss_entry.get() or "10"
        take_profit = self.take_profit_entry.get() or "20"

        if not symbol or not levels.isdigit() or not drawdown.replace('.', '', 1).isdigit():
            messagebox.showerror("Error", "Invalid Input")
            return
        
        if not stop_loss.replace('.', '', 1).isdigit() or not take_profit.replace('.', '', 1).isdigit():
            messagebox.showerror("Error", "Invalid Stop Loss or Take Profit")
            return
        
        levels = int(levels)
        drawdown = float(drawdown) / 100
        stop_loss_pct = float(stop_loss) / 100
        take_profit_pct = float(take_profit) / 100
        entry_price = fetch_real_price(symbol)['price']

        level_prices = {i+1 : round(entry_price * (1-drawdown*(i+1)), 2) for i in range(levels)}
        stop_loss_price = round(entry_price * (1 - stop_loss_pct), 2)
        take_profit_price = round(entry_price * (1 + take_profit_pct), 2)

        self.equities[symbol] = {
            "position":0,
            "entry_price":entry_price,
            "current_price":entry_price,
            "levels":level_prices,
            "drawdown": drawdown,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "unrealized_pl": 0.0,
            "technical_signal": "HOLD",
            "rsi": 50.0,
            "status": "Off"
        }
        self.save_equities()
        self.refresh_table()
        
        # Clear input fields
        self.symbol_entry.delete(0, tk.END)
        self.levels_entry.delete(0, tk.END)
        self.drawdown_entry.delete(0, tk.END)
        self.stop_loss_entry.delete(0, tk.END)
        self.take_profit_entry.delete(0, tk.END)

    def toggle_selected_system(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No Equity is Selected")
            return
        
        for item in selected_items:
            symbol = self.tree.item(item)['values'][0]
            self.equities[symbol]['status'] = "On" if self.equities[symbol]['status'] == "Off" else "Off"

        self.save_equities()
        self.refresh_table()
    
    def remove_selected_equity(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No Equity Selected")
            return
        
        for item in selected_items:
            symbol = self.tree.item(item)['values'][0]
            if symbol in self.equities:
                del self.equities[symbol]
        
        self.save_equities()
        self.refresh_table()

    def send_message(self):
        message = self.chat_input.get()
        if not message:
            return
        
        response = chatgpt_response(message)

        self.chat_output.config(state=tk.NORMAL)
        self.chat_output.insert(tk.END, f"You: {message}\n{response}\n\n")
        self.chat_output.config(state=tk.DISABLED)
        self.chat_input.delete(0, tk.END)

    def fetch_alpaca_data(self, symbol):
        try:
            barset = api.get_latest_trade(symbol)
            return {"price":barset.price}
        except Exception as e:
            return {"price":-1}

    def check_existing_orders(self, symbol, price):
        try:
            orders = api.list_orders(status='open', symbols=symbol)
            for order in orders:
                if float(order.limit_price) == price:
                    return True
        except Exception as e:
            messagebox.showerror("API Error", f"Error Checking Orders {e}")
        return False

    def get_max_entry_price(self, symbol):
        try:
            orders = api.list_orders(status="filled", limit=50)
            prices = [float(order.filled_avg_price) for order in orders if order.filled_avg_price and order.symbol == symbol]
            return max(prices) if prices else -1
        except Exception as e:
            messagebox.showerror("API Error", f"Error Fetching Orders {e}")
            return 0

    def trade_systems(self):
        """Execute trading logic for active systems"""
        for symbol, data in self.equities.items():
            if data['status'] == 'On':
                try:
                    # Get current price and technical signals
                    current_price = fetch_real_price(symbol)['price']
                    self.equities[symbol]['current_price'] = current_price
                    
                    # Get technical analysis signals
                    try:
                        from technical_indicators import TechnicalIndicators
                        tech_indicators = TechnicalIndicators()
                        signals = tech_indicators.generate_signals(symbol)
                        
                        self.equities[symbol]['technical_signal'] = signals['signal']
                        self.equities[symbol]['rsi'] = signals['indicators']['rsi']
                        
                        # Simple trading logic based on signals
                        if signals['signal'] in ['BUY', 'STRONG_BUY'] and data['position'] == 0:
                            # Simulate buying 10 shares
                            self.equities[symbol]['position'] = 10
                            self.equities[symbol]['entry_price'] = current_price
                            
                            # Log the trade
                            self.log_trade(symbol, "BUY", 10, current_price, "TECHNICAL_SIGNAL", 0)
                            print(f"📈 BUY signal executed for {symbol} at ${current_price:.2f}")
                            
                        elif signals['signal'] in ['SELL', 'STRONG_SELL'] and data['position'] > 0:
                            # Simulate selling position
                            position_qty = data['position']
                            realized_pl = (current_price - data['entry_price']) * position_qty
                            
                            self.equities[symbol]['position'] = 0
                            
                            # Log the trade
                            self.log_trade(symbol, "SELL", position_qty, current_price, "TECHNICAL_SIGNAL", realized_pl)
                            print(f"📉 SELL signal executed for {symbol} at ${current_price:.2f}, P&L: ${realized_pl:.2f}")
                    
                    except Exception as tech_error:
                        print(f"⚠️ Technical analysis error for {symbol}: {tech_error}")
                        # Use mock technical data as fallback
                        self.equities[symbol]['technical_signal'] = 'HOLD'
                        self.equities[symbol]['rsi'] = 50.0
                    
                    # Check stop-loss and take-profit
                    self.check_exit_conditions(symbol, current_price)
                    
                except Exception as e:
                    print(f"❌ Trading system error for {symbol}: {e}")
                    # Use mock price as fallback
                    mock_price = data.get('current_price', data['entry_price'])
                    self.equities[symbol]['current_price'] = mock_price * (1 + random.uniform(-0.02, 0.02))
        
        self.save_equities()
        self.refresh_table()
        self.update_performance_metrics()

    def place_order(self, symbol, price, level):
        if -level in self.equities[symbol]['levels'] or '-1' in self.equities[symbol]['levels'].keys():
            return
        
        try:
            api.submit_order(
                symbol=symbol,
                qty=1,
                side='buy',
                type='limit',
                time_in_force='gtc',
                limit_price=price
            )
            self.equities[symbol]['levels'][-level] = price
            del self.equities[symbol]['levels'][level]
            print(f"Placed order for {symbol}@{price}")
        except Exception as e:
            messagebox.showerror("Order Error", f"Error placing order {e}")

    def refresh_table(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Populate table with current equity data
        for symbol, data in self.equities.items():
            current_price = data.get('current_price', data['entry_price'])
            unrealized_pl = (current_price - data['entry_price']) * data['position']
            
            # Insert row with all data
            item_id = self.tree.insert("", "end", values=(
                symbol,
                data['position'],
                f"${data['entry_price']:.2f}",
                f"${current_price:.2f}",
                f"${unrealized_pl:.2f}",
                f"{data.get('rsi', 50):.1f}",
                data.get('technical_signal', 'HOLD'),
                f"${data.get('stop_loss_price', 0):.2f}",
                f"${data.get('take_profit_price', 0):.2f}",
                data['status']
            ))
            
            # Color code based on status
            if data['status'] == 'On':
                self.tree.set(item_id, "Status", "🟢 On")
            else:
                self.tree.set(item_id, "Status", "🔴 Off")
    
    def check_exit_conditions(self, symbol, current_price):
        """Check stop-loss and take-profit conditions"""
        equity = self.equities[symbol]
        
        # Only check if we have a position
        if equity['position'] <= 0:
            return
        
        stop_loss_price = equity.get('stop_loss_price', 0)
        take_profit_price = equity.get('take_profit_price', 0)
        
        # Check stop-loss trigger
        if stop_loss_price > 0 and current_price <= stop_loss_price:
            self.execute_exit_order(symbol, "STOP_LOSS", current_price)
        
        # Check take-profit trigger
        if take_profit_price > 0 and current_price >= take_profit_price:
            self.execute_exit_order(symbol, "TAKE_PROFIT", current_price)
    
    def execute_exit_order(self, symbol, reason, price):
        """Execute stop-loss or take-profit order"""
        equity = self.equities[symbol]
        position_qty = equity['position']
        
        if position_qty <= 0:
            return
        
        try:
            # Submit market sell order for immediate execution
            submit_order(symbol, position_qty, "sell")
            
            # Calculate realized P&L
            realized_pl = (price - equity['entry_price']) * position_qty
            
            # Log the trade
            self.log_trade(symbol, "SELL", position_qty, price, reason, realized_pl)
            
            # Reset position
            equity['position'] = 0
            equity['entry_price'] = price
            
            # Show notification
            messagebox.showinfo(
                f"{reason} Executed",
                f"Symbol: {symbol}\n"
                f"Price: ${price:.2f}\n"
                f"Position: {position_qty} shares sold\n"
                f"Realized P&L: ${realized_pl:.2f}"
            )
            
            print(f"{reason} executed for {symbol} at ${price:.2f}, P&L: ${realized_pl:.2f}")
            
        except Exception as e:
            messagebox.showerror("Order Error", f"Failed to execute {reason} for {symbol}: {e}")
    def update_current_prices(self):
        """Update current prices and technical indicators for all tracked equities"""
        for symbol in self.equities.keys():
            if self.equities[symbol]['status'] == 'On':
                try:
                    # Update current price
                    current_price_data = fetch_real_price(symbol)
                    current_price = current_price_data['price']
                    self.equities[symbol]['current_price'] = current_price
                    
                    # Update technical indicators
                    tech_signals = self.technical_indicators.generate_signals(symbol)
                    self.equities[symbol]['technical_signal'] = tech_signals['signal']
                    self.equities[symbol]['rsi'] = tech_signals['indicators'].get('rsi', 50)
                    
                    # Check for stop-loss/take-profit triggers
                    self.check_stop_loss_take_profit(symbol, current_price)
                    
                    # Run trading logic
                    self.trading_logic(symbol, current_price)
                    
                except Exception as e:
                    print(f"Error updating price/indicators for {symbol}: {e}")
        
        # Save updated prices
        self.save_equities()
        # Refresh the display
        self.refresh_table()
        # Update performance metrics
        self.update_performance_metrics()
    
    def auto_update(self):
        while self.running:
            # Update prices every 5 seconds
            self.update_current_prices()
            # Check trading systems
            self.trade_systems()
            time.sleep(5)
    
    def save_equities(self):
        with open(DATA_FILE, 'w') as f:
            json.dump(self.equities, f)
    
    def load_equities(self):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def trading_logic(self, symbol, current_price):
        """Enhanced trading logic with technical indicators and optional hybrid AI"""
        equity = self.equities[symbol]
        
        # Get technical signals
        try:
            tech_signals = self.technical_indicators.generate_signals(symbol)
            signal = tech_signals['signal']
            strength = tech_signals['strength']
            indicators = tech_signals['indicators']
            
            # Update equity with technical data
            equity['technical_signal'] = signal
            equity['rsi'] = indicators.get('rsi', 50)
            
            print(f"Technical analysis for {symbol}: Signal={signal}, RSI={equity['rsi']:.1f}, Price=${current_price:.2f}")
            
        except Exception as e:
            print(f"Error getting technical signals for {symbol}: {e}")
            signal = 'HOLD'
            strength = 0
            equity['technical_signal'] = 'HOLD'
            equity['rsi'] = 50
        
        # Optionally enhance with Hybrid AI (if available and enabled)
        hybrid_signal = signal  # Default to technical signal
        hybrid_confidence = 0.5
        
        if HYBRID_AI_AVAILABLE and self.hybrid_ai and hasattr(equity, 'use_hybrid_ai') and equity.get('use_hybrid_ai', False):
            try:
                # Get hybrid AI analysis (cached for performance)
                hybrid_analysis = self.hybrid_ai.get_enhanced_analysis(symbol)
                hybrid_signal = hybrid_analysis['hybrid_recommendation']
                hybrid_confidence = hybrid_analysis['confidence']
                
                print(f"🧠 Hybrid AI for {symbol}: {hybrid_signal} (confidence: {hybrid_confidence:.2f})")
                
                # Use hybrid signal if confidence is high
                if hybrid_confidence > 0.7:
                    signal = hybrid_signal
                    strength = hybrid_confidence
                    print(f"✅ Using Hybrid AI signal for {symbol}: {signal}")
                
            except Exception as e:
                print(f"⚠️ Hybrid AI error for {symbol}: {e}")
                # Fall back to technical analysis
        
        # Enhanced trading logic combining price levels and technical signals
        if equity['position'] == 0:  # No position
            # Look for BUY signals
            if signal == 'BUY' and strength > 0:
                # Check if current price is at or below any buy level
                for level in equity['levels']:
                    if current_price <= level * (1 + 0.01):  # 1% tolerance
                        try:
                            shares_to_buy = int(10000 / current_price)  # $10k position
                            if shares_to_buy > 0:
                                submit_order(symbol, shares_to_buy, "buy")
                                equity['position'] += shares_to_buy
                                equity['entry_price'] = current_price
                                
                                # Log the trade
                                self.log_trade(symbol, "BUY", shares_to_buy, current_price, 
                                             f"Technical Signal: {signal}, RSI: {equity['rsi']:.1f}", 0)
                                
                                print(f"Technical BUY executed: {symbol} x {shares_to_buy} @ ${current_price:.2f}")
                                break
                        except Exception as e:
                            print(f"Error executing technical buy for {symbol}: {e}")
        
        elif equity['position'] > 0:  # Have position
            # Look for SELL signals or risk management triggers
            if signal == 'SELL' and strength > 0:
                try:
                    submit_order(symbol, equity['position'], "sell")
                    
                    # Calculate P&L
                    realized_pl = (current_price - equity['entry_price']) * equity['position']
                    
                    # Log the trade
                    self.log_trade(symbol, "SELL", equity['position'], current_price, 
                                 f"Technical Signal: {signal}, P&L: ${realized_pl:.2f}", realized_pl)
                    
                    print(f"Technical SELL executed: {symbol} x {equity['position']} @ ${current_price:.2f}, P&L: ${realized_pl:.2f}")
                    
                    # Reset position
                    equity['position'] = 0
                    equity['entry_price'] = current_price
                    
                except Exception as e:
                    print(f"Error executing technical sell for {symbol}: {e}")
    
    def log_trade(self, symbol, action, quantity, price, reason, pl):
        """Log trade to history for performance tracking"""
        trade = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "reason": reason,
            "pl": pl
        }
        self.trade_history.append(trade)
        self.save_trade_history()
    
    def load_trade_history(self):
        """Load trade history from file"""
        try:
            with open("quantsphere_trades.json", 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_trade_history(self):
        """Save trade history to file"""
        with open("quantsphere_trades.json", 'w') as f:
            json.dump(self.trade_history, f, indent=2)
    
    def update_performance_metrics(self):
        """Update portfolio performance metrics display"""
        # Calculate total portfolio value and P&L
        total_value = 0
        total_pl = 0
        active_positions = 0
        
        for symbol, data in self.equities.items():
            if data['position'] > 0:
                active_positions += 1
                position_value = data['position'] * data.get('current_price', data['entry_price'])
                total_value += position_value
                
                current_price = data.get('current_price', data['entry_price'])
                pl = (current_price - data['entry_price']) * data['position']
                total_pl += pl
        
        # Calculate win rate from trade history
        if self.trade_history:
            winning_trades = sum(1 for trade in self.trade_history if trade.get('pl', 0) > 0)
            win_rate = (winning_trades / len(self.trade_history)) * 100
        else:
            win_rate = 0
        
        # Add cash value (assume $10,000 starting cash minus invested amount)
        starting_cash = 10000
        invested_amount = sum(data['position'] * data['entry_price'] for data in self.equities.values() if data['position'] > 0)
        available_cash = max(0, starting_cash - invested_amount)
        total_portfolio_value = total_value + available_cash
        
        # Update labels
        self.total_value_label.config(text=f"Total Portfolio Value: ${total_portfolio_value:.2f}")
        self.total_pl_label.config(text=f"Total P&L: ${total_pl:.2f}")
        self.active_positions_label.config(text=f"Active Positions: {active_positions}")
        self.win_rate_label.config(text=f"Win Rate: {win_rate:.1f}%")

    def open_charts(self):
        """Open advanced charts window"""
        try:
            from chart_visualizer import QuantSphereChartVisualizer
            visualizer = QuantSphereChartVisualizer(self.root)
            visualizer.create_main_chart_window()
        except Exception as e:
            messagebox.showerror("Charts Error", f"Error opening charts: {e}")
    
    def show_hybrid_analysis(self):
        """Show hybrid AI analysis for selected stock"""
        if not HYBRID_AI_AVAILABLE or not self.hybrid_ai:
            messagebox.showwarning("Hybrid AI", "Hybrid AI system not available")
            return
        
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Selection", "Please select a stock to analyze")
            return
        
        symbol = self.tree.item(selected_items[0])['values'][0]
        
        # Show loading message
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Analyzing...")
        loading_window.geometry("300x100")
        tk.Label(loading_window, text=f"🧠 Analyzing {symbol} with Hybrid AI...", 
                font=('Arial', 12)).pack(expand=True)
        loading_window.update()
        
        try:
            # Get hybrid analysis
            analysis = self.hybrid_ai.get_enhanced_analysis(symbol)
            loading_window.destroy()
            
            # Create analysis window
            self.create_hybrid_analysis_window(symbol, analysis)
            
        except Exception as e:
            loading_window.destroy()
            messagebox.showerror("Analysis Error", f"Error analyzing {symbol}: {e}")
    
    def create_hybrid_analysis_window(self, symbol, analysis):
        """Create window to display hybrid analysis results"""
        window = tk.Toplevel(self.root)
        window.title(f"🧠 Hybrid AI Analysis - {symbol}")
        window.geometry("800x600")
        
        # Create scrollable frame
        canvas = tk.Canvas(window)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header
        header_frame = tk.Frame(scrollable_frame, bg='navy')
        header_frame.pack(fill='x', pady=5)
        tk.Label(header_frame, text=f"🧠 Hybrid AI Analysis: {symbol}", 
                font=('Arial', 16, 'bold'), fg='white', bg='navy').pack(pady=10)
        
        # Overall Recommendation
        rec_frame = tk.Frame(scrollable_frame, relief='raised', bd=2)
        rec_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(rec_frame, text="📊 HYBRID RECOMMENDATION", 
                font=('Arial', 14, 'bold')).pack()
        
        rec_text = f"Recommendation: {analysis['hybrid_recommendation']}\n"
        rec_text += f"Confidence: {analysis['confidence']:.2f}\n"
        rec_text += f"Expected Alpha: {analysis['expected_alpha']:.2%}\n"
        rec_text += f"Risk Score: {analysis['risk_score']:.2f}"
        
        tk.Label(rec_frame, text=rec_text, font=('Arial', 12), justify='left').pack(pady=5)
        
        # AI Analysis Section
        ai_frame = tk.Frame(scrollable_frame, relief='raised', bd=2)
        ai_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(ai_frame, text="🤖 AI ANALYSIS", font=('Arial', 14, 'bold')).pack()
        
        ai_text = f"Recommendation: {analysis['ai_analysis']['recommendation']}\n"
        ai_text += f"Confidence: {analysis['ai_analysis']['confidence']:.2f}\n"
        ai_text += f"Technical Signal: {analysis['ai_analysis']['technical_signal']}\n"
        ai_text += f"Price Prediction: ${analysis['ai_analysis']['price_prediction']:.2f}\n"
        ai_text += f"Risk Score: {analysis['ai_analysis']['risk_score']:.2f}"
        
        tk.Label(ai_frame, text=ai_text, font=('Arial', 10), justify='left').pack(pady=5)
        
        # Technical Reasoning
        if analysis['ai_analysis']['reasoning']:
            tk.Label(ai_frame, text="Technical Reasoning:", font=('Arial', 10, 'bold')).pack(anchor='w')
            for reason in analysis['ai_analysis']['reasoning']:
                tk.Label(ai_frame, text=f"• {reason}", font=('Arial', 9), 
                        wraplength=700, justify='left').pack(anchor='w', padx=20)
        
        # Human Analysis Section
        human_frame = tk.Frame(scrollable_frame, relief='raised', bd=2)
        human_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(human_frame, text="🧠 HUMAN-LIKE AI ANALYSIS", font=('Arial', 14, 'bold')).pack()
        
        human_text = f"Recommendation: {analysis['human_analysis']['recommendation']}\n"
        human_text += f"Confidence: {analysis['human_analysis']['confidence']:.2f}\n"
        human_text += f"Company Assessment: {analysis['human_analysis']['company_assessment'][:100]}...\n"
        human_text += f"Market Sentiment: {analysis['human_analysis']['market_sentiment'][:100]}..."
        
        tk.Label(human_frame, text=human_text, font=('Arial', 10), justify='left', 
                wraplength=700).pack(pady=5)
        
        # Opportunities and Risks
        if analysis['human_analysis']['opportunities']:
            tk.Label(human_frame, text="💡 Key Opportunities:", font=('Arial', 10, 'bold')).pack(anchor='w')
            for opp in analysis['human_analysis']['opportunities']:
                tk.Label(human_frame, text=f"• {opp}", font=('Arial', 9), 
                        wraplength=700, justify='left').pack(anchor='w', padx=20)
        
        if analysis['human_analysis']['risks']:
            tk.Label(human_frame, text="⚠️ Key Risks:", font=('Arial', 10, 'bold')).pack(anchor='w')
            for risk in analysis['human_analysis']['risks']:
                tk.Label(human_frame, text=f"• {risk}", font=('Arial', 9), 
                        wraplength=700, justify='left').pack(anchor='w', padx=20)
        
        # Weighting Information
        weight_frame = tk.Frame(scrollable_frame, relief='raised', bd=2)
        weight_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(weight_frame, text="⚖️ ADAPTIVE WEIGHTING", font=('Arial', 14, 'bold')).pack()
        
        weight_text = f"AI Weight: {analysis['weighting']['ai_weight']:.1%}\n"
        weight_text += f"Human Weight: {analysis['weighting']['human_weight']:.1%}\n"
        weight_text += f"Rationale: {analysis['weighting']['rationale'][:200]}..."
        
        tk.Label(weight_frame, text=weight_text, font=('Arial', 10), justify='left', 
                wraplength=700).pack(pady=5)
        
        # Performance Metrics
        perf_frame = tk.Frame(scrollable_frame, relief='raised', bd=2)
        perf_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(perf_frame, text="📈 PERFORMANCE METRICS", font=('Arial', 14, 'bold')).pack()
        
        perf_text = f"Processing Time: {analysis['performance']['processing_time']:.2f}s\n"
        perf_text += f"Data Completeness: {analysis['performance']['data_completeness']:.1%}\n"
        perf_text += f"Data Freshness: {analysis['performance']['data_freshness']}\n"
        perf_text += f"Risk-Adjusted Score: {analysis['performance']['risk_adjusted_score']:.3f}"
        
        tk.Label(perf_frame, text=perf_text, font=('Arial', 10), justify='left').pack(pady=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def show_portfolio_recommendations(self):
        """Show portfolio recommendations from hybrid AI"""
        if not HYBRID_AI_AVAILABLE or not self.hybrid_ai:
            messagebox.showwarning("Hybrid AI", "Hybrid AI system not available")
            return
        
        symbols = list(self.equities.keys())
        if not symbols:
            messagebox.showwarning("Portfolio", "No stocks in portfolio to analyze")
            return
        
        # Show loading
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Analyzing Portfolio...")
        loading_window.geometry("350x100")
        tk.Label(loading_window, text="🧠 Analyzing portfolio with Hybrid AI...", 
                font=('Arial', 12)).pack(expand=True)
        loading_window.update()
        
        try:
            recommendations = self.hybrid_ai.get_portfolio_recommendations(symbols, top_n=5)
            loading_window.destroy()
            
            # Create recommendations window
            self.create_recommendations_window(recommendations)
            
        except Exception as e:
            loading_window.destroy()
            messagebox.showerror("Analysis Error", f"Error analyzing portfolio: {e}")
    
    def create_recommendations_window(self, recommendations):
        """Create window to display portfolio recommendations"""
        window = tk.Toplevel(self.root)
        window.title("📊 Hybrid AI Portfolio Recommendations")
        window.geometry("900x500")
        
        # Header
        header_frame = tk.Frame(window, bg='darkgreen')
        header_frame.pack(fill='x', pady=5)
        tk.Label(header_frame, text="📊 Portfolio Recommendations", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkgreen').pack(pady=10)
        
        # Create treeview for recommendations
        columns = ("Rank", "Symbol", "Recommendation", "Confidence", "Expected Alpha", "Risk Score", "AI Weight")
        tree = ttk.Treeview(window, columns=columns, show='headings', height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Populate recommendations
        for rec in recommendations:
            tree.insert("", "end", values=(
                rec['rank'],
                rec['symbol'],
                rec['recommendation'],
                f"{rec['confidence']:.2f}",
                f"{rec['expected_alpha']:.2%}",
                f"{rec['risk_score']:.2f}",
                f"{rec['ai_weight']:.1%}"
            ))
        
        tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
    
    def show_market_insights(self):
        """Show market insights from hybrid AI"""
        if not HYBRID_AI_AVAILABLE or not self.hybrid_ai:
            messagebox.showwarning("Hybrid AI", "Hybrid AI system not available")
            return
        
        symbols = list(self.equities.keys())
        if not symbols:
            messagebox.showwarning("Portfolio", "No stocks in portfolio to analyze")
            return
        
        try:
            insights = self.hybrid_ai.get_market_insights(symbols)
            
            # Create insights window
            window = tk.Toplevel(self.root)
            window.title("🌍 Hybrid AI Market Insights")
            window.geometry("600x400")
            
            # Header
            header_frame = tk.Frame(window, bg='purple')
            header_frame.pack(fill='x', pady=5)
            tk.Label(header_frame, text="🌍 Market Insights", 
                    font=('Arial', 16, 'bold'), fg='white', bg='purple').pack(pady=10)
            
            # Create scrollable text
            text_frame = tk.Frame(window)
            text_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap='word', font=('Arial', 11))
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            # Format insights
            insights_text = "🤖 AI SENTIMENT ANALYSIS\n"
            insights_text += f"Bullish Signals: {insights['ai_sentiment']['bullish_signals']}\n"
            insights_text += f"Bearish Signals: {insights['ai_sentiment']['bearish_signals']}\n"
            insights_text += f"Overall AI Bias: {insights['ai_sentiment']['overall_bias']}\n\n"
            
            insights_text += "🧠 HUMAN-LIKE SENTIMENT ANALYSIS\n"
            insights_text += f"Bullish Signals: {insights['human_sentiment']['bullish_signals']}\n"
            insights_text += f"Bearish Signals: {insights['human_sentiment']['bearish_signals']}\n"
            insights_text += f"Overall Human Bias: {insights['human_sentiment']['overall_bias']}\n\n"
            
            insights_text += "💡 KEY OPPORTUNITIES\n"
            for opp in insights['key_opportunities']:
                insights_text += f"• {opp}\n"
            
            insights_text += "\n⚠️ KEY RISKS\n"
            for risk in insights['key_risks']:
                insights_text += f"• {risk}\n"
            
            insights_text += f"\n📊 Analysis Coverage: {insights['analysis_coverage']} stocks"
            
            text_widget.insert('1.0', insights_text)
            text_widget.config(state='disabled')
            
            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            messagebox.showerror("Insights Error", f"Error getting market insights: {e}")
    
    def on_close(self):
        self.running = False
        self.save_equities()
        self.save_trade_history()
        self.root.destroy()
    
if __name__ == '__main__':
    root = tk.Tk()
    app = QuantSphereGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

            





