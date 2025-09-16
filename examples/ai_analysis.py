"""
QuantSphere AI Trading Platform - AI Analysis Example
Author: Your Name
Created: 2024

Example script demonstrating OpenAI integration for portfolio analysis.
"""

import openai
from config import OPENAI_API_KEY, AI_MODEL, AI_SYSTEM_NAME
from alpaca_integration import api

def fetch_portfolio():
    """Fetch current portfolio positions."""
    try:
        positions = api.list_positions()
        portfolio = []
        for pos in positions:
            portfolio.append({
                'symbol': pos.symbol,
                'qty': pos.qty,
                'entry_price': pos.avg_entry_price,
                'current_price': pos.current_price,
                'unrealized_pl': pos.unrealized_pl,
                'side': 'buy'
            })
        return portfolio
    except Exception as e:
        print(f"Error fetching portfolio: {e}")
        return []

def fetch_open_orders():
    """Fetch current open orders."""
    try:
        orders = api.list_orders(status='open')
        open_orders = []
        for order in orders:
            open_orders.append({
                'symbol': order.symbol,
                'qty': order.qty,
                'limit_price': order.limit_price,
                'side': order.side
            })
        return open_orders
    except Exception as e:
        print(f"Error fetching open orders: {e}")
        return []

def analyze_portfolio(message):
    """
    Analyze portfolio using QuantSphere AI.
    
    Args:
        message (str): User question about the portfolio
    
    Returns:
        str: AI analysis response
    """
    portfolio_data = fetch_portfolio()
    open_orders = fetch_open_orders()

    pre_prompt = f"""
    You are {AI_SYSTEM_NAME}, an advanced portfolio manager responsible for analyzing trading portfolios.
    Your tasks are the following:
    1.) Evaluate risk exposures of my current holdings
    2.) Analyze my open limit orders and their potential impact
    3.) Provide insights into portfolio health, diversification, trade adjustments, etc.
    4.) Speculate on the market outlook based on current market conditions
    5.) Identify potential market risks and suggest risk management strategies

    Here is my portfolio: {portfolio_data}

    Here are my open orders: {open_orders}

    Overall, answer the following question with priority having that background: {message}
    """

    try:
        response = openai.ChatCompletion.create(
            model=AI_MODEL,
            messages=[{"role": "system", "content": pre_prompt}],
            api_key=OPENAI_API_KEY
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error getting AI analysis: {e}"

if __name__ == "__main__":
    # Example usage
    print("QuantSphere AI Analysis Test")
    print("=" * 40)
    
    # Test AI analysis
    analysis = analyze_portfolio("How is my portfolio performing today?")
    print("AI Analysis:")
    print(analysis)
