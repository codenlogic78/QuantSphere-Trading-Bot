"""
QuantSphere AI Trading Platform - Alpaca Integration Example
Author: Your Name
Created: 2024

Example script demonstrating Alpaca API integration for the QuantSphere platform.
"""

import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")

def get_stock_price(symbol):
    """
    Fetch the latest stock price for a given symbol.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
    
    Returns:
        dict: Dictionary containing the current price
    """
    try:
        barset = api.get_latest_trade(symbol)
        return {"price": barset.price}
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return {"price": -1}

def get_max_entry_price(symbol):
    """
    Get the maximum entry price from filled orders for a symbol.
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        float: Maximum filled price or -1 if no orders found
    """
    try:
        orders = api.list_orders(status="filled", limit=50)
        prices = [float(order.filled_avg_price) for order in orders 
                 if order.filled_avg_price and order.symbol == symbol]
        return max(prices) if prices else -1
    except Exception as e:
        print(f"Error fetching max entry price for {symbol}: {e}")
        return 0

def get_account_info():
    """
    Get account information and buying power.
    
    Returns:
        dict: Account information
    """
    try:
        account = api.get_account()
        return {
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "cash": float(account.cash)
        }
    except Exception as e:
        print(f"Error fetching account info: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    print("QuantSphere Alpaca Integration Test")
    print("=" * 40)
    
    # Test stock price fetch
    price_data = get_stock_price("AAPL")
    print(f"AAPL Current Price: ${price_data['price']}")
    
    # Test account info
    account_info = get_account_info()
    if account_info:
        print(f"Account Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        print(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
