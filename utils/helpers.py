"""
QuantSphere AI Trading Platform - Utility Functions
Author: Your Name
Created: 2024

Helper functions for the QuantSphere trading platform.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for the QuantSphere platform.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('quantsphere.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('QuantSphere')

def validate_symbol(symbol: str) -> bool:
    """
    Validate a stock symbol format.
    
    Args:
        symbol (str): Stock symbol to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.upper().strip()
    return len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalpha()

def calculate_position_size(account_value: float, risk_percent: float, 
                          entry_price: float, stop_loss_price: float) -> int:
    """
    Calculate position size based on risk management rules.
    
    Args:
        account_value (float): Total account value
        risk_percent (float): Risk percentage (e.g., 0.02 for 2%)
        entry_price (float): Entry price per share
        stop_loss_price (float): Stop loss price per share
    
    Returns:
        int: Number of shares to buy
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    
    risk_amount = account_value * risk_percent
    price_difference = abs(entry_price - stop_loss_price)
    
    if price_difference == 0:
        return 0
    
    position_size = int(risk_amount / price_difference)
    return max(0, position_size)

def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount (float): Amount to format
    
    Returns:
        str: Formatted currency string
    """
    return f"${amount:,.2f}"

def save_json_data(data: Dict[str, Any], filename: str) -> bool:
    """
    Save data to a JSON file safely.
    
    Args:
        data (Dict): Data to save
        filename (str): Filename to save to
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON data to {filename}: {e}")
        return False

def load_json_data(filename: str) -> Dict[str, Any]:
    """
    Load data from a JSON file safely.
    
    Args:
        filename (str): Filename to load from
    
    Returns:
        Dict: Loaded data or empty dict if error
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Error loading JSON data from {filename}: {e}")
        return {}

def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        str: Current timestamp
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
