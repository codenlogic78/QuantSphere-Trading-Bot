"""
QuantSphere AI Trading Platform Configuration
Author: Your Name
Created: 2024

Configuration settings for the QuantSphere trading platform.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/"  # Use paper trading by default

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# Trading Configuration
DEFAULT_PORTFOLIO_FILE = "quantsphere_portfolio.json"
DEFAULT_DRAWDOWN_PERCENT = 5.0
DEFAULT_LEVELS = 3
MAX_POSITION_SIZE = 1000  # Maximum shares per position

# Risk Management
MAX_PORTFOLIO_DRAWDOWN = 15.0  # Maximum portfolio drawdown percentage
STOP_LOSS_PERCENT = 10.0  # Stop loss percentage

# UI Configuration
WINDOW_TITLE = "QuantSphere AI Trading Platform"
REFRESH_INTERVAL = 5  # seconds

# AI Configuration
AI_MODEL = "gpt-4"
AI_SYSTEM_NAME = "QuantSphere AI"
