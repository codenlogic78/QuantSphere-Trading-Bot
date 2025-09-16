#!/usr/bin/env python3
"""
QuantSphere AI Trading Platform - Advanced Risk Management System
Comprehensive risk assessment and portfolio protection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    """Advanced risk management system with multiple risk models"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.risk_config = self.load_risk_config()
        self.position_limits = {
            'max_position_size': 0.20,  # 20% max per position
            'max_sector_exposure': 0.40,  # 40% max per sector
            'max_correlation_exposure': 0.60,  # 60% max in correlated assets
            'max_drawdown': 0.15,  # 15% max portfolio drawdown
            'max_leverage': 1.0,  # No leverage by default
            'min_liquidity_ratio': 0.10  # 10% cash minimum
        }
        
        # Risk metrics tracking
        self.risk_metrics = {
            'var_95': 0.0,
            'var_99': 0.0,
            'expected_shortfall': 0.0,
            'beta': 1.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'position_concentration': 0.25,
            'correlation_warning': 0.8,
            'volatility_spike': 2.0,
            'drawdown_warning': 0.10,
            'liquidity_warning': 0.05
        }
    
    def load_risk_config(self) -> Dict:
        """Load risk configuration from file"""
        try:
            with open('risk_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'max_position_pct': 20,
                'max_portfolio_risk': 15,
                'stop_loss_pct': 10,
                'take_profit_pct': 25
            }
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_price: float, risk_per_trade: float = 0.02) -> int:
        """Calculate optimal position size using Kelly Criterion and risk management"""
        
        # Risk per trade (default 2% of portfolio)
        risk_amount = self.initial_capital * risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # Basic position size
        position_size = int(risk_amount / risk_per_share)
        
        # Apply position limits
        max_position_value = self.initial_capital * self.position_limits['max_position_size']
        max_shares = int(max_position_value / entry_price)
        
        return min(position_size, max_shares)
    
    def calculate_portfolio_var(self, positions: Dict, confidence: float = 0.95) -> float:
        """Calculate Portfolio Value at Risk"""
        if not positions:
            return 0.0
        
        # Simplified VaR calculation using historical simulation
        portfolio_values = []
        total_value = sum(pos['position'] * pos.get('current_price', pos['entry_price']) 
                         for pos in positions.values() if pos['position'] > 0)
        
        if total_value == 0:
            return 0.0
        
        # Monte Carlo simulation for VaR
        np.random.seed(42)
        num_simulations = 1000
        
        for _ in range(num_simulations):
            portfolio_change = 0
            for symbol, pos in positions.items():
                if pos['position'] > 0:
                    # Simulate price change (normal distribution)
                    daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% volatility
                    position_value = pos['position'] * pos.get('current_price', pos['entry_price'])
                    position_change = position_value * daily_return
                    portfolio_change += position_change
            
            portfolio_values.append(portfolio_change)
        
        # Calculate VaR at specified confidence level
        var = np.percentile(portfolio_values, (1 - confidence) * 100)
        return abs(var)
    
    def calculate_expected_shortfall(self, positions: Dict, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.calculate_portfolio_var(positions, confidence)
        
        # Simplified ES calculation
        # In practice, this would use the tail distribution beyond VaR
        expected_shortfall = var * 1.3  # Approximation
        return expected_shortfall
    
    def calculate_portfolio_beta(self, positions: Dict) -> float:
        """Calculate portfolio beta (simplified)"""
        # In a real implementation, this would use market data
        # For now, using sector-based beta estimates
        sector_betas = {
            'AAPL': 1.2, 'MSFT': 1.1, 'GOOGL': 1.3, 'AMZN': 1.4, 'TSLA': 1.8,
            'NVDA': 1.6, 'META': 1.4, 'NFLX': 1.3, 'AMD': 1.7, 'CRM': 1.2
        }
        
        total_value = sum(pos['position'] * pos.get('current_price', pos['entry_price']) 
                         for pos in positions.values() if pos['position'] > 0)
        
        if total_value == 0:
            return 1.0
        
        weighted_beta = 0
        for symbol, pos in positions.items():
            if pos['position'] > 0:
                weight = (pos['position'] * pos.get('current_price', pos['entry_price'])) / total_value
                beta = sector_betas.get(symbol, 1.0)
                weighted_beta += weight * beta
        
        return weighted_beta
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        cumulative = np.array(portfolio_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def assess_portfolio_risk(self, positions: Dict, trade_history: List[Dict]) -> Dict:
        """Comprehensive portfolio risk assessment"""
        
        # Calculate basic metrics
        var_95 = self.calculate_portfolio_var(positions, 0.95)
        var_99 = self.calculate_portfolio_var(positions, 0.99)
        expected_shortfall = self.calculate_expected_shortfall(positions, 0.95)
        portfolio_beta = self.calculate_portfolio_beta(positions)
        
        # Calculate performance metrics from trade history
        if trade_history:
            returns = [trade.get('pl', 0) for trade in trade_history[-252:]]  # Last year
            portfolio_values = np.cumsum(returns)
            
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculate_sortino_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(portfolio_values)
            
            # Calmar ratio (annual return / max drawdown)
            annual_return = np.sum(returns) if returns else 0
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        else:
            sharpe_ratio = sortino_ratio = max_drawdown = calmar_ratio = volatility = 0
        
        # Update risk metrics
        self.risk_metrics.update({
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'beta': portfolio_beta,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        })
        
        return self.risk_metrics
    
    def check_position_limits(self, symbol: str, proposed_position: int, 
                            entry_price: float, current_positions: Dict) -> Tuple[bool, str]:
        """Check if proposed position violates risk limits"""
        
        # Calculate total portfolio value
        total_value = sum(pos['position'] * pos.get('current_price', pos['entry_price']) 
                         for pos in current_positions.values() if pos['position'] > 0)
        
        # Add proposed position value
        proposed_value = proposed_position * entry_price
        new_total_value = total_value + proposed_value
        
        # Check position size limit
        position_weight = proposed_value / new_total_value if new_total_value > 0 else 0
        if position_weight > self.position_limits['max_position_size']:
            return False, f"Position size {position_weight:.1%} exceeds limit {self.position_limits['max_position_size']:.1%}"
        
        # Check sector concentration (simplified - would need sector mapping)
        # For now, just check if we already have a large position in this symbol
        if symbol in current_positions and current_positions[symbol]['position'] > 0:
            current_value = current_positions[symbol]['position'] * current_positions[symbol].get('current_price', current_positions[symbol]['entry_price'])
            combined_value = current_value + proposed_value
            combined_weight = combined_value / new_total_value
            
            if combined_weight > self.position_limits['max_position_size']:
                return False, f"Combined position weight {combined_weight:.1%} exceeds limit"
        
        # Check leverage
        if new_total_value > self.initial_capital * self.position_limits['max_leverage']:
            return False, f"Position would exceed leverage limit"
        
        return True, "Position approved"
    
    def generate_risk_alerts(self, positions: Dict, trade_history: List[Dict]) -> List[str]:
        """Generate risk alerts based on current portfolio state"""
        alerts = []
        
        # Calculate current metrics
        risk_metrics = self.assess_portfolio_risk(positions, trade_history)
        
        # Check concentration risk
        total_value = sum(pos['position'] * pos.get('current_price', pos['entry_price']) 
                         for pos in positions.values() if pos['position'] > 0)
        
        for symbol, pos in positions.items():
            if pos['position'] > 0:
                position_value = pos['position'] * pos.get('current_price', pos['entry_price'])
                weight = position_value / total_value if total_value > 0 else 0
                
                if weight > self.alert_thresholds['position_concentration']:
                    alerts.append(f"🚨 HIGH CONCENTRATION: {symbol} represents {weight:.1%} of portfolio")
        
        # Check drawdown
        if risk_metrics['max_drawdown'] > self.alert_thresholds['drawdown_warning']:
            alerts.append(f"📉 DRAWDOWN WARNING: Current drawdown {risk_metrics['max_drawdown']:.1%}")
        
        # Check volatility
        if risk_metrics['volatility'] > self.alert_thresholds['volatility_spike'] * 0.20:  # 40% annualized
            alerts.append(f"📊 HIGH VOLATILITY: Portfolio volatility {risk_metrics['volatility']:.1%}")
        
        # Check VaR
        if risk_metrics['var_95'] > self.initial_capital * 0.05:  # 5% of capital
            alerts.append(f"⚠️ HIGH VAR: 95% VaR is ${risk_metrics['var_95']:,.2f}")
        
        return alerts
    
    def suggest_risk_adjustments(self, positions: Dict, trade_history: List[Dict]) -> List[str]:
        """Suggest portfolio adjustments to reduce risk"""
        suggestions = []
        
        risk_metrics = self.assess_portfolio_risk(positions, trade_history)
        
        # Suggest diversification if concentrated
        total_value = sum(pos['position'] * pos.get('current_price', pos['entry_price']) 
                         for pos in positions.values() if pos['position'] > 0)
        
        concentrated_positions = []
        for symbol, pos in positions.items():
            if pos['position'] > 0:
                position_value = pos['position'] * pos.get('current_price', pos['entry_price'])
                weight = position_value / total_value if total_value > 0 else 0
                
                if weight > 0.20:  # 20% threshold
                    concentrated_positions.append((symbol, weight))
        
        if concentrated_positions:
            suggestions.append("💡 Consider reducing concentration in: " + 
                             ", ".join([f"{sym} ({w:.1%})" for sym, w in concentrated_positions]))
        
        # Suggest hedging if high beta
        if risk_metrics['beta'] > 1.5:
            suggestions.append(f"🛡️ Consider hedging: Portfolio beta {risk_metrics['beta']:.2f} is high")
        
        # Suggest stop-losses if high drawdown
        if risk_metrics['max_drawdown'] > 0.10:
            suggestions.append("🔒 Consider tightening stop-losses to limit drawdown")
        
        # Suggest cash allocation if fully invested
        cash_weight = max(0, (self.initial_capital - total_value) / self.initial_capital)
        if cash_weight < 0.05:  # Less than 5% cash
            suggestions.append("💰 Consider maintaining higher cash allocation for opportunities")
        
        return suggestions
    
    def create_risk_report(self, positions: Dict, trade_history: List[Dict]) -> Dict:
        """Generate comprehensive risk report"""
        risk_metrics = self.assess_portfolio_risk(positions, trade_history)
        alerts = self.generate_risk_alerts(positions, trade_history)
        suggestions = self.suggest_risk_adjustments(positions, trade_history)
        
        # Portfolio composition
        total_value = sum(pos['position'] * pos.get('current_price', pos['entry_price']) 
                         for pos in positions.values() if pos['position'] > 0)
        
        composition = {}
        for symbol, pos in positions.items():
            if pos['position'] > 0:
                position_value = pos['position'] * pos.get('current_price', pos['entry_price'])
                weight = position_value / total_value if total_value > 0 else 0
                composition[symbol] = {
                    'weight': weight,
                    'value': position_value,
                    'shares': pos['position'],
                    'entry_price': pos['entry_price'],
                    'current_price': pos.get('current_price', pos['entry_price'])
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': risk_metrics,
            'portfolio_composition': composition,
            'total_portfolio_value': total_value,
            'cash_allocation': max(0, self.initial_capital - total_value),
            'alerts': alerts,
            'suggestions': suggestions,
            'risk_limits': self.position_limits,
            'alert_thresholds': self.alert_thresholds
        }
    
    def save_risk_report(self, positions: Dict, trade_history: List[Dict], filename: str = None):
        """Save risk report to file"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.create_risk_report(positions, trade_history)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename

# Test the risk manager
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(initial_capital=100000)
    
    # Mock portfolio data
    mock_positions = {
        'AAPL': {'position': 100, 'entry_price': 150.0, 'current_price': 155.0},
        'MSFT': {'position': 50, 'entry_price': 300.0, 'current_price': 295.0},
        'GOOGL': {'position': 30, 'entry_price': 2500.0, 'current_price': 2600.0}
    }
    
    mock_trades = [
        {'timestamp': '2024-01-01 10:00:00', 'pl': 500},
        {'timestamp': '2024-01-02 11:00:00', 'pl': -250},
        {'timestamp': '2024-01-03 14:00:00', 'pl': 750},
        {'timestamp': '2024-01-04 09:00:00', 'pl': -100},
        {'timestamp': '2024-01-05 15:00:00', 'pl': 300}
    ]
    
    # Generate risk report
    report = risk_manager.create_risk_report(mock_positions, mock_trades)
    
    print("=== QUANTSPHERE RISK MANAGEMENT REPORT ===")
    print(f"Portfolio Value: ${report['total_portfolio_value']:,.2f}")
    print(f"Cash Allocation: ${report['cash_allocation']:,.2f}")
    print(f"\nRisk Metrics:")
    for metric, value in report['risk_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print(f"\nAlerts ({len(report['alerts'])}):")
    for alert in report['alerts']:
        print(f"  {alert}")
    
    print(f"\nSuggestions ({len(report['suggestions'])}):")
    for suggestion in report['suggestions']:
        print(f"  {suggestion}")
    
    # Save report
    filename = risk_manager.save_risk_report(mock_positions, mock_trades)
    print(f"\nRisk report saved to: {filename}")
