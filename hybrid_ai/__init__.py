"""
Hybrid AI Trading System Package
Based on Cao et al. (2024) - "Man + Machine" approach

This package implements a complete hybrid AI trading system that combines:
- AI/Machine learning analysis (quantitative, technical)
- Human-like AI reasoning (qualitative, contextual)
- Adaptive weighting based on stock characteristics
- Performance tracking and continuous learning

Main Components:
- data_collector: Multi-source data collection and preprocessing
- ai_analyst: AI/ML-based quantitative analysis
- human_ai_analyst: GPT-4 powered human-like reasoning
- adaptive_weighting: Dynamic weighting system
- hybrid_system: Main orchestrator

Usage:
    from hybrid_ai import HybridAITradingSystem
    
    system = HybridAITradingSystem(config)
    report = system.analyze_stock('AAPL')
"""

from .data_collector import MultiSourceDataCollector, StockCharacteristics
from .ai_analyst import AIAnalyst, AIAnalysisResult
from .human_ai_analyst import HumanAIAnalyst, HumanAIAnalysisResult
from .adaptive_weighting import AdaptiveWeightingSystem, HybridRecommendation, WeightingFactors
from .hybrid_system import HybridAITradingSystem, HybridAnalysisReport

__version__ = "1.0.0"
__author__ = "QuantSphere AI Trading Team"
__description__ = "Hybrid AI Trading System implementing Cao et al. (2024) research"

# Main exports
__all__ = [
    'HybridAITradingSystem',
    'HybridAnalysisReport',
    'MultiSourceDataCollector',
    'StockCharacteristics',
    'AIAnalyst',
    'AIAnalysisResult',
    'HumanAIAnalyst',
    'HumanAIAnalysisResult',
    'AdaptiveWeightingSystem',
    'HybridRecommendation',
    'WeightingFactors'
]
