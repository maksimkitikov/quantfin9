import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from market import MarketAnalyzer


def test_calculate_market_summary():
    analyzer = MarketAnalyzer()
    reactions = {
        'AAPL': {'return': 0.02, 'volume_spike': 1.5, 'significant_overall': True},
        'MSFT': {'return': -0.01, 'volume_spike': 1.2, 'significant_overall': False},
    }
    summary = analyzer.calculate_market_summary(reactions)
    assert summary['total_assets'] == 2
    assert summary['significant_reactions'] == 1
    assert abs(summary['avg_return'] - 0.005) < 1e-6
