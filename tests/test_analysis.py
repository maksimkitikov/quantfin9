import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis import EventStudyAnalyzer


def test_estimate_capm_parameters():
    # create synthetic data
    dates = pd.date_range('2024-01-01', periods=100)
    market_returns = np.random.normal(0, 0.01, size=100)
    asset_returns = 0.1 + 1.2 * market_returns + np.random.normal(0, 0.01, size=100)

    asset_df = pd.DataFrame({'Returns': asset_returns}, index=dates)
    market_df = pd.DataFrame({'Returns': market_returns}, index=dates)

    analyzer = EventStudyAnalyzer()
    alpha, beta, diag = analyzer.estimate_capm_parameters(asset_df, market_df, dates[-1])

    assert alpha is not None
    assert beta is not None
    assert abs(alpha - 0.1) < 0.05
    assert abs(beta - 1.2) < 0.2
