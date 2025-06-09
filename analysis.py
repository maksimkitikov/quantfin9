"""
Event Study Analysis Module
CAPM-based abnormal returns and statistical testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventStudyAnalyzer:
    """Professional event study analysis using CAMP model"""
    
    def __init__(self, market_symbol: str = '^GSPC'):
        self.market_symbol = market_symbol
        self.estimation_window = 60
        self.event_window = 11  # [-5, +5] days
    
    def estimate_capm_parameters(self, asset_data: pd.DataFrame, market_data: pd.DataFrame, 
                                estimation_end: pd.Timestamp) -> Tuple[float, float, Dict]:
        """
        Estimate CAPM parameters (alpha, beta) using rolling estimation window
        
        Args:
            asset_data: Asset price data with Returns column
            market_data: Market data with Returns column
            estimation_end: End date for estimation window
            
        Returns:
            Tuple of (alpha, beta, diagnostics)
        """
        try:
            # Define estimation window
            estimation_start = estimation_end - pd.Timedelta(days=self.estimation_window + 30)
            
            # Extract estimation data
            asset_est = asset_data.loc[estimation_start:estimation_end, 'Returns'].dropna()
            market_est = market_data.loc[estimation_start:estimation_end, 'Returns'].dropna()
            
            # Align data
            combined = pd.concat([asset_est, market_est], axis=1, join='inner')
            combined.columns = ['Asset', 'Market']
            combined = combined.dropna()
            
            if len(combined) < 30:
                logger.warning("Insufficient data for CAPM estimation")
                return None, None, {}
            
            # Remove extreme outliers (3-sigma rule)
            z_scores = np.abs(stats.zscore(combined))
            combined = combined[(z_scores < 3).all(axis=1)]
            
            if len(combined) < 20:
                return None, None, {}
            
            # Linear regression: R_asset = alpha + beta * R_market + epsilon
            x = combined['Market'].values
            y = combined['Asset'].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Additional diagnostics
            residuals = y - (intercept + slope * x)
            mse = np.mean(residuals ** 2)
            
            diagnostics = {
                'alpha': intercept,
                'beta': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'mse': mse,
                'observations': len(combined),
                'durbin_watson': self._durbin_watson_test(residuals)
            }
            
            return intercept, slope, diagnostics
            
        except Exception as e:
            logger.error(f"CAPM estimation failed: {e}")
            return None, None, {}
    
    def compute_abnormal_returns(self, asset_data: pd.DataFrame, market_data: pd.DataFrame,
                                alpha: float, beta: float, event_date: pd.Timestamp) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute abnormal returns and cumulative abnormal returns
        
        Args:
            asset_data: Asset price data
            market_data: Market data
            alpha: CAPM alpha parameter
            beta: CAPM beta parameter
            event_date: Event date
            
        Returns:
            Tuple of (event_data, statistics)
        """
        try:
            # Define event window [-5, +5]
            event_start = event_date - pd.Timedelta(days=5)
            event_end = event_date + pd.Timedelta(days=5)
            
            # Extract event window data
            asset_event = asset_data.loc[event_start:event_end, 'Returns'].dropna()
            market_event = market_data.loc[event_start:event_end, 'Returns'].dropna()
            
            # Align data
            event_data = pd.concat([asset_event, market_event], axis=1, join='inner')
            event_data.columns = ['Asset_Return', 'Market_Return']
            
            if event_data.empty:
                return None, {}
            
            # CAPM expected returns: E(R) = alpha + beta * R_market
            event_data['Expected_Return'] = alpha + beta * event_data['Market_Return']
            
            # Abnormal returns: AR = R_actual - R_expected
            event_data['Abnormal_Return'] = event_data['Asset_Return'] - event_data['Expected_Return']
            
            # Cumulative abnormal returns
            event_data['CAR'] = event_data['Abnormal_Return'].cumsum()
            
            # Calculate days relative to event
            event_data['Days_to_Event'] = (event_data.index - event_date).days
            
            # Statistical tests
            ar_values = event_data['Abnormal_Return'].values
            ar_mean = np.mean(ar_values)
            ar_std = np.std(ar_values, ddof=1)
            car_total = event_data['CAR'].iloc[-1] if not event_data['CAR'].empty else 0
            
            # T-test for abnormal returns
            if ar_std > 0 and len(ar_values) > 1:
                t_stat = ar_mean / (ar_std / np.sqrt(len(ar_values)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ar_values) - 1))
            else:
                t_stat = 0
                p_value = 1
            
            # Sign test
            positive_days = np.sum(ar_values > 0)
            negative_days = np.sum(ar_values < 0)
            
            # CAR significance test
            car_variance = ar_std ** 2 * len(ar_values)  # Simplified variance
            car_t_stat = car_total / np.sqrt(car_variance) if car_variance > 0 else 0
            car_p_value = 2 * (1 - stats.t.cdf(abs(car_t_stat), len(ar_values) - 1)) if car_variance > 0 else 1
            
            statistics = {
                'mean_ar': ar_mean,
                'std_ar': ar_std,
                'car_total': car_total,
                't_statistic': t_stat,
                'p_value': p_value,
                'car_t_statistic': car_t_stat,
                'car_p_value': car_p_value,
                'observations': len(ar_values),
                'positive_days': positive_days,
                'negative_days': negative_days,
                'max_ar': np.max(ar_values),
                'min_ar': np.min(ar_values),
                'significant_5pct': p_value < 0.05,
                'significant_1pct': p_value < 0.01
            }
            
            return event_data, statistics
            
        except Exception as e:
            logger.error(f"Abnormal returns calculation failed: {e}")
            return None, {}
    
    def analyze_volatility_clustering(self, asset_data: pd.DataFrame, event_date: pd.Timestamp) -> Dict:
        """
        Analyze volatility clustering around event (simplified GARCH)
        
        Args:
            asset_data: Asset data with Returns
            event_date: Event date
            
        Returns:
            Volatility analysis results
        """
        try:
            returns = asset_data['Returns'].dropna()
            
            # Pre and post event periods
            pre_event = returns[returns.index < event_date]
            post_event = returns[returns.index >= event_date]
            
            if len(pre_event) < 20 or len(post_event) < 5:
                return {}
            
            # Calculate rolling volatility
            window = min(20, len(pre_event) // 2)
            pre_vol = pre_event.rolling(window=window).std().mean()
            post_vol = post_event.rolling(window=min(window, len(post_event))).std().mean()
            
            vol_change = (post_vol - pre_vol) / pre_vol if pre_vol > 0 else 0
            
            # ARCH test (simplified)
            squared_returns = returns ** 2
            arch_lags = min(5, len(returns) // 10)
            
            volatility_analysis = {
                'pre_event_volatility': pre_vol,
                'post_event_volatility': post_vol,
                'volatility_change_pct': vol_change * 100,
                'volatility_spike': vol_change > 0.2,
                'volatility_clustering': abs(vol_change) > 0.15
            }
            
            return volatility_analysis
            
        except Exception as e:
            logger.warning(f"Volatility analysis failed: {e}")
            return {}
    
    def run_complete_analysis(self, asset_data: pd.DataFrame, market_data: pd.DataFrame, 
                             event_date: pd.Timestamp, asset_name: str) -> Dict:
        """
        Run complete event study analysis
        
        Args:
            asset_data: Asset price data
            market_data: Market data
            event_date: Event date
            asset_name: Asset name for reporting
            
        Returns:
            Complete analysis results
        """
        results = {
            'asset_name': asset_name,
            'event_date': event_date.strftime('%Y-%m-%d'),
            'success': False
        }
        
        try:
            # Step 1: Estimate CAPM parameters
            estimation_end = event_date - pd.Timedelta(days=1)
            alpha, beta, capm_diagnostics = self.estimate_camp_parameters(
                asset_data, market_data, estimation_end
            )
            
            if alpha is None:
                results['error'] = 'CAPM estimation failed'
                return results
            
            # Step 2: Compute abnormal returns
            event_data, ar_statistics = self.compute_abnormal_returns(
                asset_data, market_data, alpha, beta, event_date
            )
            
            if event_data is None:
                results['error'] = 'Abnormal returns calculation failed'
                return results
            
            # Step 3: Volatility analysis
            volatility_analysis = self.analyze_volatility_clustering(asset_data, event_date)
            
            # Compile results
            results.update({
                'success': True,
                'alpha': alpha,
                'beta': beta,
                'capm_diagnostics': camp_diagnostics,
                'event_data': event_data,
                'ar_statistics': ar_statistics,
                'volatility_analysis': volatility_analysis
            })
            
            logger.info(f"Analysis completed for {asset_name}: CAR={ar_statistics.get('car_total', 0):.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Complete analysis failed for {asset_name}: {e}")
            results['error'] = str(e)
            return results
    
    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        try:
            diff = np.diff(residuals)
            dw = np.sum(diff ** 2) / np.sum(residuals ** 2)
            return dw
        except:
            return 2.0  # No autocorrelation

def main():
    """Test event study analysis"""
    import yfinance as yf
    
    # Load test data
    symbol = 'AAPL'
    event_date = pd.Timestamp('2025-06-02')
    
    # Download data
    start_date = event_date - pd.Timedelta(days=100)
    end_date = event_date + pd.Timedelta(days=10)
    
    asset_ticker = yf.Ticker(symbol)
    market_ticker = yf.Ticker('^GSPC')
    
    asset_data = asset_ticker.history(start=start_date, end=end_date)
    market_data = market_ticker.history(start=start_date, end=end_date)
    
    asset_data['Returns'] = asset_data['Close'].pct_change()
    market_data['Returns'] = market_data['Close'].pct_change()
    
    # Run analysis
    analyzer = EventStudyAnalyzer()
    results = analyzer.run_complete_analysis(asset_data, market_data, event_date, symbol)
    
    if results['success']:
        print(f"Event Study Results for {symbol}:")
        print(f"Alpha: {results['alpha']:.6f}")
        print(f"Beta: {results['beta']:.4f}")
        print(f"CAR: {results['ar_statistics']['car_total']:.4f}")
        print(f"P-value: {results['ar_statistics']['p_value']:.4f}")
        print(f"Significant: {results['ar_statistics']['significant_5pct']}")
    else:
        print(f"Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()