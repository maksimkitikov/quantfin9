"""
Market Data Module
Handles market data collection and reaction detection
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Handles market data collection and reaction detection"""
    
    def __init__(self):
        self.default_symbols = ['^GSPC', 'FXI', 'SOXX', 'IYT']
        self.reaction_thresholds = {
            'return': 0.008,  # 0.8%
            'volume': 1.5     # 1.5x average
        }
    
    def load_market_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Load market data for given symbols and date range
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Calculate additional metrics
                    data['Returns'] = data['Close'].pct_change()
                    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
                    data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
                    
                    # Clean timezone info
                    if hasattr(data.index, 'tz') and data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    market_data[symbol] = data
                    logger.info(f"Loaded data for {symbol}: {len(data)} records")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                continue
        
        return market_data
    
    def detect_market_reaction(self, market_data: Dict[str, pd.DataFrame], event_date: str) -> Dict[str, Dict]:
        """
        Detect significant market reactions on event date
        
        Args:
            market_data: Dictionary of market data
            event_date: Event date in YYYY-MM-DD format
            
        Returns:
            Dictionary with reaction analysis for each symbol
        """
        reactions = {}
        event_date_ts = pd.Timestamp(event_date)
        
        for symbol, data in market_data.items():
            try:
                # Find closest trading day to event date
                available_dates = data.index
                if event_date_ts not in available_dates:
                    # Find nearest trading day
                    nearest_date = min(available_dates, key=lambda x: abs((x - event_date_ts).days))
                    if abs((nearest_date - event_date_ts).days) > 3:
                        logger.warning(f"No trading data near {event_date} for {symbol}")
                        continue
                    event_date_ts = nearest_date
                
                if event_date_ts not in data.index:
                    continue
                
                # Get event day data
                event_return = data.loc[event_date_ts, 'Returns']
                event_volume = data.loc[event_date_ts, 'Volume']
                
                # Calculate pre-event averages
                pre_event_data = data[data.index < event_date_ts].tail(10)
                if len(pre_event_data) < 5:
                    continue
                
                avg_volume = pre_event_data['Volume'].mean()
                avg_return = pre_event_data['Returns'].mean()
                vol_return = pre_event_data['Returns'].std()
                
                # Calculate metrics
                volume_spike = event_volume / avg_volume if avg_volume > 0 else 1
                return_zscore = (event_return - avg_return) / vol_return if vol_return > 0 else 0
                
                # Determine significance
                significant_return = abs(event_return) > self.reaction_thresholds['return']
                significant_volume = volume_spike > self.reaction_thresholds['volume']
                significant_overall = significant_return or significant_volume
                
                reactions[symbol] = {
                    'event_date': event_date_ts.strftime('%Y-%m-%d'),
                    'return': float(event_return),
                    'volume_spike': float(volume_spike),
                    'return_zscore': float(return_zscore),
                    'significant_return': significant_return,
                    'significant_volume': significant_volume,
                    'significant_overall': significant_overall,
                    'avg_volume_5d': float(avg_volume),
                    'event_volume': float(event_volume)
                }
                
                logger.info(f"{symbol}: {event_return:.3f} return, {volume_spike:.1f}x volume")
                
            except Exception as e:
                logger.error(f"Error analyzing reaction for {symbol}: {e}")
                continue
        
        return reactions
    
    def validate_trading_day(self, date: str) -> bool:
        """
        Check if given date is a trading day
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            True if trading day, False otherwise
        """
        try:
            # Use SPY as proxy for market open
            ticker = yf.Ticker('SPY')
            data = ticker.history(start=date, end=date)
            return not data.empty
        except Exception:
            return False
    
    def get_trading_days_around_event(self, event_date: str, days_before: int = 10, days_after: int = 5) -> Tuple[str, str]:
        """
        Get trading day range around event
        
        Args:
            event_date: Event date in YYYY-MM-DD format
            days_before: Days before event to include
            days_after: Days after event to include
            
        Returns:
            Tuple of (start_date, end_date)
        """
        event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        
        # Add buffer for weekends/holidays
        start_date = (event_dt - timedelta(days=days_before + 7)).strftime('%Y-%m-%d')
        end_date = (event_dt + timedelta(days=days_after + 7)).strftime('%Y-%m-%d')
        
        return start_date, end_date
    
    def calculate_market_summary(self, reactions: Dict[str, Dict]) -> Dict:
        """
        Calculate overall market reaction summary
        
        Args:
            reactions: Market reaction data
            
        Returns:
            Summary statistics
        """
        if not reactions:
            return {}
        
        returns = [r['return'] for r in reactions.values() if 'return' in r]
        volume_spikes = [r['volume_spike'] for r in reactions.values() if 'volume_spike' in r]
        significant_count = sum(1 for r in reactions.values() if r.get('significant_overall', False))
        
        summary = {
            'total_assets': len(reactions),
            'significant_reactions': significant_count,
            'significant_pct': significant_count / len(reactions) * 100 if reactions else 0,
            'avg_return': np.mean(returns) if returns else 0,
            'max_abs_return': max([abs(r) for r in returns]) if returns else 0,
            'avg_volume_spike': np.mean(volume_spikes) if volume_spikes else 1,
            'max_volume_spike': max(volume_spikes) if volume_spikes else 1,
            'market_reaction_detected': significant_count >= len(reactions) * 0.3  # 30% threshold
        }
        
        return summary

def main():
    """Test market analysis functionality"""
    analyzer = MarketAnalyzer()
    
    # Test with recent date
    event_date = '2025-06-02'
    start_date, end_date = analyzer.get_trading_days_around_event(event_date)
    
    print(f"Loading market data from {start_date} to {end_date}")
    market_data = analyzer.load_market_data(analyzer.default_symbols, start_date, end_date)
    
    if market_data:
        reactions = analyzer.detect_market_reaction(market_data, event_date)
        summary = analyzer.calculate_market_summary(reactions)
        
        print(f"\nMarket Reaction Analysis for {event_date}:")
        print(f"Significant reactions: {summary.get('significant_reactions', 0)}/{summary.get('total_assets', 0)}")
        print(f"Average return: {summary.get('avg_return', 0):.3f}")
        print(f"Max volume spike: {summary.get('max_volume_spike', 1):.1f}x")
        print(f"Market reaction detected: {summary.get('market_reaction_detected', False)}")
    else:
        print("No market data loaded")

if __name__ == "__main__":
    main()
