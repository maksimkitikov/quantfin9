"""
Alpha Vantage Data Module
Fetches authentic market data using Alpha Vantage API
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import time


class AlphaVantageClient:
    """
    Alpha Vantage API client for fetching market data
    """
    
    def __init__(self, api_key: str = "81GVO787XLOHC287"):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def fetch_daily_data(self, symbol: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
        """
        Fetch daily time series data from Alpha Vantage
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data and returns
        """
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                if 'Note' in data:
                    print(f"Alpha Vantage rate limit: {data['Note']}")
                    time.sleep(12)  # Wait 12 seconds for rate limit
                    return self.fetch_daily_data(symbol, outputsize)
                elif 'Error Message' in data:
                    print(f"Alpha Vantage error: {data['Error Message']}")
                    return None
                else:
                    print(f"Unexpected Alpha Vantage response: {data}")
                    return None
            
            # Parse time series data
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adj_Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.set_index('Date').sort_index()
            
            # Calculate returns using adjusted close
            df['Returns'] = df['Adj_Close'].pct_change()
            
            # Add validation prints
            print(f"=== ALPHA VANTAGE DATA VALIDATION ===")
            print(f"Symbol: {symbol}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Data points: {len(df)}")
            print(f"Latest close: {df['Adj_Close'].iloc[-1]:.2f}")
            print(f"Latest return: {df['Returns'].iloc[-1]:.6f} ({df['Returns'].iloc[-1]*100:.4f}%)")
            print(f"========================================")
            
            return df
            
        except Exception as e:
            print(f"Alpha Vantage fetch error for {symbol}: {str(e)}")
            return None
    
    def fetch_market_data(self, symbols: list, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        
        results = {}
        
        for symbol in symbols:
            print(f"Fetching {symbol} from Alpha Vantage...")
            data = self.fetch_daily_data(symbol)
            
            if data is not None and not data.empty:
                # Filter by date range if specified
                if start_date and end_date:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                
                results[symbol] = data
                
                # Rate limiting - Alpha Vantage allows 5 calls per minute for free tier
                time.sleep(12)  # Wait 12 seconds between calls
            else:
                print(f"Failed to fetch data for {symbol}")
        
        return results


def enhance_event_study_with_alpha_vantage(asset_symbol: str, market_symbol: str = "SPY", 
                                         event_date: str = "2025-06-02") -> Dict:
    """
    Perform event study using Alpha Vantage data
    
    Args:
        asset_symbol: Symbol to analyze
        market_symbol: Market benchmark symbol
        event_date: Event date
        
    Returns:
        Event study results with authentic data
    """
    
    client = AlphaVantageClient()
    
    # Fetch data for both asset and market
    symbols = [asset_symbol, market_symbol]
    market_data = client.fetch_market_data(symbols)
    
    if asset_symbol not in market_data or market_symbol not in market_data:
        return {"error": "Failed to fetch required data from Alpha Vantage"}
    
    asset_data = market_data[asset_symbol]
    benchmark_data = market_data[market_symbol]
    
    # Calculate event study metrics
    event_dt = pd.to_datetime(event_date)
    
    # Get event window data
    window_start = event_dt - pd.Timedelta(days=5)
    window_end = event_dt + pd.Timedelta(days=5)
    
    asset_window = asset_data[(asset_data.index >= window_start) & (asset_data.index <= window_end)]
    market_window = benchmark_data[(benchmark_data.index >= window_start) & (benchmark_data.index <= window_end)]
    
    # Align data
    aligned_data = pd.concat([asset_window['Returns'], market_window['Returns']], 
                            axis=1, join='inner')
    aligned_data.columns = ['Asset_Returns', 'Market_Returns']
    
    # Simple market model for abnormal returns
    # AR = Asset Return - Market Return (simplified single-factor model)
    aligned_data['Abnormal_Returns'] = aligned_data['Asset_Returns'] - aligned_data['Market_Returns']
    aligned_data['Cumulative_AR'] = aligned_data['Abnormal_Returns'].cumsum()
    
    # Calculate statistics
    event_day_ar = 0
    event_day_actual = 0
    if event_dt in aligned_data.index:
        event_day_ar = aligned_data.loc[event_dt, 'Abnormal_Returns']
        event_day_actual = aligned_data.loc[event_dt, 'Asset_Returns']
    
    results = {
        'abnormal_data': aligned_data,
        'ar_statistics': {
            'event_day_ar': event_day_ar,
            'event_day_actual': event_day_actual,
            'mean_ar': aligned_data['Abnormal_Returns'].mean(),
            'std_ar': aligned_data['Abnormal_Returns'].std(),
            'car_total': aligned_data['Cumulative_AR'].iloc[-1] if not aligned_data.empty else 0,
            'observations': len(aligned_data)
        },
        'data_source': 'Alpha Vantage',
        'asset_symbol': asset_symbol,
        'market_symbol': market_symbol
    }
    
    print(f"=== ALPHA VANTAGE EVENT STUDY ===")
    print(f"Asset: {asset_symbol}, Market: {market_symbol}")
    print(f"Event Date: {event_date}")
    print(f"Event Day AR: {event_day_ar:.6f} ({event_day_ar*100:.4f}%)")
    print(f"Event Day Actual: {event_day_actual:.6f} ({event_day_actual*100:.4f}%)")
    print(f"CAR Total: {results['ar_statistics']['car_total']:.6f}")
    print(f"==================================")
    
    return results


def main():
    """Test Alpha Vantage integration"""
    
    # Test single symbol fetch
    client = AlphaVantageClient()
    spy_data = client.fetch_daily_data("SPY")
    
    if spy_data is not None:
        print(f"Successfully fetched {len(spy_data)} days of SPY data")
        print(spy_data.tail())
    
    # Test event study
    results = enhance_event_study_with_alpha_vantage("QQQ", "SPY", "2025-06-02")
    print(f"Event study results: {results}")


if __name__ == "__main__":
    main()