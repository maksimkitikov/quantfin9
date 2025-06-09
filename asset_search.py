"""
Comprehensive Asset Search Module
Provides universal search functionality for stocks, bonds, crypto, futures, and all asset types
Uses Alpha Vantage API and yfinance for comprehensive coverage
"""
import requests
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Tuple
import time

class UniversalAssetSearch:
    """
    Universal asset search engine supporting all financial instruments:
    - Stocks (US, International)
    - Bonds (Government, Corporate)
    - Cryptocurrencies 
    - Futures (Commodities, Indices)
    - ETFs, REITs, Options
    - Forex pairs
    """
    
    def __init__(self, alpha_vantage_key: str = "81GVO787XLOHC287"):
        self.av_key = alpha_vantage_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Comprehensive asset type mappings
        self.asset_categories = {
            'stocks': ['Common Stock', 'Preferred Stock', 'ADR'],
            'etfs': ['ETF', 'Exchange Traded Fund'],
            'crypto': ['Digital Currency', 'Cryptocurrency', 'Bitcoin', 'Ethereum'],
            'bonds': ['Government Bond', 'Corporate Bond', 'Municipal Bond'],
            'futures': ['Commodity Future', 'Index Future', 'Currency Future'],
            'forex': ['Currency Pair', 'FX'],
            'reits': ['REIT', 'Real Estate Investment Trust'],
            'indices': ['Index', 'Market Index']
        }
        
    def search_assets(self, query: str, asset_type: str = "all", limit: int = 20) -> List[Dict]:
        """
        Universal asset search across all financial instruments
        
        Args:
            query: Search term (name, symbol, or keyword)
            asset_type: Filter by type ('stocks', 'crypto', 'bonds', 'futures', 'all')
            limit: Maximum number of results
            
        Returns:
            List of asset dictionaries with symbol, name, type, exchange
        """
        results = []
        
        # Search using Alpha Vantage API
        av_results = self._search_alpha_vantage(query, limit)
        results.extend(av_results)
        
        # Enhance with yfinance data for better coverage
        yf_results = self._search_yfinance(query, asset_type, limit)
        results.extend(yf_results)
        
        # Remove duplicates and filter by type
        filtered_results = self._filter_and_deduplicate(results, asset_type, limit)
        
        return filtered_results[:limit]
    
    def _search_alpha_vantage(self, query: str, limit: int) -> List[Dict]:
        """Search using Alpha Vantage Symbol Search API"""
        try:
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': query,
                'apikey': self.av_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            if 'bestMatches' in data:
                for match in data['bestMatches'][:limit]:
                    asset_info = {
                        'symbol': match.get('1. symbol', ''),
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'market_open': match.get('5. marketOpen', ''),
                        'market_close': match.get('6. marketClose', ''),
                        'timezone': match.get('7. timezone', ''),
                        'currency': match.get('8. currency', ''),
                        'match_score': match.get('9. matchScore', '0'),
                        'source': 'Alpha Vantage'
                    }
                    results.append(asset_info)
            
            return results
            
        except Exception as e:
            print(f"Alpha Vantage search error: {e}")
            return []
    
    def _search_yfinance(self, query: str, asset_type: str, limit: int) -> List[Dict]:
        """Enhanced search using yfinance with comprehensive symbol patterns"""
        results = []
        
        # Common symbol patterns for different asset types
        symbol_patterns = self._generate_symbol_patterns(query, asset_type)
        
        for symbol in symbol_patterns[:limit]:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info and 'symbol' in info:
                    asset_info = {
                        'symbol': info.get('symbol', symbol),
                        'name': info.get('longName', info.get('shortName', query)),
                        'type': self._determine_asset_type(info),
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A'),
                        'exchange': info.get('exchange', 'N/A'),
                        'currency': info.get('currency', 'USD'),
                        'market_cap': info.get('marketCap', 'N/A'),
                        'source': 'Yahoo Finance'
                    }
                    results.append(asset_info)
                    
                # Avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                continue
                
        return results
    
    def _generate_symbol_patterns(self, query: str, asset_type: str) -> List[str]:
        """Generate comprehensive symbol patterns for different asset types"""
        patterns = []
        base_query = query.upper().strip()
        
        if asset_type in ['all', 'stocks', 'etfs']:
            # Stock patterns
            patterns.extend([
                base_query,
                f"{base_query}.L",     # London
                f"{base_query}.TO",    # Toronto
                f"{base_query}.AX",    # Australia
                f"{base_query}.HK",    # Hong Kong
                f"{base_query}.T",     # Tokyo
                f"{base_query}.DE",    # Frankfurt
                f"{base_query}.PA",    # Paris
            ])
        
        if asset_type in ['all', 'crypto']:
            # Cryptocurrency patterns
            crypto_pairs = ['USD', 'USDT', 'BTC', 'ETH']
            for pair in crypto_pairs:
                patterns.extend([
                    f"{base_query}-{pair}",
                    f"{base_query}{pair}",
                    f"{base_query}-USD"
                ])
        
        if asset_type in ['all', 'futures']:
            # Futures patterns
            futures_months = ['H25', 'M25', 'U25', 'Z25']  # 2025 contracts
            for month in futures_months:
                patterns.append(f"{base_query}{month}")
        
        if asset_type in ['all', 'forex']:
            # Forex patterns
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
            for curr in major_currencies:
                if base_query != curr:
                    patterns.extend([
                        f"{base_query}{curr}=X",
                        f"{curr}{base_query}=X"
                    ])
        
        return list(set(patterns))  # Remove duplicates
    
    def _determine_asset_type(self, info: Dict) -> str:
        """Determine asset type from yfinance info"""
        quote_type = info.get('quoteType', '').lower()
        sector = info.get('sector', '').lower()
        
        if quote_type == 'equity':
            return 'Stock'
        elif quote_type == 'etf':
            return 'ETF'
        elif quote_type == 'cryptocurrency':
            return 'Cryptocurrency'
        elif quote_type == 'future':
            return 'Future'
        elif quote_type == 'currency':
            return 'Forex'
        elif 'reit' in sector:
            return 'REIT'
        else:
            return 'Unknown'
    
    def _filter_and_deduplicate(self, results: List[Dict], asset_type: str, limit: int) -> List[Dict]:
        """Filter by asset type and remove duplicates"""
        if asset_type == 'all':
            filtered = results
        else:
            type_keywords = self.asset_categories.get(asset_type, [])
            filtered = []
            
            for result in results:
                result_type = result.get('type', '').lower()
                if any(keyword.lower() in result_type for keyword in type_keywords):
                    filtered.append(result)
        
        # Remove duplicates by symbol
        seen_symbols = set()
        unique_results = []
        
        for result in filtered:
            symbol = result.get('symbol', '').upper()
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_results.append(result)
        
        return unique_results
    
    def get_asset_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information for a specific asset"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                return None
            
            # Get basic price data
            hist = ticker.history(period="5d")
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
            
            details = {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'type': self._determine_asset_type(info),
                'current_price': current_price,
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')[:200] + '...' if info.get('longBusinessSummary') else 'N/A',
                'website': info.get('website', 'N/A'),
                'country': info.get('country', 'N/A')
            }
            
            return details
            
        except Exception as e:
            print(f"Error getting asset details for {symbol}: {e}")
            return None

def search_all_assets(query: str, asset_type: str = "all", limit: int = 20) -> List[Dict]:
    """
    Main function to search for any financial asset
    
    Args:
        query: Search term (name, symbol, or keyword)
        asset_type: Filter by type ('stocks', 'crypto', 'bonds', 'futures', 'all')
        limit: Maximum number of results
        
    Returns:
        List of matching assets
    """
    searcher = UniversalAssetSearch()
    return searcher.search_assets(query, asset_type, limit)

def get_asset_info(symbol: str) -> Optional[Dict]:
    """
    Get detailed information for a specific asset symbol
    
    Args:
        symbol: Asset symbol
        
    Returns:
        Detailed asset information
    """
    searcher = UniversalAssetSearch()
    return searcher.get_asset_details(symbol)