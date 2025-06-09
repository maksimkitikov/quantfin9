"""
News Reasoning Module
Enhances AI market analysis with real-time news context and semantic matching
"""

import numpy as np
from typing import Dict, List, Optional
import requests
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class NewsReasoner:
    """
    News-enhanced reasoning for market event interpretation
    Uses Polygon.io and NewsAPI for contextual analysis
    """
    
    def __init__(self, polygon_api_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        self.polygon_api_key = polygon_api_key or os.getenv('POLYGON_API_KEY')
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        
        # Market-relevant keywords for different asset classes
        self.keyword_mappings = {
            'china_trade': ['china', 'trade war', 'tariff', 'export', 'import', 'xi jinping', 'biden'],
            'technology': ['ai', 'artificial intelligence', 'chip', 'semiconductor', 'nvidia', 'microsoft'],
            'financial': ['fed', 'interest rate', 'banking', 'credit', 'inflation', 'monetary policy'],
            'energy': ['oil', 'crude', 'opec', 'energy', 'gas', 'petroleum'],
            'crypto': ['bitcoin', 'cryptocurrency', 'digital currency', 'blockchain', 'regulation']
        }
    
    def get_relevant_news(self, asset_name: str, event_date: str, lookback_days: int = 3) -> List[Dict]:
        """
        Fetch relevant news for the asset and timeframe
        
        Args:
            asset_name: Name of the asset being analyzed
            event_date: Event date string (YYYY-MM-DD)
            lookback_days: Days before event to search for news
            
        Returns:
            List of relevant news articles
        """
        
        try:
            # Determine relevant keywords based on asset
            keywords = self._get_asset_keywords(asset_name)
            
            # Try Polygon.io first if available
            if self.polygon_api_key:
                news_articles = self._fetch_polygon_news(asset_name, event_date, lookback_days)
                if news_articles:
                    return self._filter_news_by_keywords(news_articles, keywords)
            
            # Fallback to NewsAPI if available
            if self.newsapi_key:
                news_articles = self._fetch_newsapi_news(keywords, event_date, lookback_days)
                if news_articles:
                    return news_articles
            
            # Return empty if no API keys available
            return []
            
        except Exception as e:
            print(f"News fetch error: {str(e)}")
            return []
    
    def _get_asset_keywords(self, asset_name: str) -> List[str]:
        """Determine relevant keywords based on asset name"""
        
        asset_lower = asset_name.lower()
        keywords = []
        
        if any(term in asset_lower for term in ['china', 'fxi', 'ashr']):
            keywords.extend(self.keyword_mappings['china_trade'])
        
        if any(term in asset_lower for term in ['tech', 'qqq', 'xlk', 'nvidia', 'apple']):
            keywords.extend(self.keyword_mappings['technology'])
        
        if any(term in asset_lower for term in ['financial', 'xlf', 'bank']):
            keywords.extend(self.keyword_mappings['financial'])
        
        if any(term in asset_lower for term in ['energy', 'xle', 'oil']):
            keywords.extend(self.keyword_mappings['energy'])
        
        if any(term in asset_lower for term in ['crypto', 'bitcoin', 'btc']):
            keywords.extend(self.keyword_mappings['crypto'])
        
        if 's&p' in asset_lower or 'spy' in asset_lower:
            # For broad market, include all major themes
            keywords.extend(self.keyword_mappings['china_trade'])
            keywords.extend(self.keyword_mappings['technology'])
            keywords.extend(self.keyword_mappings['financial'])
        
        return list(set(keywords))  # Remove duplicates
    
    def _fetch_polygon_news(self, asset_name: str, event_date: str, lookback_days: int) -> List[Dict]:
        """Fetch news from Polygon.io"""
        
        if not self.polygon_api_key:
            return []
        
        try:
            # Convert date format
            event_dt = datetime.strptime(event_date, '%Y-%m-%d')
            start_date = event_dt - timedelta(days=lookback_days)
            
            # Polygon.io news endpoint
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                'apikey': self.polygon_api_key,
                'published_utc.gte': start_date.strftime('%Y-%m-%d'),
                'published_utc.lt': event_dt.strftime('%Y-%m-%d'),
                'limit': 50,
                'order': 'desc'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                print(f"Polygon API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Polygon news fetch error: {str(e)}")
            return []
    
    def _fetch_newsapi_news(self, keywords: List[str], event_date: str, lookback_days: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        
        if not self.newsapi_key:
            return []
        
        try:
            # Convert date format
            event_dt = datetime.strptime(event_date, '%Y-%m-%d')
            start_date = event_dt - timedelta(days=lookback_days)
            
            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            
            # Create search query from keywords
            query = ' OR '.join(keywords[:5])  # Limit to top 5 keywords
            
            params = {
                'apiKey': self.newsapi_key,
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': event_dt.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                print(f"NewsAPI error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"NewsAPI fetch error: {str(e)}")
            return []
    
    def _filter_news_by_keywords(self, articles: List[Dict], keywords: List[str]) -> List[Dict]:
        """Filter articles by keyword relevance"""
        
        relevant_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"
            
            # Check if any keywords match
            relevance_score = 0
            for keyword in keywords:
                if keyword.lower() in content:
                    relevance_score += 1
            
            if relevance_score > 0:
                article['relevance_score'] = relevance_score
                relevant_articles.append(article)
        
        # Sort by relevance score
        relevant_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_articles[:10]  # Return top 10 most relevant
    
    def analyze_news_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze overall news sentiment"""
        
        if not articles:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'key_themes': [],
                'impact_assessment': 'minimal'
            }
        
        # Simple sentiment analysis based on keywords
        negative_words = ['decline', 'fall', 'drop', 'crisis', 'concern', 'worry', 'tension', 'conflict']
        positive_words = ['rise', 'growth', 'gain', 'optimism', 'recovery', 'boost', 'agreement']
        
        sentiment_scores = []
        themes = []
        
        for article in articles:
            content = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            negative_count = sum(1 for word in negative_words if word in content)
            positive_count = sum(1 for word in positive_words if word in content)
            
            # Calculate article sentiment (-1 to 1)
            if negative_count + positive_count > 0:
                article_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                article_sentiment = 0
            
            sentiment_scores.append(article_sentiment)
            
            # Extract key themes
            if 'trade' in content or 'china' in content:
                themes.append('trade_policy')
            if 'tech' in content or 'ai' in content:
                themes.append('technology')
            if 'fed' in content or 'rate' in content:
                themes.append('monetary_policy')
        
        # Calculate overall sentiment
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Determine sentiment label
        if overall_sentiment > 0.2:
            sentiment_label = 'positive'
        elif overall_sentiment < -0.2:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Assess impact
        if abs(overall_sentiment) > 0.4:
            impact = 'high'
        elif abs(overall_sentiment) > 0.2:
            impact = 'moderate'
        else:
            impact = 'minimal'
        
        return {
            'sentiment_score': overall_sentiment,
            'sentiment_label': sentiment_label,
            'key_themes': list(set(themes)),
            'impact_assessment': impact,
            'article_count': len(articles)
        }
    
    def enhance_interpretation(self, base_interpretation: str, asset_name: str, event_date: str) -> str:
        """
        Enhance base AI interpretation with news context
        
        Args:
            base_interpretation: Original AI interpretation
            asset_name: Asset being analyzed
            event_date: Event date
            
        Returns:
            Enhanced interpretation with news context
        """
        
        try:
            # Fetch relevant news
            news_articles = self.get_relevant_news(asset_name, event_date)
            
            if not news_articles:
                # Add note about no API keys if neither are available
                if not self.polygon_api_key and not self.newsapi_key:
                    enhanced = base_interpretation + "\n\n📰 **News Context:** *Real-time news analysis requires API keys for Polygon.io or NewsAPI.*"
                else:
                    enhanced = base_interpretation + "\n\n📰 **News Context:** No relevant news articles found for the specified timeframe."
                return enhanced
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_news_sentiment(news_articles)
            
            # Create news context section
            news_context = f"""
📰 **News Context Analysis:**
- **Sentiment:** {sentiment_analysis['sentiment_label'].title()} ({sentiment_analysis['sentiment_score']:+.2f})
- **Impact Assessment:** {sentiment_analysis['impact_assessment'].title()}
- **Key Themes:** {', '.join(sentiment_analysis['key_themes']) if sentiment_analysis['key_themes'] else 'Mixed topics'}
- **Articles Analyzed:** {sentiment_analysis['article_count']}

**Top Headlines:**"""
            
            # Add top 3 headlines
            for i, article in enumerate(news_articles[:3], 1):
                title = article.get('title', 'No title')[:80] + ('...' if len(article.get('title', '')) > 80 else '')
                news_context += f"\n{i}. {title}"
            
            # Combine with base interpretation
            enhanced = base_interpretation + "\n\n" + news_context
            
            return enhanced
            
        except Exception as e:
            return base_interpretation + f"\n\n📰 **News Context:** Error fetching news data - {str(e)}"


def main():
    """Test the news reasoner"""
    reasoner = NewsReasoner()
    
    # Test news fetching
    news = reasoner.get_relevant_news("S&P 500", "2025-06-02")
    print(f"Found {len(news)} relevant articles")
    
    if news:
        sentiment = reasoner.analyze_news_sentiment(news)
        print(f"Sentiment: {sentiment}")


if __name__ == "__main__":
    main()
