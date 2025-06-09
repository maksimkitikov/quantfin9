import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from news_reasoner import NewsReasoner


def test_analyze_news_sentiment():
    articles = [
        {'title': 'Market gains on optimism', 'description': 'growth and recovery'},
        {'title': 'Trade tensions cause decline', 'description': 'worry among investors'},
    ]
    reasoner = NewsReasoner(polygon_api_key=None, newsapi_key=None)
    sentiment = reasoner.analyze_news_sentiment(articles)
    assert sentiment['sentiment_label'] == 'neutral'
    assert 'trade_policy' in sentiment['key_themes']
    assert sentiment['article_count'] == 2
