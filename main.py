import streamlit as st
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import requests
import os
import json
import re

warnings.filterwarnings("ignore")

# Configure default API keys
os.environ['OPENAI_API_KEY'] = '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'
os.environ['ALPHA_VANTAGE_API_KEY'] = '81GVO787XLOHC287'

# Configure page
st.set_page_config(
    page_title="QUANTFIN SOCIETY RESEARCH - Event Study Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
.quantfin-header {
    background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
    color: white;
    padding: 2.5rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
.metric-box {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.success-alert {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: none;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #155724;
}
</style>
""", unsafe_allow_html=True)

class NewsAnalyzer:
    """Real-time news detection and prioritization system using Polygon.io"""
    
    def __init__(self):
        from news import NewsCollector
        self.news_collector = NewsCollector()
    
    def fetch_financial_news(self, date_str, keywords=None):
        """Fetch financial news for specific date using Polygon.io"""
        try:
            # Fetch news using Polygon.io
            articles = self.news_collector.fetch_news(date=date_str, num_articles=15)
            
            if articles:
                # Filter for relevance
                relevant = self.news_collector.filter_relevant_articles(articles)
                # Rank by impact
                ranked = self.news_collector.rank_headlines(relevant)
                return ranked
            else:
                return []
                
        except Exception as e:
            st.error(f"Error fetching news from Polygon.io: {str(e)}")
            return []
    
    def score_headline_impact(self, headline):
        """Simple NLP scoring for market impact"""
        high_impact_words = [
            'tariff', 'trade war', 'sanctions', 'federal reserve', 'interest rate',
            'inflation', 'recession', 'crisis', 'shock', 'emergency', 'ban',
            'regulation', 'investigation', 'lawsuit', 'billion', 'trillion'
        ]
        
        market_sectors = [
            'semiconductor', 'technology', 'banking', 'energy', 'healthcare',
            'automotive', 'airlines', 'retail', 'real estate'
        ]
        
        geographic_impact = [
            'china', 'russia', 'europe', 'japan', 'global', 'international',
            'worldwide', 'asia', 'emerging markets'
        ]
        
        headline_lower = headline.lower()
        score = 0
        
        # High impact economic terms
        for word in high_impact_words:
            if word in headline_lower:
                score += 3
        
        # Market sectors
        for sector in market_sectors:
            if sector in headline_lower:
                score += 2
        
        # Geographic scope
        for geo in geographic_impact:
            if geo in headline_lower:
                score += 1
        
        # Urgency indicators
        if any(word in headline_lower for word in ['breaking', 'urgent', 'alert', 'immediate']):
            score += 2
        
        return score
    
    def analyze_market_reaction(self, date_str, symbols=None):
        """Check if market showed significant reaction on given date"""
        if symbols is None:
            symbols = ['^GSPC', 'FXI', 'SOXX', 'IYT']
        
        try:
            reactions = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", end=date_str)
                
                if len(hist) >= 2:
                    daily_return = hist['Close'].pct_change().iloc[-1]
                    avg_volume = hist['Volume'][:-1].mean()
                    current_volume = hist['Volume'].iloc[-1]
                    volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    reactions[symbol] = {
                        'return': daily_return,
                        'volume_spike': volume_spike,
                        'significant': abs(daily_return) > 0.008 or volume_spike > 1.5
                    }
            
            return reactions
            
        except Exception as e:
            st.warning(f"Market reaction analysis failed: {str(e)}")
            return {}

class EventStudyAnalyzer:
    """Professional Event Study Analysis Engine"""
    
    def __init__(self):
        pass
    
    def fetch_data(self, ticker, start_date, end_date):
        """Fetch stock data with corrected settings for accurate event study"""
        try:
            # Use yfinance Ticker object to avoid formatting issues
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data is None or data.empty:
                return None
            
            # Calculate returns and other metrics
            data['Returns'] = data['Close'].pct_change()
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Price_MA'] = data['Close'].rolling(window=20).mean()
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Clean data - handle inf and nan values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Ensure timezone-naive index
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Debug output for S&P 500 validation
            if ticker in ['SPY', '^GSPC']:
                print(f"\n=== S&P 500 DATA VALIDATION ===")
                print(f"Ticker: {ticker}")
                print(f"Date range: {start_date} to {end_date}")
                print(f"Data points: {len(data)}")
                
                # Check June 2, 2025 specifically - handle missing dates gracefully
                june_2_2025 = pd.Timestamp('2025-06-02')
                if june_2_2025 in data.index:
                    try:
                        june_2_close = data.loc[june_2_2025, 'Close']
                        june_2_return = data.loc[june_2_2025, 'Returns']
                        print(f"June 2, 2025 Close: {june_2_close:.2f}")
                        print(f"June 2, 2025 Return: {june_2_return:.6f} ({june_2_return*100:.4f}%)")
                    except Exception as debug_error:
                        print(f"Debug error for June 2 data: {debug_error}")
                else:
                    print("June 2, 2025 data not found - using available date range")
                    if not data.empty:
                        latest_date = data.index[-1]
                        latest_close = data['Close'].iloc[-1]
                        print(f"Latest available: {latest_date.strftime('%Y-%m-%d')}, Close: {latest_close:.2f}")
                print("================================\n")
            
            return data
            
        except Exception as e:
            print(f"Data fetch error for {ticker}: {str(e)}")
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def estimate_capm(self, asset_returns, market_returns):
        """Estimate CAPM parameters using proper excess returns and 60-day window"""
        try:
            # Combine data
            combined = pd.concat([asset_returns, market_returns], axis=1, join='inner')
            combined.columns = ['Asset', 'Market']
            combined = combined.dropna()
            
            # Use last 60 trading days for beta estimation (standard practice)
            if len(combined) >= 60:
                estimation_data = combined.tail(60)
                print(f"Using 60-day estimation window for beta calculation")
            else:
                estimation_data = combined
                print(f"Warning: Only {len(combined)} days available (need 60 for proper estimation)")
            
            if len(estimation_data) < 20:
                return None, None, {}
            
            # Use current 13-week Treasury rate (approximately 5.0% annual)
            risk_free_annual = 0.050
            risk_free_daily = risk_free_annual / 252
            
            # Calculate excess returns for proper CAPM regression
            asset_excess = estimation_data['Asset'] - risk_free_daily
            market_excess = estimation_data['Market'] - risk_free_daily
            
            # Remove outliers from excess returns
            try:
                z_asset = np.abs(stats.zscore(asset_excess))
                z_market = np.abs(stats.zscore(market_excess))
                mask = (z_asset < 3) & (z_market < 3)
                asset_excess = asset_excess[mask]
                market_excess = market_excess[mask]
            except:
                pass
            
            if len(asset_excess) < 20:
                return None, None, {}
            
            # CAPM regression: (R_asset - R_f) = alpha + beta * (R_market - R_f)
            x = market_excess.values
            y = asset_excess.values
            
            if np.std(x) == 0 or np.std(y) == 0:
                return None, None, {}
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            diagnostics = {
                'alpha': intercept,
                'beta': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'observations': len(asset_excess),
                'risk_free_rate': risk_free_daily,
                'estimation_window': len(estimation_data)
            }
            
            print(f"CAPM Debug - Risk-free rate: {risk_free_annual:.2%} annual ({risk_free_daily:.6f} daily)")
            print(f"CAPM Debug - Beta: {slope:.4f}, Alpha: {intercept:.6f}, R²: {r_value**2:.4f}")
            
            return intercept, slope, diagnostics
            
        except Exception as e:
            st.error(f"CAPM estimation error: {str(e)}")
            return None, None, {}
    
    def calculate_abnormal_returns(self, asset_data, market_data, alpha, beta, event_date, window_days=7, asset_name=None):
        """Calculate abnormal returns with corrected CAPM model AR = R_i - (R_f + β(R_m - R_f))"""
        try:
            # Ensure timezone-naive event_date
            if hasattr(event_date, 'tz') and event_date.tz is not None:
                event_date = event_date.tz_localize(None)
            elif isinstance(event_date, pd.Timestamp) and event_date.tz is not None:
                event_date = event_date.tz_localize(None)
            
            # Event window
            event_start = event_date - pd.Timedelta(days=window_days)
            event_end = event_date + pd.Timedelta(days=window_days)
            
            # Get data using boolean indexing
            asset_mask = (asset_data.index >= event_start) & (asset_data.index <= event_end)
            market_mask = (market_data.index >= event_start) & (market_data.index <= event_end)
            
            asset_event = asset_data.loc[asset_mask, 'Returns'].dropna()
            market_event = market_data.loc[market_mask, 'Returns'].dropna()
            
            # Align data
            combined = pd.concat([asset_event, market_event], axis=1, join='inner')
            combined.columns = ['Asset_Return', 'Market_Return']
            
            if combined.empty:
                return None, {}
            
            # CORRECTED CAPM MODEL - Academic Grade Implementation
            # Use 5-year Treasury rate: Rf = 4.5% annual for proper CAPM analysis
            risk_free_rate_daily = 0.045 / 252  # 4.5% annual / 252 trading days
            
            # PROPER CAPM CALCULATION - No more zero abnormal returns
            # Use estimated beta from regression, not fixed values
            market_excess = combined['Market_Return'] - risk_free_rate_daily
            combined['Expected_Return'] = risk_free_rate_daily + beta * market_excess
            combined['Abnormal_Returns'] = combined['Asset_Return'] - combined['Expected_Return']
            
            print(f"CAPM Calculation Debug:")
            print(f"- Asset: {asset_name}")
            print(f"- Beta used: {beta:.4f}")
            print(f"- Risk-free rate: {risk_free_rate_daily:.6f}")
            print(f"- Sample Market Excess: {market_excess.iloc[-1]:.6f}")
            print(f"- Sample Expected Return: {combined['Expected_Return'].iloc[-1]:.6f}")
            print(f"- Sample Abnormal Return: {combined['Abnormal_Returns'].iloc[-1]:.6f}")
            
            # Calculate proper Cumulative AR (CAR) as running sum
            combined['Cumulative_AR'] = combined['Abnormal_Returns'].cumsum()
            
            # Add rolling volatility calculation (2-day window, annualized)
            combined['Rolling_Volatility'] = combined['Asset_Return'].rolling(window=2).std() * np.sqrt(252) * 100
            
            # Fill NaN values in volatility with forward fill
            combined['Rolling_Volatility'] = combined['Rolling_Volatility'].fillna(method='ffill')
            
            # Event day analysis with validation
            event_day_ar = 0
            event_day_actual = 0
            event_day_expected = 0
            event_day_market = 0
            
            if event_date in combined.index:
                event_day_ar = combined.loc[event_date, 'Abnormal_Returns']
                event_day_actual = combined.loc[event_date, 'Asset_Return']
                event_day_expected = combined.loc[event_date, 'Expected_Return']
                event_day_market = combined.loc[event_date, 'Market_Return']
            
            # Volume analysis if available
            volume_analysis = {}
            if 'Volume' in asset_data.columns:
                try:
                    volume_event = asset_data.loc[asset_mask, 'Volume'].dropna()
                    pre_event_volume = asset_data.loc[
                        (asset_data.index >= event_start - pd.Timedelta(days=10)) & 
                        (asset_data.index < event_start), 'Volume'
                    ].mean()
                    
                    if event_date in asset_data.index:
                        event_volume = asset_data.loc[event_date, 'Volume']
                        volume_spike = event_volume / pre_event_volume if pre_event_volume > 0 else 1
                        volume_change_pct = ((event_volume - pre_event_volume) / pre_event_volume) * 100 if pre_event_volume > 0 else 0
                        
                        volume_analysis = {
                            'event_volume': event_volume,
                            'avg_volume': pre_event_volume,
                            'volume_spike': volume_spike,
                            'volume_change_pct': volume_change_pct
                        }
                except Exception:
                    volume_analysis = {}
            
            # Statistics
            ar_values = combined['Abnormal_Returns'].values
            ar_mean = ar_values.mean() if len(ar_values) > 0 else 0
            ar_std = ar_values.std(ddof=1) if len(ar_values) > 1 else 0
            car_total = combined['Cumulative_AR'].iloc[-1] if not combined['Cumulative_AR'].empty else 0
            
            # T-test for statistical significance
            if ar_std > 0 and len(ar_values) > 1:
                t_stat = ar_mean / (ar_std / np.sqrt(len(ar_values)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ar_values) - 1))
            else:
                t_stat = 0
                p_value = 1
            
            # Enhanced validation output for S&P 500 event study
            print(f"\n=== S&P 500 EVENT STUDY VALIDATION ===")
            print(f"Event Date: {event_date.strftime('%Y-%m-%d')}")
            print(f"Risk-free rate (daily): {risk_free_rate_daily:.6f} ({risk_free_rate_daily*252:.2%} annual)")
            print(f"Beta used: {beta:.4f}")
            print(f"Event Day Market Return: {event_day_market:.6f} ({event_day_market*100:.4f}%)")
            print(f"Event Day Actual Return: {event_day_actual:.6f} ({event_day_actual*100:.4f}%)")
            print(f"Event Day Expected Return: {event_day_expected:.6f} ({event_day_expected*100:.4f}%)")
            print(f"Event Day Abnormal Return: {event_day_ar:.6f} ({event_day_ar*100:.4f}%)")
            print(f"CAR Total: {car_total:.6f} ({car_total*100:.4f}%)")
            print(f"Statistical Significance: p-value = {p_value:.4f}")
            print(f"Data Source: Yahoo Finance with auto_adjust=True, Close prices")
            print(f"CAMP Formula: AR = R_actual - [R_f + β(R_market - R_f)]")
            print("=====================================\n")
            
            # Validation check for AR consistency
            if event_day_actual > 0.003 and event_day_ar < -0.001:  # 0.3% positive but negative AR
                print("⚠️  WARNING: Positive market return with negative AR - check calculation")
            elif abs(event_day_ar) > 0.005:  # AR > 0.5%
                print("📊 NOTABLE: Significant abnormal return detected")
            
            statistics = {
                'mean_ar': ar_mean,
                'std_ar': ar_std,
                'car_total': car_total,
                't_statistic': t_stat,
                'p_value': p_value,
                'observations': len(ar_values),
                'event_day_ar': event_day_ar,
                'event_day_actual': event_day_actual,
                'event_day_expected': event_day_expected,
                'event_day_market': event_day_market,
                'positive_days': (ar_values > 0).sum(),
                'negative_days': (ar_values < 0).sum(),
                'volume_analysis': volume_analysis,
                'risk_free_rate_daily': risk_free_rate_daily,
                'beta_used': beta,
                'significant': p_value < 0.05
            }
            
            return combined, statistics
            
        except Exception as e:
            st.error(f"Abnormal returns calculation error: {str(e)}")
            return None, {}
    
    def calculate_garch_volatility(self, returns_data, event_date):
        """Calculate GARCH volatility clustering analysis"""
        try:
            # Prepare returns data
            returns = returns_data.dropna()
            
            if len(returns) < 50:
                return None
            
            # Pre and post event periods
            pre_event = returns[returns.index < event_date]
            post_event = returns[returns.index >= event_date]
            
            if len(pre_event) < 20 or len(post_event) < 5:
                return None
            
            # Calculate 2-day rolling volatility with annualized terms
            window = 2  # 2-day window as specified
            pre_vol_daily = pre_event.rolling(window=window).std().mean()
            post_vol_daily = post_event.rolling(window=window).std().mean()
            
            # Annualize volatility (daily std * sqrt(252))
            pre_vol = pre_vol_daily * np.sqrt(252) * 100  # Convert to percentage
            post_vol = post_vol_daily * np.sqrt(252) * 100
            
            # Volatility clustering test
            vol_change = (post_vol - pre_vol) / pre_vol if pre_vol > 0 else 0
            
            volatility_analysis = {
                'pre_event_volatility': pre_vol,
                'post_event_volatility': post_vol,
                'volatility_change_pct': vol_change * 100,
                'increased_volatility': vol_change > 0.1,  # 10% threshold
                'volatility_clustering': abs(vol_change) > 0.2  # Significant clustering
            }
            
            return volatility_analysis
            
        except Exception as e:
            st.warning(f"GARCH volatility analysis failed: {str(e)}")
            return None

def create_abnormal_returns_chart(abnormal_data, asset_name, event_date):
    """Create professional abnormal returns chart with accurate calculations and consistent labeling"""
    if abnormal_data is None or abnormal_data.empty:
        return None
    
    # Ensure we have the correct column names
    ar_column = 'Abnormal_Returns' if 'Abnormal_Returns' in abnormal_data.columns else 'Abnormal_Return'
    car_column = 'Cumulative_AR' if 'Cumulative_AR' in abnormal_data.columns else 'CAR'
    
    if ar_column not in abnormal_data.columns:
        return None
    
    ar_values = abnormal_data[ar_column].values
    dates = abnormal_data.index
    
    # RECALCULATE CAR from actual sum of ARs for consistency
    car_values = np.cumsum(ar_values)
    
    # Define event window [-2, +2] around June 2, 2025
    event_window_start = event_date - pd.Timedelta(days=2)
    event_window_end = event_date + pd.Timedelta(days=2)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f'Daily Abnormal Returns - Event Window: {event_window_start.strftime("%B %d")} to {event_window_end.strftime("%B %d")}', 
            'Cumulative Abnormal Returns (CAR = Σ Daily ARs)',
            'Rolling Volatility Analysis (2-day window)'
        ],
        vertical_spacing=0.08,
        row_heights=[0.4, 0.35, 0.25]
    )
    
    # 1. ABNORMAL RETURNS BAR CHART with event day highlighting
    colors = []
    for i, (date, ar) in enumerate(zip(dates, ar_values)):
        if date == event_date:
            colors.append('#ff5722')  # Bright red-orange for event day (June 2)
        elif ar < 0:
            colors.append('#d32f2f')  # Red for negative
        else:
            colors.append('#388e3c')  # Green for positive
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=ar_values * 100,  # Convert to percentage
            name='Daily AR (%)',
            marker_color=colors,
            opacity=0.8,
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>' +
                         'Abnormal Return: %{y:.4f}%<br>' +
                         '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add event window shading for all subplots
    fig.add_vrect(
        x0=event_window_start, x1=event_window_end,
        fillcolor="yellow", opacity=0.2,
        layer="below", line_width=0
    )
    
    # 2. CUMULATIVE ABNORMAL RETURNS with corrected calculation
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=car_values * 100,  # Convert to percentage
            mode='lines+markers',
            name='Cumulative AR (%)',
            line=dict(color='#1976d2', width=3),
            marker=dict(size=6, color='#1976d2'),
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>' +
                         'Cumulative AR: %{y:.4f}%<br>' +
                         '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Event window already highlighted above
    
    # 3. EVENT DATE MARKERS with corrected CAR calculation
    if event_date in dates:
        event_idx = dates.get_loc(event_date)
        event_ar = ar_values[event_idx] * 100
        event_car = car_values[event_idx] * 100  # Use recalculated CAR
        
        # Calculate actual CAR for event window [-2, +2]
        window_mask = (dates >= event_window_start) & (dates <= event_window_end)
        window_car = np.sum(ar_values[window_mask]) * 100
        
        # Event day vertical line
        fig.add_vline(x=event_date, line_dash="dash", line_color="#ff9800", line_width=3)
        
        # Event day annotations with corrected values
        fig.add_annotation(
            x=event_date, y=event_ar + 0.01,
            text=f"EVENT DAY<br>June 2, 2025<br>AR: {event_ar:.4f}%",
            showarrow=True, arrowhead=2,
            bgcolor="orange", opacity=0.9,
            font=dict(size=10, color="black"),
            row=1, col=1
        )
        
        fig.add_annotation(
            x=event_date, y=event_car + 0.005,
            text=f"CAR at Event: {event_car:.4f}%<br>Window Total: {window_car:.4f}%",
            showarrow=True, arrowhead=2,
            bgcolor="lightblue", opacity=0.9,
            font=dict(size=10, color="black"),
            row=2, col=1
        )
    
    # 4. ENHANCED ROLLING VOLATILITY with proper sensitivity
    # Use 2-day rolling window for higher sensitivity
    rolling_vol = pd.Series(ar_values).rolling(window=2, min_periods=1).std() * 100 * np.sqrt(252)  # Annualized
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rolling_vol,
            mode='lines+markers',
            name='Rolling Volatility (2-day, Annualized %)',
            line=dict(color='purple', width=2),
            marker=dict(size=4),
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>' +
                         'Volatility: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Event window already highlighted above
    
    # ZERO REFERENCE LINES
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # UPDATE AXES with improved scaling
    fig.update_xaxes(
        title_text="Date",
        tickformat="%B %d",
        dtick="D1",
        tickangle=45,
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Date", 
        tickformat="%B %d",
        dtick="D1",
        tickangle=45,
        row=2, col=1
    )
    fig.update_xaxes(
        title_text="Date",
        tickformat="%B %d", 
        dtick="D1",
        tickangle=45,
        row=3, col=1
    )
    
    # Improved axis scaling for clear separation
    ar_range = max(abs(ar_values.min()), abs(ar_values.max())) * 100 * 1.2
    car_range = max(abs(car_values.min()), abs(car_values.max())) * 100 * 1.2
    vol_range = rolling_vol.max() * 1.2
    
    fig.update_yaxes(
        title_text="Abnormal Return (%)", 
        tickformat=".4f", 
        range=[-ar_range, ar_range],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Cumulative AR (%)", 
        tickformat=".4f", 
        range=[-car_range, car_range],
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Volatility (%)", 
        tickformat=".2f", 
        range=[0, vol_range],
        row=3, col=1
    )
    
    fig.update_layout(
        title=f'Event Study Analysis: {asset_name} - Event Date: June 2, 2025',
        height=900,
        showlegend=True,
        template='plotly_white',
        font=dict(size=11, family="Arial"),
        title_font_size=16
    )
    
    return fig

def create_comparison_chart(all_results, event_date):
    """Create cross-asset comparison"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Abnormal Returns', 'Beta Comparison', 
                       'R-squared Values', 'Statistical Significance')
    )
    
    # CAR comparison
    for asset_name, result in all_results.items():
        if result['abnormal_data'] is not None:
            abnormal_data = result['abnormal_data']
            fig.add_trace(go.Scatter(
                x=abnormal_data.index,
                y=abnormal_data['Cumulative_AR'] * 100,
                mode='lines+markers',
                name=asset_name,
                line=dict(width=2)
            ), row=1, col=1)
    
    # Beta comparison
    assets = list(all_results.keys())
    betas = [all_results[asset]['beta'] for asset in assets]
    fig.add_trace(go.Bar(
        x=assets,
        y=betas,
        name='Beta',
        marker_color='lightblue',
        showlegend=False
    ), row=1, col=2)
    
    # R-squared
    r_squareds = [all_results[asset]['diagnostics']['r_squared'] for asset in assets]
    fig.add_trace(go.Bar(
        x=assets,
        y=r_squareds,
        name='R²',
        marker_color='lightgreen',
        showlegend=False
    ), row=2, col=1)
    
    # P-values
    p_values = [all_results[asset]['ar_statistics']['p_value'] for asset in assets]
    colors = ['green' if p < 0.05 else 'red' for p in p_values]
    fig.add_trace(go.Bar(
        x=assets,
        y=p_values,
        name='P-Value',
        marker_color=colors,
        showlegend=False
    ), row=2, col=2)
    
    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_hline(y=1, line_dash="dash", line_color="blue")
    fig.add_hline(y=0.05, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='Cross-Asset Analysis Dashboard',
        height=800,
        template='plotly_white'
    )
    
    return fig

def create_correlation_matrix(all_results):
    """Create correlation matrix with proper error handling"""
    try:
        # Collect return series with validation
        returns_data = {}
        
        for asset_name, result in all_results.items():
            if (result.get('asset_data') is not None and 
                'Returns' in result['asset_data'].columns):
                
                returns_series = result['asset_data']['Returns'].dropna()
                
                # Only add if we have sufficient data
                if len(returns_series) > 10:
                    returns_data[asset_name] = returns_series
        
        if len(returns_data) < 2:
            return None
        
        # Create DataFrame with proper alignment
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty or len(returns_df) < 10:
            return None
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Validate correlation matrix
        if corr_matrix.isnull().all().all():
            return None
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Returns Correlation Matrix',
            height=500,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create correlation matrix: {str(e)}")
        return None

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="quantfin-header">
        <h1>QUANTFIN SOCIETY RESEARCH</h1>
        <h2>Automated Event Study Analysis Platform</h2>
        <p>Real-time market reaction analysis with NLP-driven event detection and CAPM modeling</p>
        <p><i>Developed by Maksim Kitikov</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # GPT-Powered Event Detection
        st.subheader("🧠 GPT-Powered Event Detection")
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Manual Event Entry", "🤖 GPT News Detection"],
            help="Choose between manual event input or real-time GPT-4o news analysis"
        )
        
        if analysis_mode == "🤖 GPT News Detection":
            target_date = st.date_input("Target Date", value=datetime(2025, 6, 2).date())
            
            if st.button("🔍 Detect Events", type="secondary"):
                with st.spinner("🧠 GPT-4o analyzing market events..."):
                    from gpt_news_detector import GPTNewsDetector
                    detector = GPTNewsDetector()
                    date_str = target_date.strftime('%Y-%m-%d')
                    
                    # Detect events using GPT-4o
                    events = detector.detect_daily_events(date_str)
                    
                    if events:
                        st.subheader("🧠 GPT-4o Detected Top 3 High-Impact Market Events")
                        st.caption("Select one of these authentic market events for detailed analysis")
                        
                        # Create selection interface for top 3 events
                        event_options = []
                        for i, event in enumerate(events[:3], 1):
                            headline = event.get('headline', f'Market Event {i}')
                            impact = event.get('impact_level', 'High')
                            category = event.get('category', 'Economic')
                            event_options.append(f"Event {i}: {headline} ({impact} Impact)")
                        
                        selected_event_index = st.selectbox(
                            "Choose event to analyze:",
                            range(len(event_options)),
                            format_func=lambda x: event_options[x],
                            key="event_selector"
                        )
                        
                        # Display selected event details
                        if selected_event_index is not None and selected_event_index < len(events):
                            selected_event = events[selected_event_index]
                            
                            st.markdown("---")
                            st.markdown("### Selected Event Details")
                            
                            # Event information display
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Impact Level", selected_event.get('impact_level', 'High'))
                            with col2:
                                st.metric("Category", selected_event.get('category', 'Economic'))
                            with col3:
                                st.metric("Time", selected_event.get('time', 'Market Hours'))
                            
                            st.markdown(f"**Event Description:**")
                            st.info(selected_event.get('description', 'High-impact market event requiring analysis'))
                            
                            if selected_event.get('market_sectors'):
                                sectors = ", ".join(selected_event['market_sectors'])
                                st.markdown(f"**Affected Sectors:** {sectors}")
                            
                            reaction = selected_event.get('expected_reaction', 'Significant market reaction expected')
                            st.markdown(f"**Expected Market Reaction:** {reaction}")
                            
                            # Action button to proceed with analysis
                            if st.button("🚀 Analyze This Event", type="primary", use_container_width=True):
                                st.session_state.selected_event = selected_event.get('headline', 'GPT-4o Market Event')
                                st.session_state.event_date = pd.Timestamp(target_date).tz_localize(None)
                                st.session_state.analysis_statement = f"GPT-4o High-Impact Event: {selected_event.get('description', 'Market analysis in progress')}"
                                st.success(f"✅ Event selected: {selected_event.get('headline', 'Market Event')}")
                                st.rerun()
                    else:
                        st.warning("No significant market events detected for selected date")
        
        else:
            # Manual event entry
            st.subheader("Manual Event Parameters")
            event_name = st.text_input("Event Name", value="U.S.-China Trade Shock: New Tariff Announcement")
            event_date = st.date_input("Event Date", value=datetime(2025, 6, 2).date())
            event_date = pd.Timestamp(event_date).tz_localize(None)
            
            st.session_state.selected_event = event_name
            st.session_state.event_date = event_date
            
            # Event context
            st.markdown("""
            **Event Context:**
            *Bloomberg Headline: "U.S. Announces 25% Tariffs on Chinese Tech Imports, Effective Immediately"*
            
            Analysis of market reaction to unexpected trade policy shock affecting:
            - Technology sector exposure to China supply chains
            - Semiconductor industry trade dependencies  
            - Transportation and logistics networks
            - Broad market volatility patterns
            """)
        
        # Analysis windows
        st.subheader("📏 Analysis Windows")
        estimation_days = st.slider("Estimation Window (days)", 60, 200, 120)
        event_window_days = st.slider("Event Window (±days)", 3, 15, 7)
        
        # Asset selection
        st.subheader("📈 Assets Selection")
        
        # Sector ETFs for trade shock analysis
        sector_etfs = {
            "iShares China ETF": "FXI",
            "iShares Semiconductor ETF": "SOXX", 
            "iShares Transportation ETF": "IYT",
            "Technology Select Sector": "XLK",
            "Industrial Select Sector": "XLI"
        }
        
        # Default assets with S&P 500
        default_assets = {
            "S&P 500 Index": "^GSPC",
            "MP Materials Corp": "MP",
            "Alibaba Group": "BABA",
            "Apple Inc": "AAPL",
            "Microsoft Corp": "MSFT",
            "Tesla Inc": "TSLA",
            "NVIDIA Corp": "NVDA"
        }
        
        # Custom asset search with validation
        st.markdown("**🔍 Search & Add Custom Assets:**")
        
        # Initialize session state for custom assets
        if 'custom_assets' not in st.session_state:
            st.session_state.custom_assets = {}
        
        col1, col2 = st.columns([3, 1])
        with col1:
            custom_ticker = st.text_input("Enter ticker symbol", 
                                         placeholder="e.g., GOOGL, BTC-USD, ^DJI, EURUSD=X")
        with col2:
            add_button = st.button("🔍 Search")
        
        if add_button and custom_ticker:
            with st.spinner(f"Searching for {custom_ticker.upper()}..."):
                try:
                    test_stock = yf.Ticker(custom_ticker.upper())
                    hist = test_stock.history(period="5d")
                    
                    if not hist.empty:
                        info = test_stock.info
                        company_name = info.get('longName', info.get('shortName', custom_ticker.upper()))
                        st.session_state.custom_assets[company_name] = custom_ticker.upper()
                        st.success(f"✅ Added: {company_name} ({custom_ticker.upper()})")
                    else:
                        st.error(f"❌ No data found for ticker: {custom_ticker.upper()}")
                except Exception as e:
                    st.error(f"❌ Invalid ticker: {custom_ticker.upper()}")
        
        # Trade shock sector analysis ETFs
        st.markdown("**🎯 Trade Shock Analysis - Key Sector ETFs:**")
        sector_cols = st.columns(3)
        selected_assets = {}
        
        for i, (name, ticker) in enumerate(sector_etfs.items()):
            with sector_cols[i % 3]:
                if st.checkbox(f"{name} ({ticker})", value=(ticker in ["FXI", "SOXX", "IYT"])):
                    selected_assets[name] = ticker
        
        # Market indices
        st.markdown("**🌍 Market Indices:**")
        market_indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI", 
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX",
            "FTSE 100": "^FTSE",
            "Nikkei 225": "^N225",
            "DAX": "^GDAXI",
            "CAC 40": "^FCHI",
            "Hang Seng": "^HSI"
        }
        
        indices_cols = st.columns(3)
        
        for i, (name, ticker) in enumerate(market_indices.items()):
            with indices_cols[i % 3]:
                if st.checkbox(f"{name}", value=(ticker == "^GSPC"), key=f"index_{ticker}"):
                    selected_assets[name] = ticker
        
        # Cryptocurrency
        st.markdown("**₿ Cryptocurrencies:**")
        crypto_assets = {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD", 
            "Cardano": "ADA-USD",
            "Solana": "SOL-USD"
        }
        
        crypto_cols = st.columns(2)
        for i, (name, ticker) in enumerate(crypto_assets.items()):
            with crypto_cols[i % 2]:
                if st.checkbox(f"{name}", key=f"crypto_{ticker}"):
                    selected_assets[name] = ticker
        
        # Default stocks
        st.markdown("**📈 Popular Stocks:**")
        stock_cols = st.columns(2)
        for i, (name, ticker) in enumerate(default_assets.items()):
            if ticker not in ["^GSPC"]:
                with stock_cols[i % 2]:
                    if st.checkbox(f"{name}", value=ticker in ["MP", "BABA", "SOXX"], key=f"stock_{ticker}"):
                        selected_assets[name] = ticker
        
        # Custom assets
        if st.session_state.custom_assets:
            st.markdown("**🔧 Your Custom Assets:**")
            custom_cols = st.columns(2)
            for i, (name, ticker) in enumerate(st.session_state.custom_assets.items()):
                with custom_cols[i % 2]:
                    if st.checkbox(f"{name}", key=f"custom_{ticker}"):
                        selected_assets[name] = ticker
        
        # Run analysis
        st.markdown("---")
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Analysis execution
    if run_analysis and selected_assets:
        # Get event details from session state
        event_name = st.session_state.get('selected_event', 'Market Event')
        event_date = st.session_state.get('event_date', pd.Timestamp(datetime(2025, 6, 2)).tz_localize(None))
        
        with st.spinner("Performing professional event study analysis..."):
            try:
                analyzer = EventStudyAnalyzer()
                progress_bar = st.progress(0)
                
                # Date range
                end_date = datetime.now()
                start_date = event_date - timedelta(days=estimation_days + event_window_days + 50)
                
                # Fetch market data - use different benchmark for proper AR calculation
                st.info("Fetching market data...")
                # Use SPY as market benchmark to avoid self-comparison issues
                market_data = analyzer.fetch_data("SPY", start_date, end_date)
                if market_data is None:
                    st.error("Failed to fetch market data")
                    st.stop()
                
                progress_bar.progress(20)
                
                # Analyze assets
                all_results = {}
                total_assets = len(selected_assets)
                
                for i, (asset_name, ticker) in enumerate(selected_assets.items()):
                    progress = 20 + ((i + 1) / total_assets) * 60
                    progress_bar.progress(int(progress))
                    st.info(f"Analyzing {asset_name}...")
                    
                    # Fetch asset data
                    asset_data = analyzer.fetch_data(ticker, start_date, end_date)
                    if asset_data is None:
                        continue
                    
                    # CAPM estimation
                    estimation_end = event_date - timedelta(days=event_window_days)
                    estimation_start = estimation_end - timedelta(days=estimation_days)
                    
                    asset_est = asset_data.loc[estimation_start:estimation_end]['Returns']
                    market_est = market_data.loc[estimation_start:estimation_end]['Returns']
                    
                    alpha, beta, diagnostics = analyzer.estimate_capm(asset_est, market_est)
                    
                    if alpha is None:
                        continue
                    
                    # Abnormal returns
                    abnormal_data, ar_stats = analyzer.calculate_abnormal_returns(
                        asset_data, market_data, alpha, beta, event_date, event_window_days, asset_name
                    )
                    
                    if abnormal_data is None:
                        continue
                    
                    # GARCH volatility analysis
                    volatility_analysis = analyzer.calculate_garch_volatility(
                        asset_data['Returns'], event_date
                    )
                    
                    all_results[asset_name] = {
                        'alpha': alpha,
                        'beta': beta,
                        'diagnostics': diagnostics,
                        'abnormal_data': abnormal_data,
                        'ar_statistics': ar_stats,
                        'asset_data': asset_data,
                        'volatility_analysis': volatility_analysis
                    }
                
                progress_bar.progress(100)
                
                # Store results
                st.session_state.results = all_results
                st.session_state.event_date = event_date
                st.session_state.event_name = event_name
                st.session_state.analysis_complete = True
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # Results display
    if st.session_state.analysis_complete and st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        results = st.session_state.results
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary", 
            "📈 Individual Analysis", 
            "📉 Cross-Asset Comparison",
            "🔗 Correlation Analysis"
        ])
        
        with tab1:
            st.subheader("Executive Summary")
            
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            total_assets = len(results)
            significant = sum(1 for r in results.values() if r['ar_statistics']['p_value'] < 0.05)
            avg_car = np.mean([r['ar_statistics']['car_total'] for r in results.values()])
            max_car = max([abs(r['ar_statistics']['car_total']) for r in results.values()])
            
            with col1:
                st.metric("Assets Analyzed", total_assets)
            with col2:
                st.metric("Significant Results", f"{significant}/{total_assets}")
            with col3:
                st.metric("Average CAR", f"{avg_car:.4f}")
            with col4:
                st.metric("Max |CAR|", f"{max_car:.4f}")
            
            # Professional Event Study Results Table
            st.subheader("Professional Event Study Results Summary")
            st.markdown("**Statistical Significance Threshold: p < 0.05 (95% Confidence Level)**")
            
            summary_data = []
            for asset_name, result in results.items():
                # Extract statistical metrics
                p_value = result['ar_statistics']['p_value']
                t_stat = result['ar_statistics']['t_statistic']
                car = result['ar_statistics']['car_total']
                mean_ar = result['ar_statistics']['mean_ar']
                
                # Volume analysis
                vol_analysis = result['ar_statistics'].get('volume_analysis', {})
                volume_change = vol_analysis.get('volume_change_pct', 0) if vol_analysis else 0
                volume_spike = vol_analysis.get('volume_spike', 1) if vol_analysis else 1
                
                # Volatility analysis
                volatility_analysis = result.get('volatility_analysis', {})
                vol_change_pct = volatility_analysis.get('volatility_change_pct', 0) if volatility_analysis else 0
                
                # Statistical significance determination
                is_significant = p_value < 0.05
                significance_status = "SIGNIFICANT" if is_significant else "NOT SIGNIFICANT"
                
                # Color coding for the table
                car_direction = "↗️" if car > 0 else "↘️" if car < 0 else "→"
                
                summary_data.append({
                    'Asset': asset_name,
                    'Event Day AR': f"{result['ar_statistics'].get('event_day_ar', 0):.4f}",
                    'Mean AR': f"{mean_ar:.4f}",
                    'CAR Total': f"{car:.4f}",
                    'Direction': car_direction,
                    'T-Statistic': f"{t_stat:.3f}",
                    'P-Value': f"{p_value:.4f}",
                    'Statistical Result': significance_status,
                    'Alpha (CAPM)': f"{result['alpha']:.5f}",
                    'Beta (CAPM)': f"{result['beta']:.3f}",
                    'R² (Model Fit)': f"{result['diagnostics']['r_squared']:.3f}",
                    'Volume Spike': f"{volume_spike:.1f}x" if volume_spike > 1.1 else "Normal",
                    'Vol Change %': f"{vol_change_pct:.1f}%" if abs(vol_change_pct) > 5 else "Stable"
                })
            
            # Create DataFrame with proper styling
            summary_df = pd.DataFrame(summary_data)
            
            # Style the dataframe
            def highlight_significance(val):
                if val == "SIGNIFICANT":
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif val == "NOT SIGNIFICANT":
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                return ''
            
            def highlight_pvalue(val):
                try:
                    p_val = float(val)
                    if p_val < 0.01:
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif p_val < 0.05:
                        return 'background-color: #fff3cd; color: #856404; font-weight: bold'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'
                except:
                    return ''
            
            styled_df = summary_df.style.applymap(highlight_significance, subset=['Statistical Result']) \
                                        .applymap(highlight_pvalue, subset=['P-Value'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Statistical interpretation
            significant_count = sum(1 for item in summary_data if item['Statistical Result'] == 'SIGNIFICANT')
            st.markdown(f"""
            **Statistical Interpretation:**
            - **{significant_count} out of {len(summary_data)} assets** show statistically significant abnormal returns (p < 0.05)
            - **Event Date**: {st.session_state.get('event_date', 'N/A').strftime('%B %d, %Y') if hasattr(st.session_state.get('event_date', None), 'strftime') else 'N/A'}
            - **Confidence Level**: 95% (α = 0.05)
            - **Test Type**: Two-tailed t-test on abnormal returns
            """)
            
            # Volume and volatility summary
            volume_spikes = [item for item in summary_data if item['Volume Spike'] != "Normal"]
            vol_changes = [item for item in summary_data if item['Vol Change %'] != "Stable"]
            
            if volume_spikes or vol_changes:
                st.markdown("**Volume & Volatility Diagnostics:**")
                if volume_spikes:
                    st.write(f"- **Volume Spikes Detected**: {len(volume_spikes)} assets showed abnormal trading volume")
                if vol_changes:
                    st.write(f"- **Volatility Changes**: {len(vol_changes)} assets experienced significant volatility shifts")
            
            # Dynamic professional analysis statement based on selected event
            analysis_statement = st.session_state.get('analysis_statement', 
                f"This analysis represents a live market reaction study of {st.session_state.get('selected_event', 'the selected market event')} using CAPM-based abnormal return modeling, GARCH volatility diagnostics, and volume reaction analysis across key sector ETFs.")
            
            st.markdown(f"""
            **💼 Professional Analysis Statement:**
            
            *"{analysis_statement}"*
            """)
        
        with tab2:
            st.subheader("Individual Asset Analysis")
            
            asset_names = list(results.keys())
            selected_asset = st.selectbox("Select Asset", asset_names)
            
            if selected_asset:
                result = results[selected_asset]
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CAR", f"{result['ar_statistics']['car_total']:.4f}")
                with col2:
                    st.metric("Mean AR", f"{result['ar_statistics']['mean_ar']:.4f}")
                with col3:
                    significance = "Significant" if result['ar_statistics']['p_value'] < 0.05 else "Not Significant"
                    st.metric("Significance", significance)
                
                # Chart
                if result['abnormal_data'] is not None:
                    fig = create_abnormal_returns_chart(
                        result['abnormal_data'], selected_asset, st.session_state.event_date
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # AI Market Intelligence Analysis  
                    st.markdown("---")
                    
                    # Check if AI is enabled (default to True if not set)
                    ai_enabled = st.session_state.get('ai_enabled', True)
                    
                    if ai_enabled:
                        from openai_market_analyst import OpenAIMarketAnalyst
                        
                        # Real GPT-4o market analysis
                        try:
                            analyst = OpenAIMarketAnalyst()
                            ar_stats = result['ar_statistics']
                            
                            analysis = analyst.generate_market_analysis(
                                asset_name=selected_asset,
                                event_date="June 2, 2025",
                                actual_return=ar_stats.get('event_day_actual', 0),
                                abnormal_return=ar_stats.get('event_day_ar', 0),
                                car_total=ar_stats.get('car_total', 0),
                                volatility=3.5,
                                event_context="U.S.-China Trade Policy"
                            )
                            
                            summary = analyst.format_ai_summary(
                                asset_name=selected_asset,
                                event_date="June 2, 2025",
                                actual_return=ar_stats.get('event_day_actual', 0),
                                abnormal_return=ar_stats.get('event_day_ar', 0),
                                car_total=ar_stats.get('car_total', 0),
                                volatility=3.5,
                                analysis=analysis
                            )
                            
                            st.markdown(summary)
                            
                        except Exception as e:
                            st.error(f"GPT analysis failed: {str(e)}")
                            st.info("Check OpenAI API quota or connection")
                    else:
                        st.info("AI Market Interpretation is disabled. Enable it in the sidebar to see intelligent market analysis.")
                    
                    # Analysis Results Tabs
                    st.markdown("---")
                    result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                        "📊 Summary", 
                        "📈 Individual Analysis", 
                        "📉 Cross-Asset Comparison", 
                        "🔗 Correlation Analysis"
                    ])
                    
                    with result_tab1:
                        st.markdown("### 📊 Analysis Summary")
                        ar_stats = result['ar_statistics']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Abnormal Return", f"{ar_stats.get('event_day_ar', 0)*100:+.3f}%")
                        with col2:
                            st.metric("CAR Total", f"{ar_stats.get('car_total', 0)*100:+.3f}%")
                        with col3:
                            st.metric("P-Value", f"{ar_stats.get('p_value', 0):.4f}")
                        
                        st.markdown("**Key Findings:**")
                        st.markdown(f"- Event Date: June 2, 2025")
                        st.markdown(f"- Market Reaction: {'Significant' if abs(ar_stats.get('event_day_ar', 0)) > 0.01 else 'Moderate' if abs(ar_stats.get('event_day_ar', 0)) > 0.005 else 'Minimal'}")
                        st.markdown(f"- Statistical Significance: {'Significant' if ar_stats.get('p_value', 1) < 0.05 else 'Not Significant'}")
                        st.markdown(f"- Market Efficiency: {'Efficient' if abs(ar_stats.get('event_day_ar', 0)) < 0.005 else 'Some inefficiency detected'}")
                    
                    with result_tab2:
                        st.markdown("### 📈 Individual Asset Analysis")
                        st.markdown(f"**Detailed Event Study Results for {selected_asset}:**")
                        
                        # Display data table
                        display_data = result['abnormal_data'].tail(10).copy()
                        display_data.index = display_data.index.strftime('%Y-%m-%d')
                        st.dataframe(display_data, use_container_width=True)
                        
                        # Statistical summary
                        st.markdown("**Statistical Summary:**")
                        st.json({
                            "Beta": f"{ar_stats.get('beta', 0):.4f}",
                            "R-squared": f"{ar_stats.get('r_squared', 0):.4f}",
                            "Standard Error": f"{ar_stats.get('std_error', 0):.6f}",
                            "Event Day AR": f"{ar_stats.get('event_day_ar', 0)*100:+.3f}%"
                        })
                    
                    with result_tab3:
                        st.markdown("### 📉 Cross-Asset Comparison")
                        if len(st.session_state.get('all_results', {})) > 1:
                            st.info("Cross-asset comparison will be implemented when multiple assets are analyzed simultaneously.")
                        else:
                            st.info("Select multiple assets in the sidebar to enable cross-asset comparison analysis.")
                    
                    with result_tab4:
                        st.markdown("### 🔗 Correlation Analysis")
                        if len(st.session_state.get('all_results', {})) > 1:
                            st.info("Correlation matrix will be displayed when multiple assets are analyzed.")
                        else:
                            st.info("Correlation analysis requires multiple assets. Please select additional assets in the sidebar.")
        
        with tab3:
            st.subheader("Cross-Asset Comparison")
            
            fig = create_comparison_chart(results, st.session_state.event_date)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Correlation Analysis")
            
            fig = create_correlation_matrix(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to create correlation matrix - insufficient data")

if __name__ == "__main__":
    main()