"""
Complete Automated Event Study System
Orchestrates news detection, market analysis, and event study calculations
"""

import argparse
import json
import os
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, Optional

# Import our modules
from news import NewsCollector
from market import MarketAnalyzer
from analysis import EventStudyAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('event_study.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomatedEventStudy:
    """Complete automated event study system"""
    
    def __init__(self):
        self.news_collector = NewsCollector()
        self.market_analyzer = MarketAnalyzer()
        self.event_analyzer = EventStudyAnalyzer()
        
        # Key market symbols for analysis
        self.symbols = ['^GSPC', 'FXI', 'SOXX', 'IYT']
        self.symbol_names = {
            '^GSPC': 'S&P 500',
            'FXI': 'China Large Cap ETF',
            'SOXX': 'Semiconductor ETF',
            'IYT': 'Transportation ETF'
        }
    
    def detect_daily_event(self, target_date: str) -> Optional[Dict]:
        """
        Detect the most impactful financial event for given date
        
        Args:
            target_date: Date in YYYY-MM-DD format
            
        Returns:
            Event information or None if no significant event found
        """
        logger.info(f"Detecting events for {target_date}")
        
        # Step 1: Fetch news
        articles = self.news_collector.fetch_news(target_date, num_articles=15)
        if not articles:
            logger.warning("No news articles found")
            return None
        
        # Step 2: Filter and rank
        relevant_articles = self.news_collector.filter_relevant_articles(articles)
        if not relevant_articles:
            logger.warning("No relevant financial articles found")
            return None
        
        ranked_articles = self.news_collector.rank_headlines(relevant_articles)
        
        # Select top event
        top_event = ranked_articles[0]
        
        logger.info(f"Top event detected: {top_event['title']}")
        logger.info(f"Impact score: {top_event.get('impact_score', 0)}")
        
        return {
            'date': target_date,
            'headline': top_event['title'],
            'description': top_event.get('description', ''),
            'source': top_event['source']['name'],
            'impact_score': top_event.get('impact_score', 0),
            'url': top_event.get('url', ''),
            'all_articles': ranked_articles[:5]  # Keep top 5 for reference
        }
    
    def validate_market_reaction(self, event_date: str) -> Dict:
        """
        Check if market showed significant reaction on event date
        
        Args:
            event_date: Event date in YYYY-MM-DD format
            
        Returns:
            Market reaction validation results
        """
        logger.info(f"Validating market reaction for {event_date}")
        
        # Load market data
        start_date, end_date = self.market_analyzer.get_trading_days_around_event(event_date)
        market_data = self.market_analyzer.load_market_data(self.symbols, start_date, end_date)
        
        if not market_data:
            logger.error("Failed to load market data")
            return {'validated': False, 'error': 'No market data available'}
        
        # Detect reactions
        reactions = self.market_analyzer.detect_market_reaction(market_data, event_date)
        summary = self.market_analyzer.calculate_market_summary(reactions)
        
        # Validation criteria: significant reaction in at least 30% of assets
        validated = summary.get('market_reaction_detected', False)
        
        logger.info(f"Market reaction validated: {validated}")
        logger.info(f"Significant reactions: {summary.get('significant_reactions', 0)}/{summary.get('total_assets', 0)}")
        
        return {
            'validated': validated,
            'summary': summary,
            'reactions': reactions,
            'market_data': market_data
        }
    
    def run_event_study_analysis(self, market_data: Dict, event_date: str) -> Dict:
        """
        Run complete event study analysis for all symbols
        
        Args:
            market_data: Market data dictionary
            event_date: Event date string
            
        Returns:
            Event study results for all symbols
        """
        logger.info(f"Running event study analysis for {event_date}")
        
        event_date_ts = pd.Timestamp(event_date)
        results = {}
        
        # Get market benchmark data
        market_benchmark = market_data.get('^GSPC')
        if market_benchmark is None:
            logger.error("No market benchmark data available")
            return {}
        
        # Analyze each symbol
        for symbol in self.symbols:
            if symbol == '^GSPC':
                continue  # Skip market benchmark itself
                
            asset_data = market_data.get(symbol)
            if asset_data is None:
                logger.warning(f"No data for {symbol}")
                continue
            
            asset_name = self.symbol_names.get(symbol, symbol)
            logger.info(f"Analyzing {asset_name} ({symbol})")
            
            # Run event study
            try:
                analysis_result = self.event_analyzer.run_complete_analysis(
                    asset_data, market_benchmark, event_date_ts, asset_name
                )
                results[symbol] = analysis_result
                
                if analysis_result['success']:
                    car = analysis_result['ar_statistics']['car_total']
                    p_val = analysis_result['ar_statistics']['p_value']
                    logger.info(f"{asset_name}: CAR={car:.4f}, p-value={p_val:.4f}")
                else:
                    logger.warning(f"{asset_name}: Analysis failed - {analysis_result.get('error', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                results[symbol] = {'success': False, 'error': str(e)}
        
        return results
    
    def generate_interpretation(self, event: Dict, market_validation: Dict, analysis_results: Dict) -> str:
        """
        Generate professional interpretation of results
        
        Args:
            event: Event information
            market_validation: Market reaction validation
            analysis_results: Event study analysis results
            
        Returns:
            Professional interpretation text
        """
        interpretation = []
        
        # Event summary
        interpretation.append(f"EVENT STUDY ANALYSIS - {event['date']}")
        interpretation.append("=" * 50)
        interpretation.append(f"Event: {event['headline']}")
        interpretation.append(f"Source: {event['source']}")
        interpretation.append("")
        
        # Market reaction summary
        summary = market_validation.get('summary', {})
        interpretation.append("MARKET REACTION SUMMARY:")
        interpretation.append(f"• Assets analyzed: {summary.get('total_assets', 0)}")
        interpretation.append(f"• Significant reactions: {summary.get('significant_reactions', 0)}")
        interpretation.append(f"• Average return: {summary.get('avg_return', 0):.3f}")
        interpretation.append(f"• Maximum volume spike: {summary.get('max_volume_spike', 1):.1f}x")
        interpretation.append("")
        
        # Individual asset analysis
        interpretation.append("ABNORMAL RETURNS ANALYSIS (CAPM-based):")
        for symbol, result in analysis_results.items():
            if not result.get('success', False):
                continue
                
            asset_name = self.symbol_names.get(symbol, symbol)
            stats = result['ar_statistics']
            
            car = stats['car_total']
            p_val = stats['p_value']
            significance = "SIGNIFICANT" if stats['significant_5pct'] else "Not significant"
            
            interpretation.append(f"• {asset_name} ({symbol}):")
            interpretation.append(f"  - CAR (5-day): {car:.4f} ({car*100:.2f}%)")
            interpretation.append(f"  - T-statistic: {stats['t_statistic']:.3f}")
            interpretation.append(f"  - P-value: {p_val:.4f} ({significance})")
            interpretation.append(f"  - Beta: {result.get('beta', 0):.3f}")
        
        interpretation.append("")
        interpretation.append("INTERPRETATION:")
        
        # Generate insights
        significant_count = sum(1 for r in analysis_results.values() 
                              if r.get('success') and r['ar_statistics']['significant_5pct'])
        
        if significant_count >= 2:
            interpretation.append("• Market showed statistically significant abnormal returns")
            interpretation.append("• Event had measurable impact on sector ETFs")
        elif significant_count == 1:
            interpretation.append("• Limited but measurable market impact detected")
        else:
            interpretation.append("• No statistically significant abnormal returns detected")
        
        # Sector-specific insights
        china_impact = analysis_results.get('FXI', {})
        tech_impact = analysis_results.get('SOXX', {})
        transport_impact = analysis_results.get('IYT', {})
        
        if china_impact.get('success') and china_impact['ar_statistics']['significant_5pct']:
            interpretation.append("• China-exposed assets showed significant reaction")
        
        if tech_impact.get('success') and tech_impact['ar_statistics']['significant_5pct']:
            interpretation.append("• Technology/semiconductor sector significantly affected")
        
        if transport_impact.get('success') and transport_impact['ar_statistics']['significant_5pct']:
            interpretation.append("• Transportation/logistics sector showed reaction")
        
        return "\n".join(interpretation)
    
    def save_results(self, event: Dict, market_validation: Dict, analysis_results: Dict, 
                    interpretation: str, output_dir: str = "results") -> None:
        """
        Save all results to files
        
        Args:
            event: Event information
            market_validation: Market validation results
            analysis_results: Analysis results
            interpretation: Text interpretation
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_date = event['date'].replace('-', '')
        
        # Save event metadata
        event_file = f"{output_dir}/event_{event_date}_{timestamp}.json"
        with open(event_file, 'w') as f:
            json.dump({
                'event': event,
                'market_validation': {
                    'validated': market_validation['validated'],
                    'summary': market_validation['summary'],
                    'reactions': market_validation['reactions']
                },
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        # Save analysis results
        results_file = f"{output_dir}/analysis_{event_date}_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert complex objects to serializable format
            serializable_results = {}
            for symbol, result in analysis_results.items():
                if result.get('success'):
                    serializable_results[symbol] = {
                        'asset_name': result['asset_name'],
                        'alpha': result['alpha'],
                        'beta': result['beta'],
                        'ar_statistics': result['ar_statistics'],
                        'volatility_analysis': result.get('volatility_analysis', {}),
                        'capm_diagnostics': result['camp_diagnostics']
                    }
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save interpretation
        interpretation_file = f"{output_dir}/interpretation_{event_date}_{timestamp}.txt"
        with open(interpretation_file, 'w') as f:
            f.write(interpretation)
        
        logger.info(f"Results saved to {output_dir}")
    
    def run_daily_analysis(self, target_date: str = None) -> Dict:
        """
        Run complete daily event study analysis
        
        Args:
            target_date: Target date (default: today)
            
        Returns:
            Complete analysis results
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting automated event study analysis for {target_date}")
        
        # Step 1: Detect events
        event = self.detect_daily_event(target_date)
        if not event:
            return {'success': False, 'error': 'No significant events detected'}
        
        # Step 2: Validate market reaction
        market_validation = self.validate_market_reaction(target_date)
        if not market_validation['validated']:
            logger.warning("No significant market reaction detected")
            # Continue analysis anyway for research purposes
        
        # Step 3: Run event study analysis
        analysis_results = self.run_event_study_analysis(
            market_validation['market_data'], target_date
        )
        
        if not analysis_results:
            return {'success': False, 'error': 'Event study analysis failed'}
        
        # Step 4: Generate interpretation
        interpretation = self.generate_interpretation(event, market_validation, analysis_results)
        
        # Step 5: Save results
        self.save_results(event, market_validation, analysis_results, interpretation)
        
        # Print interpretation
        print("\n" + interpretation)
        
        return {
            'success': True,
            'event': event,
            'market_validation': market_validation,
            'analysis_results': analysis_results,
            'interpretation': interpretation
        }

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='Automated Financial Event Study Analysis')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD format)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AutomatedEventStudy()
    
    try:
        # Run analysis
        results = system.run_daily_analysis(args.date)
        
        if results['success']:
            logger.info("Analysis completed successfully")
        else:
            logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"System error: {e}")
        raise

if __name__ == "__main__":
    main()
