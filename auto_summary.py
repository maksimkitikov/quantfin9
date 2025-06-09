"""
Auto Summary Module - Streamlit Integration
Automatically generates AI-powered market interpretations for the dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from openai_market_analyst import OpenAIMarketAnalyst


class AutoSummaryGenerator:
    """
    Streamlit component that automatically generates real AI interpretations
    using OpenAI GPT-4o for market event analysis results
    """
    
    def __init__(self):
        try:
            self.analyst = OpenAIMarketAnalyst()
            self.ai_enabled = True
            print("✅ OpenAI Market Analyst initialized successfully")
        except Exception as e:
            self.ai_enabled = False
            print(f"❌ OpenAI initialization failed: {e}")
    
    def generate_auto_summary(self, 
                            asset_name: str,
                            event_date: str,
                            analysis_results: Dict,
                            abnormal_data: pd.DataFrame,
                            event_context: str = "U.S.-China Trade Policy") -> str:
        """
        Generate automatic AI summary from analysis results
        
        Args:
            asset_name: Name of the analyzed asset
            event_date: Event date string
            analysis_results: Dictionary containing AR statistics
            abnormal_data: DataFrame with abnormal returns data
            event_context: Economic context for interpretation
            
        Returns:
            Formatted AI interpretation string
        """
        
        try:
            # Extract key metrics from analysis results
            ar_stats = analysis_results.get('ar_statistics', {})
            
            # Get time series data
            ar_series = self._extract_ar_series(abnormal_data)
            car_series = self._extract_car_series(abnormal_data)
            volatility_series = self._extract_volatility_series(abnormal_data)
            volume_series = self._extract_volume_series(abnormal_data)
            
            # Get exact values from ar_statistics to ensure consistency
            actual_return = ar_stats.get('event_day_actual', 0)
            abnormal_return = ar_stats.get('event_day_ar', 0)
            car_total = ar_stats.get('car_total', 0)
            
            # Calculate volatility from abnormal data if available
            volatility_value = 0
            if abnormal_data is not None and 'Rolling_Volatility' in abnormal_data.columns:
                volatility_value = abnormal_data['Rolling_Volatility'].iloc[-1] if not abnormal_data.empty else 0
            
            print(f"🧠 AI Summary Data Source Debug:")
            print(f"- Actual Return: {actual_return:.6f} ({actual_return*100:.4f}%)")
            print(f"- Abnormal Return: {abnormal_return:.6f} ({abnormal_return*100:.4f}%)")
            print(f"- CAR Total: {car_total:.6f} ({car_total*100:.4f}%)")
            print(f"- Volatility: {volatility_value:.2f}%")
            
            # Generate real AI interpretation using OpenAI GPT-4o
            if self.ai_enabled:
                try:
                    analysis = self.analyst.generate_market_analysis(
                        asset_name=asset_name,
                        event_date=event_date,
                        actual_return=actual_return,
                        abnormal_return=abnormal_return,
                        car_total=car_total,
                        volatility=volatility_value,
                        event_context=event_context
                    )
                    
                    interpretation = self.analyst.format_ai_summary(
                        asset_name=asset_name,
                        event_date=event_date,
                        actual_return=actual_return,
                        abnormal_return=abnormal_return,
                        car_total=car_total,
                        volatility=volatility_value,
                        analysis=analysis
                    )
                    
                    print("✅ Real AI analysis generated successfully using OpenAI GPT-4o")
                    
                except Exception as e:
                    print(f"❌ OpenAI analysis failed: {e}")
                    interpretation = f"**AI Analysis Error:** Unable to generate real-time interpretation - {str(e)}"
            else:
                interpretation = "**AI Disabled:** OpenAI API key not available for real-time analysis."
            
            return interpretation
            
        except Exception as e:
            return f"**AI Analysis Error:** Unable to generate interpretation - {str(e)}"
    
    def _extract_ar_series(self, data: pd.DataFrame) -> List[float]:
        """Extract abnormal returns series"""
        if data is None or 'Abnormal_Returns' not in data.columns:
            return []
        return data['Abnormal_Returns'].dropna().tolist()
    
    def _extract_car_series(self, data: pd.DataFrame) -> List[float]:
        """Extract cumulative abnormal returns series"""
        if data is None or 'Cumulative_AR' not in data.columns:
            return []
        return data['Cumulative_AR'].dropna().tolist()
    
    def _extract_volatility_series(self, data: pd.DataFrame) -> List[float]:
        """Extract volatility series"""
        if data is None or 'Rolling_Volatility' not in data.columns:
            # Calculate simple rolling volatility if not available
            if 'Abnormal_Returns' in data.columns:
                returns = data['Abnormal_Returns'].dropna()
                volatility = returns.rolling(window=2).std() * np.sqrt(252) * 100
                return volatility.dropna().tolist()
        else:
            return data['Rolling_Volatility'].dropna().tolist()
        return []
    
    def _extract_volume_series(self, data: pd.DataFrame) -> List[float]:
        """Extract volume series"""
        if data is None:
            return []
        
        # Check for volume columns
        volume_cols = ['Volume', 'volume', 'Volume_Ratio']
        for col in volume_cols:
            if col in data.columns:
                return data[col].dropna().tolist()
        
        return []
    
    def display_ai_summary(self, 
                          asset_name: str,
                          event_date: str,
                          analysis_results: Dict,
                          abnormal_data: pd.DataFrame,
                          event_context: str = "U.S.-China Trade Policy"):
        """
        Display AI summary in Streamlit interface
        
        Args:
            asset_name: Name of the analyzed asset
            event_date: Event date string
            analysis_results: Dictionary containing AR statistics
            abnormal_data: DataFrame with abnormal returns data
            event_context: Economic context for interpretation
        """
        
        # Generate the interpretation
        interpretation = self.generate_auto_summary(
            asset_name, event_date, analysis_results, abnormal_data, event_context
        )
        
        # Display in an expandable section
        with st.expander("🧠 AI Market Intelligence Analysis", expanded=True):
            st.markdown(interpretation)
    
    def get_asset_context(self, asset_name: str) -> str:
        """Determine appropriate economic context based on asset"""
        
        asset_lower = asset_name.lower()
        
        if any(keyword in asset_lower for keyword in ['china', 'fxi', 'ashr', 'alibaba', 'tencent']):
            return "U.S.-China Trade and Technology Policy"
        elif any(keyword in asset_lower for keyword in ['technology', 'tech', 'qqq', 'xlk', 'nvidia', 'apple']):
            return "Technology Sector and AI Policy"
        elif any(keyword in asset_lower for keyword in ['financial', 'bank', 'xlf', 'jpmorgan']):
            return "Financial Regulation and Interest Rate Policy"
        elif any(keyword in asset_lower for keyword in ['energy', 'oil', 'xle', 'exxon']):
            return "Energy Policy and Geopolitical Events"
        elif any(keyword in asset_lower for keyword in ['crypto', 'bitcoin', 'btc', 'ethereum']):
            return "Cryptocurrency Regulation and Market Adoption"
        else:
            return "U.S.-China Trade Policy"


def integrate_ai_summary_component(asset_name: str,
                                 event_date: str,
                                 analysis_results: Dict,
                                 abnormal_data: pd.DataFrame) -> None:
    """
    Main integration function to add AI summary to existing Streamlit app
    
    Args:
        asset_name: Name of the analyzed asset
        event_date: Event date string
        analysis_results: Dictionary containing AR statistics
        abnormal_data: DataFrame with abnormal returns data
    """
    
    # Initialize the auto summary generator
    summary_generator = AutoSummaryGenerator()
    
    # Determine appropriate context
    context = summary_generator.get_asset_context(asset_name)
    
    # Display the AI summary
    summary_generator.display_ai_summary(
        asset_name=asset_name,
        event_date=event_date,
        analysis_results=analysis_results,
        abnormal_data=abnormal_data,
        event_context=context
    )


# Streamlit component functions for easy integration
def show_ai_interpretation(asset_name: str, analysis_results: Dict, abnormal_data: pd.DataFrame):
    """Simplified function for quick integration into existing Streamlit apps"""
    
    if analysis_results and abnormal_data is not None:
        event_date = "June 2, 2025"  # Default or extract from session state
        integrate_ai_summary_component(asset_name, event_date, analysis_results, abnormal_data)
    else:
        st.info("AI interpretation will appear after analysis is complete.")


def main():
    """Test the auto summary component"""
    st.title("AI Summary Test")
    
    # Sample test data
    sample_results = {
        'ar_statistics': {
            'event_day_actual': 0.004102,
            'event_day_expected': 0.004102,
            'event_day_ar': 0.000000,
            'mean_ar': 0.0001,
            'car_total': -0.0002
        }
    }
    
    # Sample DataFrame
    sample_data = pd.DataFrame({
        'Abnormal_Returns': [0.001, -0.002, 0.000],
        'Cumulative_AR': [0.001, -0.001, -0.001],
        'Asset_Return': [0.003, 0.002, 0.004102]
    })
    
    # Test the component
    integrate_ai_summary_component(
        asset_name="S&P 500",
        event_date="June 2, 2025",
        analysis_results=sample_results,
        abnormal_data=sample_data
    )


if __name__ == "__main__":
    main()