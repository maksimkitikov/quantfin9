# Automated Financial Event Study Analysis System

**Developed by Maksim Kitikov**

A professional-grade quantitative finance system that automatically detects market-moving news events and performs CAPM-based event study analysis on major market indices and sector ETFs.

## Overview

This system replicates the event study methodology used by quantitative researchers and hedge funds to analyze market reactions to macroeconomic shocks. It automatically:

1. **Detects Financial Events** - Uses NewsAPI to scan major financial news sources
2. **Validates Market Reactions** - Confirms significant price/volume responses occurred
3. **Performs Event Studies** - Calculates abnormal returns using CAPM methodology
4. **Generates Professional Reports** - Produces statistical analysis with interpretation

## Key Features

- **Real-time News Detection** with NLP-based impact scoring
- **CAPM-based Abnormal Returns** calculation (AR = R_actual - R_expected)
- **Statistical Significance Testing** with t-tests and confidence intervals
- **Volatility Analysis** including GARCH-style clustering detection
- **Volume Spike Detection** for confirmation of news-driven reactions
- **Professional Reporting** with automated interpretation generation

## System Architecture

```
/event_study_project/
├── main.py                    # Streamlit web interface
├── event_study_main.py        # Complete automation pipeline
├── news.py                    # NewsAPI integration and filtering
├── market.py                  # Market data collection and validation
├── analysis.py                # CAPM event study calculations
├── README.md                  # This documentation
└── results/                   # Output directory for analysis results
```

## Quick Start

### 1. Environment Setup

Set your NewsAPI key as an environment variable:
```bash
export NEWSAPI_KEY="your_newsapi_key_here"
```

### 2. Web Interface (Streamlit)

Launch the interactive web application:
```bash
streamlit run main.py --server.port 5000
```

Features:
- Automated news detection mode
- Manual event configuration
- Real-time market reaction validation
- Interactive visualizations
- Professional results dashboard

### 3. Command Line Interface

Run automated analysis for specific dates:
```bash
python event_study_main.py --date 2025-06-02
```

This will:
- Detect the most impactful financial event for the date
- Validate market reactions occurred
- Run complete event study analysis
- Generate professional interpretation
- Save results to JSON/text files

## Analysis Methodology

### Event Detection
- Fetches financial news from Bloomberg, Reuters, WSJ, Financial Times, CNBC
- Scores headlines using financial keyword weighting
- Ranks by potential market impact
- Selects top event as analysis trigger

### Market Validation
- Analyzes S&P 500, China ETF (FXI), Semiconductor ETF (SOXX), Transportation ETF (IYT)
- Confirms significant reactions: |return| > 0.8% OR volume spike > 1.5x
- Proceeds only if market response detected

### Event Study Analysis
- **CAPM Estimation**: 60-day rolling beta vs S&P 500
- **Expected Returns**: E(R) = α + β(R_market)
- **Abnormal Returns**: AR = R_actual - E(R)
- **Cumulative AR**: CAR over [-5, +5] day window
- **Statistical Testing**: T-tests for significance at 5% and 1% levels

### Advanced Analytics
- **Volatility Clustering**: Pre/post event volatility comparison
- **Volume Analysis**: Trading spike detection and quantification
- **Cross-Asset Impact**: Sector-specific reaction patterns

## Key Symbols Analyzed

| Symbol | Description | Analysis Focus |
|--------|-------------|----------------|
| ^GSPC | S&P 500 Index | Market benchmark |
| FXI | iShares China Large-Cap ETF | China trade exposure |
| SOXX | iShares Semiconductor ETF | Technology supply chains |
| IYT | iShares Transportation ETF | Logistics networks |

## Sample Output

```
EVENT STUDY ANALYSIS - 2025-06-02
==================================================
Event: U.S. Announces 25% Tariffs on Chinese Tech Imports
Source: Bloomberg

MARKET REACTION SUMMARY:
• Assets analyzed: 4
• Significant reactions: 3
• Average return: -0.012
• Maximum volume spike: 2.3x

ABNORMAL RETURNS ANALYSIS (CAPM-based):
• China Large Cap ETF (FXI):
  - CAR (5-day): -0.0287 (-2.87%)
  - T-statistic: -3.245
  - P-value: 0.0021 (SIGNIFICANT)
  - Beta: 0.892

• Semiconductor ETF (SOXX):
  - CAR (5-day): -0.0341 (-3.41%)
  - T-statistic: -4.122
  - P-value: 0.0008 (SIGNIFICANT)
  - Beta: 1.234

INTERPRETATION:
• Market showed statistically significant abnormal returns
• Event had measurable impact on sector ETFs
• China-exposed assets showed significant reaction
• Technology/semiconductor sector significantly affected
```

## Dependencies

- **yfinance**: Market data collection
- **pandas/numpy**: Data manipulation and analysis
- **scipy**: Statistical testing
- **requests**: NewsAPI integration
- **streamlit**: Web interface
- **plotly**: Interactive visualizations

## Professional Applications

This system demonstrates quantitative finance techniques used in:

- **Event-driven Trading Strategies**
- **Risk Management Systems**
- **Market Microstructure Research**
- **Regulatory Impact Analysis**
- **Portfolio Optimization Models**

## Technical Notes

- All timestamps are handled as timezone-naive for consistency
- Market data includes proper handling of weekends/holidays
- Statistical tests include multiple significance levels
- Error handling covers API rate limits and data availability
- Results are saved in both machine-readable (JSON) and human-readable formats

## Compliance

The system uses only publicly available data sources and follows standard academic event study methodologies. It is designed for research and educational purposes in quantitative finance.

---

*This system represents professional-grade quantitative analysis comparable to tools used in institutional finance settings.*