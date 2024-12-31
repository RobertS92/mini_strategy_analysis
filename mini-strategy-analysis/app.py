# app.py

import streamlit as st

# Set Streamlit page config
st.set_page_config(
    page_title="Mini Strategy Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main Title
st.title("📈 Mini Strategy Analysis Project")

# Description of the App and Pages
st.markdown("""
Welcome to the **Mini Strategy Analysis Project**! This Streamlit application demonstrates two main ways to analyze stock breakouts:

1. **Manual Breakout Analysis**  
   - In the sidebar, head over to the **“Manual Breakout Analysis”** page.  
   - You can input a single ticker, pick your volume and price thresholds, and specify a holding period for backtesting.  
   - This page will fetch historical data, identify breakout days, simulate trades, and display key performance metrics (e.g., total return, win rate, Sharpe ratio).
   - You’ll also get visualizations of price, volume, and cumulative returns, with breakout points and trades highlighted.

2. **Automated Comparative Analysis**  
   - Go to the **“Automated Comparative Analysis”** page in the sidebar.
   - This functionality runs a similar breakout approach **across multiple tickers** grouped by sector (Technology, Healthcare, Finance, etc.).
   - It will simulate trades for each ticker and compile performance metrics (e.g., total return, average return/trade, Sharpe ratio).
   - Use it to quickly see which sectors or tickers have historically performed well under your breakout criteria, and compare everything in a single table and various plots.

### How to Use
- **Select a Page**  
  In the sidebar, choose **"Manual Breakout Analysis"** for a single-ticker, hands-on approach, or **"Automated Comparative Analysis"** to run a bulk backtest across many tickers.
  
- **Adjust Parameters**  
  Input your desired volume threshold, price change threshold, and holding period. For automated analysis, you can also change the date range and see how different periods affect performance.

- **Generate Results**  
  After clicking “Generate Report” or “Run Automated Analysis,” the app fetches and processes data, then provides trade simulations, performance metrics, and visualizations.

We hope this mini project demonstrates a clear, end-to-end process for researching breakout strategies, from **data fetching** to **performance reporting**. Feel free to modify thresholds, date ranges, or the sector lists to explore new ideas!
""")

st.markdown("""
---
**Additional Insights**  

After experimenting with various tickers across sectors—from high-volatility tech stocks (e.g., GOOGL, AAPL) to stable consumer staples (e.g., PG, KO), as well as financials (e.g., WFC, GS)—the breakout signal showed **mixed results**. Some tickers produced solid returns, while others gave weak or false signals. But I found it is very high varience and risky. Works better with highly volaliltie names like NVDA or GS. Overall, it can be a useful trigger for momentum-driven stocks, but likely needs more filters to improve performance and consistency.
""")
