# pages/2_Comparative_Analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_data, calculate_breakouts, simulate_trades, calculate_metrics, get_market_cap
import yfinance as yf
import numpy as np

# Set page title
st.title("🤖 Automated Comparative Analysis")

# Sidebar for strategy parameters
st.sidebar.header("Comparative Analysis Parameters")
volume_threshold_auto = st.sidebar.number_input("Volume Breakout Threshold (%)", min_value=100.0, value=200.0, step=10.0)
price_threshold_auto = st.sidebar.number_input("Price Change Threshold (%)", min_value=0.0, value=2.0, step=0.1)
holding_period_auto = st.sidebar.number_input("Holding Period (Days)", min_value=1, value=10, step=1)
start_date_auto = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date_auto = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Define tickers by sector
tickers_by_sector = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'INTC', 'CSCO'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'BMY', 'AMGN'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
    'Energy': ['XOM', 'CVX', 'BP', 'NEE', 'SHEL'],
    'Consumer Goods': ['PG', 'KO', 'PEP', 'PM', 'CL'],
    'Industrials': ['BA', 'GE', 'CAT', 'DE', 'LMT']
}

# Flatten the list of tickers for iteration
all_tickers = [ticker for tickers in tickers_by_sector.values() for ticker in tickers]

if st.button("Run Automated Analysis"):
    with st.spinner("Running automated analysis on multiple tickers..."):
        all_results = []
        for sector, tickers in tickers_by_sector.items():
            for ticker in tickers:
                st.write(f"🔍 **Analyzing {ticker} in {sector} sector...**")
                data = fetch_data(ticker, start_date_auto, end_date_auto)
                if data.empty:
                    st.warning(f"No data found for {ticker}. Skipping...")
                    continue
                breakout_days = calculate_breakouts(data, volume_threshold_auto, price_threshold_auto)
                trades = simulate_trades(data, breakout_days, holding_period_auto)
                metrics = calculate_metrics(trades)
                market_cap = get_market_cap(ticker)
                metrics.update({
                    'Ticker': ticker,
                    'Sector': sector,
                    'Market Cap': market_cap
                })
                all_results.append(metrics)
        
        # Convert the results into a DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        results_df = results_df[[
            'Ticker', 'Sector', 'Market Cap', 'Breakout Days',
            'Total Return (%)', 'Avg Return/Trade (%)',
            'Win Rate (%)', 'Max Drawdown (%)', 'Sharpe Ratio'
        ]]
        
        # Display the results
        st.success("📊 Automated Analysis Complete!")
        st.dataframe(results_df)
        
        # Download CSV
        csv_auto = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Results as CSV",
            data=csv_auto,
            file_name="automated_breakout_strategy_results.csv",
            mime="text/csv",
        )
        
        # Visualizations
        st.subheader("📈 Comparative Visualizations")
        
        # A. Bar Chart: Total Return by Ticker
        plt.figure(figsize=(14, 7))
        sns.barplot(data=results_df, x='Ticker', y='Total Return (%)', hue='Sector', palette="viridis")
        plt.title('Total Return (%) by Ticker and Sector')
        plt.xlabel('Ticker')
        plt.ylabel('Total Return (%)')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)
        plt.clf()
        
        # B. Bar Chart: Average Return per Trade by Ticker
        plt.figure(figsize=(14, 7))
        sns.barplot(data=results_df, x='Ticker', y='Avg Return/Trade (%)', hue='Sector', palette="magma")
        plt.title('Average Return per Trade (%) by Ticker and Sector')
        plt.xlabel('Ticker')
        plt.ylabel('Average Return per Trade (%)')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)
        plt.clf()
        
        # C. Scatter Plot: Win Rate vs. Average Return per Trade
        plt.figure(figsize=(14, 7))
        sns.scatterplot(data=results_df, x='Win Rate (%)', y='Avg Return/Trade (%)', hue='Sector', size='Total Return (%)', sizes=(100, 1000), alpha=0.7, palette="coolwarm")
        plt.title('Win Rate vs. Average Return per Trade')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Average Return per Trade (%)')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)
        plt.clf()
        
        # D. Heatmap: Performance Metrics
        heatmap_data = results_df.set_index('Ticker').drop(['Sector', 'Market Cap'], axis=1)
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')
        plt.title('Performance Metrics Heatmap')
        plt.xlabel('Metrics')
        plt.ylabel('Ticker')
        st.pyplot(plt)
        plt.clf()
        
        # E. Cumulative Return Plot for Each Ticker
        st.subheader("📈 Cumulative Returns of Tickers Over Time")
        plt.figure(figsize=(14, 7))
        for ticker in all_tickers:
            data = fetch_data(ticker, start_date_auto, end_date_auto)
            if data.empty:
                continue
            data['Cumulative Return'] = (1 + data['Close'].pct_change()).cumprod()
            plt.plot(data.index, data['Cumulative Return'], label=ticker)
        plt.title('Cumulative Returns of Tickers')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt)
        plt.clf()
        
        # -------------------------------
        # 6. Comparative Analysis with Benchmarks
        # -------------------------------
        
        st.subheader("📊 Benchmark Comparison with S&P 500")
        sp500 = yf.Ticker('^GSPC')
        sp500_data = sp500.history(start=start_date_auto, end=end_date_auto)
        sp500_data['Cumulative Return'] = (1 + sp500_data['Close'].pct_change()).cumprod()
        
        # Calculate benchmark total return
        if not sp500_data.empty:
            benchmark_start = sp500_data.iloc[0]['Close']
            benchmark_end = sp500_data.iloc[-1]['Close']
            benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100
            st.write(f"S&P 500 Total Return from {start_date_auto} to {end_date_auto}: **{round(benchmark_return, 2)}%**")
        else:
            benchmark_return = np.nan
            st.warning("Failed to fetch S&P 500 data.")
        
        # Compare each ticker's total return with the S&P 500 benchmark
        if not results_df.empty and not np.isnan(benchmark_return):
            results_df['Outperforms S&P 500'] = results_df['Total Return (%)'] > benchmark_return
            st.subheader("📈 Comparison with S&P 500 Benchmark")
            st.dataframe(results_df[['Ticker', 'Total Return (%)', 'Outperforms S&P 500']])
        else:
            st.warning("Cannot perform benchmark comparison due to missing data.")
        
        # -------------------------------
        # 7. Additional Insights
        # -------------------------------
        
        st.subheader("💡 Insights and Recommendations")
        
        # Identify High-Performing Sectors
        sector_performance = results_df.groupby('Sector')['Total Return (%)'].mean().reset_index()
        top_sector = sector_performance.sort_values(by='Total Return (%)', ascending=False).iloc[0]
        st.write(f"**Top Performing Sector:** {top_sector['Sector']} with an average total return of **{round(top_sector['Total Return (%)'], 2)}%**.")
        
        # Assess Parameter Sensitivity (This would require additional implementation)
        # Placeholder for future enhancements
        st.write("**Parameter Sensitivity Analysis:** Future work could include varying volume and price thresholds to assess their impact on breakout signal effectiveness.")
        
        # Highlight Risk-Adjusted Performance
        high_sharpe = results_df[results_df['Sharpe Ratio'] > 1.0]
        st.write(f"**Tickers with High Sharpe Ratios (>1.0):**")
        if not high_sharpe.empty:
            st.dataframe(high_sharpe[['Ticker', 'Sharpe Ratio']])
        else:
            st.write("No tickers with Sharpe Ratio greater than 1.0 found.")
