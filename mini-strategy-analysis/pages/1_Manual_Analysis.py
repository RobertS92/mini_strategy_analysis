import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_data, calculate_breakouts, simulate_trades, calculate_metrics

# Set page title
st.title("🔍 Manual Breakout Analysis")

# Sidebar for user inputs
st.sidebar.header("Manual Analysis Parameters")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
start_date_input = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date_input = st.sidebar.date_input("End Date", pd.to_datetime("today"))
volume_threshold_input = st.sidebar.number_input("Volume Breakout Threshold (%)", min_value=100.0, value=200.0, step=10.0)
price_threshold_input = st.sidebar.number_input("Price Change Threshold (%)", min_value=0.0, value=2.0, step=0.1)
holding_period_input = st.sidebar.number_input("Holding Period (Days)", min_value=1, value=10, step=1)

if st.sidebar.button("Generate Report"):
    with st.spinner("Fetching and analyzing data..."):
        data = fetch_data(ticker_input, start_date_input, end_date_input)
        if not data.empty:
            # Calculate breakouts and get the dates
            breakout_days = calculate_breakouts(data, volume_threshold_input, price_threshold_input)
            
            # Create breakout information DataFrame
            breakout_data = data.loc[breakout_days].copy()
            
            # Debug information
            st.write("Data columns:", data.columns.tolist())
            st.write("Data index name:", data.index.name)
            
            # Create breakout info DataFrame ensuring we handle the date properly
            breakout_info = pd.DataFrame({
                'Date': breakout_data.index if 'Date' not in breakout_data.columns else breakout_data['Date'],
                'Close': breakout_data['Close'],
                'Volume': breakout_data['Volume'],
                'Price_Change': breakout_data['Close'].pct_change() * 100
            })
            
            # Simulate trades
            trades = simulate_trades(data, breakout_days, holding_period_input)
            metrics = calculate_metrics(trades)
            
            # Create metrics DataFrame
            df_metrics = pd.DataFrame([metrics])
            df_metrics.insert(0, 'Ticker', ticker_input)
            
            st.success(f"Analysis Complete for {ticker_input}")
            
            # Display breakout days
            st.subheader("📅 Breakout Days")
            st.dataframe(breakout_info)
            
            # Display metrics
            st.subheader("📊 Performance Metrics")
            st.dataframe(df_metrics)
            
            # Download CSVs
            metrics_csv = df_metrics.to_csv(index=False).encode('utf-8')
            breakout_csv = breakout_info.to_csv(index=False).encode('utf-8')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Metrics Report",
                    data=metrics_csv,
                    file_name=f"{ticker_input}_metrics_report.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    label="📥 Download Breakout Days",
                    data=breakout_csv,
                    file_name=f"{ticker_input}_breakout_days.csv",
                    mime="text/csv",
                )
            
            # Visualizations
            st.subheader("📈 Price and Volume Analysis")
            
            # Create figure with secondary y-axis
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Get dates for x-axis
            dates = data.index if 'Date' not in data.columns else data['Date']
            
            # Plot price on primary axis
            ax1.plot(dates, data['Close'], color='blue', label='Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Plot volume on secondary axis
            ax2 = ax1.twinx()
            ax2.bar(dates, data['Volume'], alpha=0.3, color='gray', label='Volume')
            ax2.set_ylabel('Volume', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            
            # Mark breakout points
            if not breakout_info.empty:
                ax1.scatter(breakout_info['Date'], 
                          breakout_info['Close'],
                          color='red', 
                          marker='^', 
                          s=100, 
                          label='Breakout Days')
            
            # Add title and legend
            plt.title(f'{ticker_input} Price, Volume, and Breakouts')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Show plot
            st.pyplot(fig)
            plt.close()
            
            # Distribution of Returns
            if not trades.empty:
                st.subheader("📊 Distribution of Trade Returns")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=trades['Return (%)'], bins=20, kde=True)
                plt.title('Distribution of Trade Returns (%)')
                plt.xlabel('Return (%)')
                plt.ylabel('Frequency')
                st.pyplot(fig)
                plt.close()
            
        
            # Cumulative Return Plot
            st.subheader("📈 Cumulative Return Over Time")
            data['Cumulative_Return'] = (1 + data['Close'].pct_change()).cumprod()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = data.index if 'Date' not in data.columns else data['Date']
            ax.plot(dates, data['Cumulative_Return'], label='Cumulative Return', color='blue')
            
            # Mark breakout points on cumulative return
            if not breakout_info.empty:
                breakout_returns = data.loc[breakout_days, 'Cumulative_Return']
                ax.scatter(breakout_info['Date'], 
                          breakout_returns,
                          color='red', 
                          marker='^', 
                          s=100, 
                          label='Breakout Points')
            
            if not trades.empty:
                # Convert sell dates to datetime if they aren't already
                trades['Sell Date'] = pd.to_datetime(trades['Sell Date'])
                
                # Create a mask for the sell dates
                sell_mask = data.index.isin(trades['Sell Date'])
                
                # Get the returns for those dates
                sell_dates = data.index[sell_mask]
                sell_returns = data.loc[sell_mask, 'Cumulative_Return']
                
                # Debug information
                st.write("Number of sell dates:", len(sell_dates))
                st.write("Number of sell returns:", len(sell_returns))
                
                # Only plot if we have matching dates and returns
                if len(sell_dates) == len(sell_returns) and len(sell_dates) > 0:
                    ax.scatter(sell_dates, 
                             sell_returns,
                             color='green', 
                             marker='v', 
                             s=100, 
                             label='Sell Points')
            
            ax.set_title(f'Cumulative Returns for {ticker_input}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
            
        else:
            st.error("Failed to retrieve data. Please check the ticker symbol and date range.")





