# utils.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data(ticker, start, end):
    """
    Fetch historical stock data using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        if data.empty:
            print(f"No data found for {ticker}.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_breakouts(data, volume_threshold, price_threshold):
    """
    Identify breakout days based on volume and price thresholds.
    """
    data = data.copy()
    data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()
    data['Price_Change'] = data['Close'].pct_change() * 100
    breakout_days = data[
        (data['Volume'] > (volume_threshold / 100) * data['Avg_Volume']) &
        (data['Price_Change'] >= price_threshold)
    ].index
    return breakout_days

def simulate_trades(data, breakout_days, holding_period):
    """
    Simulate buying on breakout days and selling after the holding period.
    Calculate returns and other metrics.
    """
    results = []
    for buy_date in breakout_days:
        try:
            buy_price = data.loc[buy_date]['Close']
            # Calculate sell date
            sell_index = data.index.get_loc(buy_date) + holding_period
            if sell_index >= len(data):
                continue  # Holding period exceeds available data
            sell_date = data.index[sell_index]
            sell_price = data.loc[sell_date]['Close']
            return_pct = ((sell_price - buy_price) / buy_price) * 100
            results.append({
                'Buy Date': buy_date.date(),
                'Buy Price': round(buy_price, 2),
                'Sell Date': sell_date.date(),
                'Sell Price': round(sell_price, 2),
                'Return (%)': round(return_pct, 2)
            })
        except (IndexError, KeyError):
            continue
    return pd.DataFrame(results)

def calculate_metrics(trades):
    """
    Calculate key performance metrics from simulated trades.
    """
    if trades.empty:
        return {
            'Breakout Days': 0,
            'Total Return (%)': 0,
            'Avg Return/Trade (%)': 0,
            'Win Rate (%)': 0,
            'Max Drawdown (%)': 0,
            'Sharpe Ratio': 0
        }
    
    total_return = trades['Return (%)'].sum()
    avg_return = trades['Return (%)'].mean()
    win_rate = (trades['Return (%)'] > 0).mean() * 100
    max_drawdown = trades['Return (%)'].min()
    
    # Sharpe Ratio Calculation (Assuming risk-free rate = 0)
    sharpe_ratio = trades['Return (%)'].mean() / trades['Return (%)'].std() if trades['Return (%)'].std() != 0 else 0
    
    return {
        'Breakout Days': len(trades),
        'Total Return (%)': round(total_return, 2),
        'Avg Return/Trade (%)': round(avg_return, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2)
    }

def get_market_cap(ticker):
    """
    Fetch the market capitalization of a ticker.
    """
    stock = yf.Ticker(ticker)
    try:
        mc = stock.info['marketCap']
        if mc >= 10**10:
            return 'Large'
        elif 10**9 <= mc < 10**10:
            return 'Mid'
        else:
            return 'Small'
    except:
        return 'Unknown'
