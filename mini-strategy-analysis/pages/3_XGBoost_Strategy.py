import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="XGBoost Trading Strategy", layout="wide")
st.title("📈 XGBoost Trading Strategy with Streamlit")

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Strategy Parameters")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
start_date_input = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date_input = st.sidebar.date_input("End Date", pd.to_datetime("today"))
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)
volume_threshold_input = st.sidebar.number_input(
    "Volume Breakout Threshold (%)", min_value=100.0, value=150.0, step=10.0
)
price_threshold_input = st.sidebar.number_input(
    "Price Change Threshold (%)", min_value=0.0, value=2.0, step=0.1
)
holding_period_input = st.sidebar.number_input(
    "Holding Period (Days)", min_value=1, value=10, step=1
)
run_button = st.sidebar.button("Run XGBoost Strategy")

# --------------------------------------------------
# 1. Fetch Data
# --------------------------------------------------
@st.cache_data
def fetch_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance without auto-adjusting."""
    if start >= end:
        st.error("❌ Start date must be before end date")
        return pd.DataFrame()
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=False)
        
        if df.empty:
            st.error(f"🚫 No data found for ticker {ticker}.")
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        
        if 'Adj Close' in df.columns and df['Adj Close'].isnull().all():
            st.warning(f"⚠️ 'Adj Close' is all NaN. Using 'Close' instead.")
            df['Adj Close'] = df['Close']
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"🚫 Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame()
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"🚫 Error fetching data: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------
# 2. Preprocess Data
# --------------------------------------------------
def preprocess_data(df, volume_threshold=150.0, price_threshold=2.0):
    """Preprocess data to detect breakouts."""
    if df.empty or len(df) < 30:
        st.error("❌ Insufficient data points. Need at least 30 trading days.")
        return pd.DataFrame()
    
    df = df.copy()
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    
    # Price & Volume changes
    df['Price_Change'] = df['Adj_Close'].pct_change() * 100
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # Extra momentum features (optional)
    df['Price_MA5'] = df['Adj_Close'].rolling(window=5).mean()
    df['Price_MA20'] = df['Adj_Close'].rolling(window=20).mean()
    df['Price_Momentum'] = (df['Price_MA5'] / df['Price_MA20'] - 1) * 100
    
    # Conditions for breakout
    volume_condition = df['Volume'] > (volume_threshold / 100.0) * df['Volume_MA20']
    price_condition = df['Price_Change'] >= price_threshold
    momentum_condition = df['Price_Momentum'] > 0
    
    df['Breakout'] = (volume_condition & price_condition & momentum_condition).astype(int)
    
    df.dropna(inplace=True)
    
    breakout_count = df['Breakout'].sum()
    st.write(f"Breakout signals found: {breakout_count}")
    if breakout_count == 0:
        st.warning("⚠️ No breakout signals found with current parameters.")
    
    return df

# --------------------------------------------------
# 3. Train XGBoost
# --------------------------------------------------
def train_xgboost_classifier(X_train, y_train):
    """Train XGBoost with scaling + class balancing."""
    if y_train.nunique() < 2:
        st.error("❌ Training data has only one class.")
        return None, None
    
    if len(X_train) < 100:
        st.warning("⚠️ Limited training data. Results may be unreliable.")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Class imbalance weighting
    if (y_train == 1).sum() > 0:
        pos_scale = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        pos_scale = 1.0
    
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        scale_pos_weight=pos_scale,
        eval_metric='logloss'
    )
    
    try:
        model.fit(X_train_scaled, y_train)
        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

# --------------------------------------------------
# 4. Evaluate Model
# --------------------------------------------------
def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Evaluate model performance metrics."""
    st.subheader(f"📊 Evaluation Metrics for {model_name}")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1-Score:** {f1:.4f}")
    st.write(f"**ROC-AUC:** {roc_auc:.4f}")
    
    # Classification report
    targets = ['Non-Breakout', 'Breakout']
    report_dict = classification_report(y_true, y_pred, target_names=targets, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)
    
    # Confusion matrix & ROC only if both classes are present
    if len(np.unique(y_true)) == 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=targets, yticklabels=targets)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f'{model_name} AUC={roc_auc:.2f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.title(f'{model_name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot(fig)
        plt.close()
    
    return roc_auc

# --------------------------------------------------
# 5. Simulate Trades
# --------------------------------------------------
def simulate_trades(df, breakout_signals, holding_period):
    """
    For each breakout signal, buy on that day, 
    sell after `holding_period` days (or the last day if data ends).
    """
    trades = []
    df = df.copy().reset_index(drop=True)  # ensure 0..N-1 indexing
    signal_sum = breakout_signals.sum()
    
    st.write(f"Number of breakout signals: {signal_sum}")
    if signal_sum == 0:
        st.warning("⚠️ No breakout signals in the test set.")
        return pd.DataFrame()
    
    # Indices where the model predicted a breakout
    breakout_indices = breakout_signals[breakout_signals == 1].index
    
    # For each breakout index, buy & sell
    for idx in breakout_indices:
        if idx < 0 or idx >= len(df):
            continue
        
        buy_date = df.loc[idx, 'Date']
        buy_price = df.loc[idx, 'Adj_Close']
        
        sell_idx = min(idx + holding_period, len(df) - 1)
        sell_date = df.loc[sell_idx, 'Date']
        sell_price = df.loc[sell_idx, 'Adj_Close']
        
        # Even if partial hold, we sell on the last day
        return_pct = ((sell_price - buy_price) / buy_price) * 100
        
        trades.append({
            'Buy Date': buy_date,
            'Buy Price': round(buy_price, 2),
            'Sell Date': sell_date,
            'Sell Price': round(sell_price, 2),
            'Return (%)': round(return_pct, 2)
        })
    
    trades_df = pd.DataFrame(trades)
    st.write(f"Number of trades generated: {len(trades_df)}")
    if not trades_df.empty:
        st.write("Sample trades:")
        st.dataframe(trades_df.head())
    
    return trades_df

# --------------------------------------------------
# 6. Calculate Metrics
# --------------------------------------------------
def calculate_metrics(trades_df):
    """Compute final performance stats from trades."""
    if trades_df.empty:
        return {
            'Total Return (%)': 0,
            'Avg Return/Trade (%)': 0,
            'Win Rate (%)': 0,
            'Max Drawdown (%)': 0,
            'Sharpe Ratio': np.nan,
            'Number of Trades': 0
        }
    
    total_return = trades_df['Return (%)'].sum()
    avg_return = trades_df['Return (%)'].mean()
    win_rate = (trades_df['Return (%)'] > 0).mean() * 100
    max_drawdown = trades_df['Return (%)'].min()  # min is negative
    std_dev = trades_df['Return (%)'].std()
    num_trades = len(trades_df)
    sharpe_ratio = np.nan if std_dev == 0 else (avg_return / std_dev)
    
    return {
        'Total Return (%)': round(total_return, 2),
        'Avg Return/Trade (%)': round(avg_return, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 4),
        'Number of Trades': num_trades
    }

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if run_button:
    st.write("### Running XGBoost Strategy...")
    
    # 1. Fetch Data
    df_fetched = fetch_data(ticker_input, start_date_input, end_date_input)
    if df_fetched.empty:
        st.stop()  # Nothing to do
    
    st.subheader("🔍 Fetched Data Preview")
    st.write(f"**Shape:** {df_fetched.shape}")
    st.dataframe(df_fetched.head())
    
    # 2. Preprocess
    df_processed = preprocess_data(
        df_fetched,
        volume_threshold=volume_threshold_input,
        price_threshold=price_threshold_input
    )
    if df_processed.empty:
        st.stop()  # No data after preprocessing
    
    # 3. Train/Test Split
    feature_cols = ['Volume', 'Volume_MA20', 'Price_Change']
    X = df_processed[feature_cols]
    y = df_processed['Breakout']
    
    # Time-series style split
    test_frac = test_size / 100.0
    split_idx = int(len(df_processed) * (1 - test_frac))
    
    train_data = df_processed.iloc[:split_idx]
    test_data = df_processed.iloc[split_idx:]
    
    X_train = train_data[feature_cols]
    y_train = train_data['Breakout']
    X_test = test_data[feature_cols]
    y_test = test_data['Breakout']
    
    st.write("Number of breakouts in entire dataset:", df_processed['Breakout'].sum())
    st.write(f"Train set breakouts: {y_train.sum()}, Test set breakouts: {y_test.sum()}")
    
    if X_train.empty or X_test.empty:
        st.error("❌ Training or testing set is empty.")
        st.stop()
    
    # 4. Train Model
    model, scaler = train_xgboost_classifier(X_train, y_train)
    if model is None:
        st.stop()
    st.success("✅ XGBoost Classifier trained successfully.")
    
    # 5. Predict on Test
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    st.write("Number of predicted breakouts in test set:", sum(y_pred))
    
    # 6. Evaluate Model
    roc_auc = evaluate_model(y_test, y_pred, y_prob, "XGBoost Classifier")
    
    # 7. Simulate Trades
    # Reset test_data index so it matches the logic in simulate_trades
    test_data_reset = test_data.reset_index(drop=True)
    signals = pd.Series(y_pred, index=test_data_reset.index)  # same 0..N-1 index
    
    trades = simulate_trades(test_data_reset, signals, holding_period_input)
    
    # 8. Calculate Performance
    perf = calculate_metrics(trades)
    perf_df = pd.DataFrame([perf])
    perf_df.insert(0, 'Model', 'XGBoost Classifier')
    
    st.subheader("📈 XGBoost Strategy Performance Metrics")
    st.dataframe(perf_df)
    
    # 9. Visualizations if trades exist
    if not trades.empty:
        # Plot Cumulative Return
        st.subheader("📊 Strategy Visualizations")
        
        # Add a 'Cumulative_Return' column based on 'Price_Change'
        test_data_reset['Cumulative_Return'] = (1 + test_data_reset['Price_Change'] / 100).cumprod()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test_data_reset['Date'], test_data_reset['Cumulative_Return'], label='Cumulative Return', color='blue')
        
        # Mark Buy & Sell
        buy_dates = trades['Buy Date']
        buy_returns = test_data_reset[test_data_reset['Date'].isin(buy_dates)]['Cumulative_Return']
        ax.scatter(buy_dates, buy_returns, color='green', marker='^', s=100, label='Buy Signal')
        
        sell_dates = trades['Sell Date']
        sell_returns = test_data_reset[test_data_reset['Date'].isin(sell_dates)]['Cumulative_Return']
        ax.scatter(sell_dates, sell_returns, color='red', marker='v', s=100, label='Sell Signal')
        
        plt.title(f"Cumulative Returns - {ticker_input}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Trade Return Distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(trades['Return (%)'], bins=30, kde=True, ax=ax)
        plt.title("Distribution of Trade Returns")
        plt.xlabel("Return (%)")
        plt.ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)
        
        # Feature Importance
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=imp_df, x='Feature', y='Importance', ax=ax)
        plt.title("Feature Importance")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
        
        # Monthly Stats
        st.subheader("📊 Additional Statistics")
        trades['Buy Date'] = pd.to_datetime(trades['Buy Date'])
        trades['Month'] = trades['Buy Date'].dt.strftime('%Y-%m')
        monthly_stats = trades.groupby('Month')['Return (%)'].agg(['count','mean','std','min','max']).round(2)
        st.write("Monthly Performance:")
        st.dataframe(monthly_stats)
        
    else:
        st.warning("⚠️ No trades generated. Consider adjusting parameters (volume/price thresholds, date range).")
