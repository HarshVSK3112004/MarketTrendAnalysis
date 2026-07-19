# ==========================================================
# Market Trend Analysis
# Streamlit Dashboard
# Part 1
# ==========================================================

import os
import time
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# ==========================================================
# Streamlit Configuration
# ==========================================================

st.set_page_config(
    page_title="Market Trend Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Market Trend Analysis")

st.markdown("""
### Machine Learning Based Stock Trend Prediction

Predict stock trends using a Random Forest Machine Learning model,
technical indicators and live Yahoo Finance market data.
""")

# ==========================================================
# Model Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "trend_model.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_names.pkl"
)


# ==========================================================
# Check Files
# ==========================================================

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found:\n{MODEL_PATH}")
    st.stop()

if not os.path.exists(FEATURE_PATH):
    st.error(f"Feature file not found:\n{FEATURE_PATH}")
    st.stop()

# ==========================================================
# Load Model
# ==========================================================

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURE_PATH)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Settings")

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Bitcoin": "BTC-USD"
}

selected_name = st.sidebar.selectbox(
    "Select Stock",
    list(stocks.keys())
)

symbol = stocks[selected_name]

manual_symbol = st.sidebar.text_input(
    "Or Enter Yahoo Finance Symbol"
)

if manual_symbol.strip():
    symbol = manual_symbol.strip().upper()

refresh = st.sidebar.selectbox(
    "Auto Refresh",
    [
        "Off",
        "15 sec",
        "30 sec",
        "60 sec"
    ],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.success(f"Selected Symbol : {symbol}")

# ==========================================================
# Helper Functions
# ==========================================================

def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=30)
def load_stock_data(symbol):

    df = yf.download(
        symbol,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    df = clean_columns(df)

    df.reset_index(inplace=True)

    return df


@st.cache_data(ttl=10)
def get_live_price(symbol):

    ticker = yf.Ticker(symbol)

    try:

        info = ticker.fast_info

        price = info["lastPrice"]
        previous = info["previousClose"]

    except:

        df = yf.download(
            symbol,
            period="5d",
            auto_adjust=True,
            progress=False
        )

        df = clean_columns(df)

        price = float(df["Close"].iloc[-1])
        previous = float(df["Close"].iloc[-2])

    return float(price), float(previous)

# ==========================================================
# Live Market
# ==========================================================

st.subheader("📊 Live Market")

try:

    current_price, previous_close = get_live_price(symbol)

    change = current_price - previous_close

    percent = (change / previous_close) * 100

    currency = "₹"

    if not symbol.endswith(".NS"):
        currency = "$"

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"{currency}{current_price:.2f}"
    )

    c2.metric(
        "Previous Close",
        f"{currency}{previous_close:.2f}"
    )

    c3.metric(
        "Change",
        f"{change:+.2f}"
    )

    c4.metric(
        "% Change",
        f"{percent:+.2f}%"
    )

    st.caption(
        f"Updated : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )

except Exception as e:

    st.error(e)

st.divider()
# ==========================================================
# Feature Engineering
# Must match train_model.py exactly
# ==========================================================

def prepare_features(df):

    # -----------------------------
    # Moving Averages
    # -----------------------------

    df["SMA20"] = df["Close"].rolling(20).mean()

    df["SMA50"] = df["Close"].rolling(50).mean()

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # -----------------------------
    # Price Features
    # -----------------------------

    df["Price_Change"] = df["Close"] - df["Open"]

    df["Daily_Return"] = df["Close"].pct_change()

    df["Return_5D"] = df["Close"].pct_change(5)

    df["Return_10D"] = df["Close"].pct_change(10)

    # -----------------------------
    # Volume Features
    # -----------------------------

    df["Volume_Change"] = df["Volume"].pct_change()

    df["Prev_Volume"] = df["Volume"].shift(1)

    # -----------------------------
    # Previous Day Values
    # -----------------------------

    df["Prev_Open"] = df["Open"].shift(1)

    df["Prev_High"] = df["High"].shift(1)

    df["Prev_Low"] = df["Low"].shift(1)

    df["Prev_Close"] = df["Close"].shift(1)

    # -----------------------------
    # Volatility
    # -----------------------------

    df["Volatility"] = df["High"] - df["Low"]

    # -----------------------------
    # RSI
    # -----------------------------

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # MACD
    # -----------------------------

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()

    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # -----------------------------
    # Bollinger Bands
    # -----------------------------

    middle = df["Close"].rolling(20).mean()

    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = middle + (2 * std)

    df["BB_Lower"] = middle - (2 * std)

    # -----------------------------
    # ATR
    # -----------------------------

    high_low = df["High"] - df["Low"]

    high_close = (df["High"] - df["Prev_Close"]).abs()

    low_close = (df["Low"] - df["Prev_Close"]).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    # -----------------------------
    # Remove Missing Values
    # -----------------------------

    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(
            "Not enough historical data to calculate indicators."
        )

    # -----------------------------
    # Latest Record
    # -----------------------------

    latest = df.iloc[-1]

    input_data = pd.DataFrame([latest])

    # -----------------------------
    # Ensure all training features exist
    # -----------------------------

    for feature in FEATURES:

        if feature not in input_data.columns:
            input_data[feature] = 0

    # -----------------------------
    # Keep EXACT order used in training
    # -----------------------------

    input_data = input_data[FEATURES]

    return df, latest, input_data
# ==========================================================
# Prediction
# ==========================================================

if st.button("🔮 Predict Trend", use_container_width=True):

    try:

        with st.spinner("Downloading latest market data..."):

            df = load_stock_data(symbol)

            if df.empty:
                st.error("Unable to download stock data.")
                st.stop()

            df, latest, input_data = prepare_features(df)

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0]

            confidence = float(np.max(probability) * 100)

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        if prediction == 1:

            trend = "UP"

            recommendation = "BUY"

            col1.success(
                f"📈 UP Trend\n\nConfidence : {confidence:.2f}%"
            )

        else:

            trend = "DOWN"

            recommendation = "SELL"

            col1.error(
                f"📉 DOWN Trend\n\nConfidence : {confidence:.2f}%"
            )

        if confidence < 60:

            recommendation = "HOLD"

        col2.info(f"### Recommendation : {recommendation}")

        # ==================================================
        # Price Chart
        # ==================================================

        st.subheader("Candlestick Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SMA20"],
                mode="lines",
                name="SMA20"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SMA50"],
                mode="lines",
                name="SMA50"
            )
        )

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==================================================
        # RSI Chart
        # ==================================================

        st.subheader("RSI")

        rsi = go.Figure()

        rsi.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["RSI"],
                mode="lines",
                name="RSI"
            )
        )

        rsi.add_hline(
            y=70,
            line_dash="dash",
            line_color="red"
        )

        rsi.add_hline(
            y=30,
            line_dash="dash",
            line_color="green"
        )

        rsi.update_layout(height=300)

        st.plotly_chart(
            rsi,
            use_container_width=True
        )

        # ==================================================
        # Technical Indicators
        # ==================================================

        st.subheader("Latest Technical Indicators")

        indicators = pd.DataFrame({

            "Indicator": [

                "Close",

                "SMA20",

                "SMA50",

                "EMA20",

                "EMA50",

                "RSI",

                "MACD",

                "ATR"

            ],

            "Value": [

                round(float(latest["Close"]),2),

                round(float(latest["SMA20"]),2),

                round(float(latest["SMA50"]),2),

                round(float(latest["EMA20"]),2),

                round(float(latest["EMA50"]),2),

                round(float(latest["RSI"]),2),

                round(float(latest["MACD"]),2),

                round(float(latest["ATR"]),2)

            ]

        })

        st.dataframe(
            indicators,
            use_container_width=True
        )

        # ==================================================
        # Prediction History
        # ==================================================

        if "history" not in st.session_state:

            st.session_state.history = pd.DataFrame(
                columns=[
                    "Time",
                    "Stock",
                    "Trend",
                    "Confidence",
                    "Recommendation"
                ]
            )

        new_row = {

            "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

            "Stock": symbol,

            "Trend": trend,

            "Confidence": f"{confidence:.2f}%",

            "Recommendation": recommendation

        }

        st.session_state.history = pd.concat(

            [

                pd.DataFrame([new_row]),

                st.session_state.history

            ],

            ignore_index=True

        )

        st.subheader("Prediction History")

        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        csv = st.session_state.history.to_csv(index=False)

        st.download_button(

            "⬇ Download History",

            csv,

            "prediction_history.csv",

            "text/csv"

        )

        st.success("Prediction Completed Successfully.")

    except Exception as e:

        st.error(str(e))
# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
    """
---
## 📈 Market Trend Analysis Dashboard

### Features

- ✅ Live Stock Price
- ✅ Machine Learning Prediction
- ✅ Random Forest Classifier
- ✅ Technical Indicators
- ✅ Candlestick Chart
- ✅ RSI Indicator
- ✅ Prediction History
- ✅ CSV Download

---

### Disclaimer

This project is developed for educational and research purposes only.

Predictions are generated using historical market data and machine learning techniques.

Do **NOT** use this application as financial or investment advice.
"""
)

# ==========================================================
# Auto Refresh
# ==========================================================

if refresh != "Off":

    import time

    refresh_seconds = {

        "15 sec": 15,
        "30 sec": 30,
        "60 sec": 60

    }[refresh]

    time.sleep(refresh_seconds)

    st.rerun()

confidence = 0
latest = None
trend = "N/A"
recommendation = "N/A"
# ==========================================================
# PART 5
# Model Information & Analytics
# ==========================================================

st.divider()

st.subheader("🤖 Model Information")

c1, c2, c3 = st.columns(3)

c1.metric("Algorithm", "Random Forest")
c2.metric("Features", len(FEATURES))
c3.metric("Prediction", "result")


# ==========================================================
# Confidence Progress Bar
# ==========================================================

if latest is not None:

    st.subheader("🎯 Prediction Confidence")

    st.progress(float(confidence) / 100)

    st.write(
        f"Confidence Score : **{confidence:.2f}%**"
    )

if confidence >= 80:
    st.success("Very High Confidence Prediction")

elif confidence >= 60:
    st.info("Moderate Confidence Prediction")

else:
    st.warning("Low Confidence. Consider Waiting.")

# ==========================================================
# Latest Feature Values
# ==========================================================
if latest is not None:

    st.subheader("📊 Model Input Features")

    feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Value": [latest[f] for f in FEATURES]
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        height=450
    )
# ==========================================================
# Feature Importance
# ==========================================================

importance_file = "reports/feature_importance.csv"

if os.path.exists(importance_file):

    st.subheader("⭐ Feature Importance")

    importance = pd.read_csv(importance_file)

    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig = go.Figure()

    fig.add_trace(

        go.Bar(

            x=importance["Importance"],

            y=importance["Feature"],

            orientation="h"

        )

    )

    fig.update_layout(

        height=600,

        yaxis=dict(autorange="reversed"),

        title="Top 15 Important Features"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# Model Report
# ==========================================================

report_file = "reports/model_report.txt"

if os.path.exists(report_file):

    st.subheader("📄 Model Report")

    with open(report_file, "r") as f:

        report = f.read()

    st.text(report)

    st.download_button(

        "⬇ Download Model Report",

        report,

        file_name="model_report.txt",

        mime="text/plain"

    )

# ==========================================================
# Feature Importance Image
# ==========================================================

image_path = "reports/feature_importance.png"

if os.path.exists(image_path):

    st.subheader("📈 Feature Importance Chart")

    st.image(
        image_path,
        use_container_width=True
    ) 
# ==========================================================
# PART 6
# Footer & Auto Refresh
# ==========================================================

st.divider()

st.markdown(
    """
## 📈 Market Trend Analysis Dashboard

### Features
- ✅ Live Stock Price
- ✅ Machine Learning Prediction
- ✅ Random Forest Classifier
- ✅ Technical Indicators
- ✅ Candlestick Chart
- ✅ RSI Indicator
- ✅ Prediction History
- ✅ Download Reports
- ✅ Feature Importance
- ✅ Live Yahoo Finance Data

---
### Model Information

**Algorithm:** Random Forest Classifier

**Technical Indicators Used**
- SMA20
- SMA50
- EMA20
- EMA50
- RSI
- MACD
- MACD Signal
- Bollinger Bands
- ATR
- Daily Returns
- Volatility

---
⚠ **Disclaimer**

This project is developed for educational and research purposes only.

Predictions are generated using historical market data and a Machine Learning model.

Do **NOT** use these predictions as financial or investment advice.

Always perform your own analysis before investing.
"""
)

# ==========================================================
# Sidebar Information
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.subheader("About")

st.sidebar.info(
    """
Market Trend Analysis

Version : 1.0

Machine Learning:
Random Forest

Data Source:
Yahoo Finance

Framework:
Streamlit
"""
)

# ==========================================================
# Auto Refresh
# ==========================================================

if refresh != "Off":

    import time

    refresh_seconds = {
        "15 sec": 15,
        "30 sec": 30,
        "60 sec": 60
    }[refresh]

    st.sidebar.success(
        f"Auto Refresh Every {refresh_seconds} Seconds"
    )

    time.sleep(refresh_seconds)

    st.rerun()               