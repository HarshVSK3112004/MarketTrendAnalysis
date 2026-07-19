import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# ==========================================
# Market Trend Analysis
# Dataset Downloader
# ==========================================

START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Top NIFTY Stocks
STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "LT.NS",
    "ITC.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "BHARTIARTL.NS",
    "ULTRACEMCO.NS",
    "TITAN.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "ADANIPORTS.NS",
    "NESTLEIND.NS",
    "HINDUNILVR.NS",
    "JSWSTEEL.NS",
    "TECHM.NS",
    "BAJAJFINSV.NS",
    "INDUSINDBK.NS"
]


def download_stock(symbol):
    """
    Download one stock.
    """

    print(f"Downloading {symbol}")

    try:

        df = yf.download(
            symbol,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"No data found for {symbol}")
            return None

        # Fix MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Required columns only
        df = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ]

        df.reset_index(inplace=True)

        df["Stock"] = symbol

        print(f"Downloaded {len(df)} rows")

        return df

    except Exception as e:

        print(e)

        return None


def create_dataset():

    all_data = []

    for stock in STOCKS:

        df = download_stock(stock)

        if df is not None:

            all_data.append(df)

    if len(all_data) == 0:

        print("No data downloaded.")

        return

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df.dropna(inplace=True)

    final_df.sort_values(
        by=[
            "Stock",
            "Date"
        ],
        inplace=True
    )

    os.makedirs(
        "dataset",
        exist_ok=True
    )

    final_df.to_csv(
        "dataset/market_data.csv",
        index=False
    )

    print("\n================================")

    print("Dataset Created Successfully")

    print("Rows :", len(final_df))

    print("Columns :", len(final_df.columns))

    print(final_df.columns.tolist())

    print(final_df.head())

    print("================================")


if __name__ == "__main__":

    create_dataset()