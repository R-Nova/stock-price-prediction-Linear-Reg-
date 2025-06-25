import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from io import BytesIO

st.set_page_config(page_title="Multi-Asset Predictor", page_icon="??", layout="wide")

st.sidebar.title("?? Asset Options")
symbols_input = st.sidebar.text_input(
    "Enter symbols (sep by comma), e.g. AAPL, GOOGL, BTC-USD", "AAPL,BTC-USD"
)
days = st.sidebar.slider("Use how many past days?", 60, 365, 180)

st.title("?? Multi-Asset Actual vs Predicted Dashboard")
st.write("---")

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
if not symbols:
    st.info("Please enter at least one symbol.")
    st.stop()

results = {}
fig, ax = plt.subplots(figsize=(14, 7))
colors = plt.cm.tab10.colors

for i, sym in enumerate(symbols):
    try:
        df = yf.download(sym, period=f"{days}d")
        df = df[['Close']].dropna()
        df['Prev'] = df['Close'].shift(1)
        df.dropna(inplace=True)

        X = df[['Prev']].values
        y = df['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)

        actual = df['Close'].values
        latest = float(actual[-1])
        predicted_today = float(preds[-1])
        tomorrow = model.predict(np.array([[latest]]))[0]

        results[sym] = {
            'df': df,
            'preds': preds,
            'latest': latest,
            'predicted_today': predicted_today,
            'tomorrow': tomorrow
        }

        ax.plot(actual, label=f"{sym} Actual", color=colors[i % len(colors)], linestyle='-')
        ax.plot(preds, label=f"{sym} Predicted", color=colors[i % len(colors)], linestyle='--')
    except Exception as e:
        st.error(f"Error with {sym}: {e}")

if results:
    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

    metrics_cols = st.columns(len(results))
    for col, sym in zip(metrics_cols, results):
        r = results[sym]
        col.metric(f"?? {sym} Latest", f"${r['latest']:.2f}")
        col.metric(f"?? {sym} Predicted Today", f"${r['predicted_today']:.2f}")
        col.metric(f"?? {sym} Tomorrow", f"${r['tomorrow']:.2f}")

    # Download combined CSV
    out = []
    for sym, r in results.items():
        df = r['df'].copy()
        df['Predicted'] = r['preds']
        df['Symbol'] = sym
        out.append(df[['Symbol', 'Close', 'Predicted']])
    combined = pd.concat(out).reset_index().rename(columns={'index': 'Date'})
    csv = combined.to_csv(index=False).encode()
    st.download_button(
        "?? Download All Predictions",
        csv,
        file_name="multi_asset_predictions.csv",
        mime="text/csv"
    )