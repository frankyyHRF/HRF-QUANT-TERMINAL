import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from tvDatafeed import TvDatafeed, Interval

# --- UNIFIED SYSTEM INITIALIZATION ---
st.set_page_config(page_title="HRF QUANT PLATFORM", layout="wide")

st.markdown("""
    <style>
    .reportview-container { background: #0d0d11; }
    .sidebar .sidebar-content { background: #13131a; }
    h1, h2, h3, p, label { color: #ffffff !important; }
    div.stButton > button:first-child {
        background-color: #8a2be2; color: white; border-radius: 8px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ HRF QUANT MASTER PLATFORM V4.7")
st.caption("Google Drive Stream Integration — Model By HRF")
st.divider()

# --- GOOGLE DRIVE DATA INTERCEPTOR FEATURE ---
st.sidebar.markdown("### 🔑 Link Remote Dataset")
drive_url_input = st.sidebar.text_input("Paste Shared Google Drive CSV Link:", value="")

@st.cache_data(show_spinner="🔄 Fetching direct matrix from Google Drive...")
def parse_google_drive_stream(url_string):
    if not url_string:
        return None
    try:
        # Extract file ID from standard sharing link structure
        if "id=" in url_string:
            file_id = url_string.split("id=")[1].split("&")[0]
        elif "file/d/" in url_string:
            file_id = url_string.split("file/d/")[1].split("/")[0]
        else:
            file_id = url_string
            
        direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        df = pd.read_csv(direct_download_url)
        
        # Clean columns to fit your exact model logic downstream
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            
        close_col = next((c for c in df.columns if 'close' in c or 'price' in c), None)
        if close_col:
            df = df[[close_col]].rename(columns={close_col: 'close'})
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']
            df['volume'] = 0
            return df
    except Exception as e:
        st.sidebar.error(f"Stream error: {e}")
    return None

# Convert the live link to data if it exists
remote_drive_data = parse_google_drive_stream(drive_url_input)
if remote_drive_data is not None:
    st.sidebar.success("✅ Remote Stream Database Online")

# --- NAVIGATION ---
app_mode = st.sidebar.selectbox("🚀 Choose Analysis Core Engine", ["Algorithmic Fractal Scan", "Structural Capitulation Wave"])

ticker_map = {
    'Bitcoin (BTC)': ('BTCUSD', 'BINANCE'),
    'Ethereum (ETH)': ('ETHUSD', 'BINANCE'),
    'S&P 500 (SPX)': ('SPX', 'SP'),
    'Nasdaq 100 (QQQ)': ('QQQ', 'NASDAQ')
}

def percentage_return_scale(series):
    arr = np.array(series, dtype=float).flatten()
    if len(arr) == 0 or arr[0] == 0:
        return np.zeros_like(arr)
    return ((arr - arr[0]) / arr[0]) * 100.0

# --- HYBRID BACKEND OVERRIDE INTERCEPTOR ---
def fetch_legacy_market_data(asset_name, interval_str):
    # If a Google Drive link is pasted, it completely overrides the data source instantly
    if remote_drive_data is not None:
        return remote_drive_data
        
    # Standard TradingView Fallback Protocol (Kept exactly identical)
    symbol, exchange = ticker_map.get(asset_name, ('BTCUSD', 'BINANCE'))
    interval_dict = {'1M': Interval.in_monthly, '1w': Interval.in_weekly, '1d': Interval.in_daily}
    tv_interval = interval_dict.get(interval_str, Interval.in_daily)
    try:
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=2000)
        df.index.name = 'time'
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return None

# --- ENGINE MODULE 1 (FRACTAL SCANNER) ---
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 System Core Fractal Scanner")
    
    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", ["All Assets"] + list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1M', '1w', '1d'], index=2)
    
    t_bars = st.sidebar.text_input("Scanning Sequence Width (Bars)", value="30")
    f_bars = st.sidebar.text_input("Forward Projection Target (Bars)", value="15")
    
    try:
        target_bars_num = int(t_bars)
        forecast_bars_num = int(f_bars)
    except:
        st.stop()

    df_target = fetch_legacy_market_data(t_asset, i_choice)
    if df_target is not None and not df_target.empty:
        close_target = df_target['close'].dropna()
        target_df = close_target.iloc[-target_bars_num:]
        target_scaled = percentage_return_scale(target_df.tolist())
        
        plt.style.use('dark_background')
        fig_frac, ax1 = plt.subplots(figsize=(16, 7))
        ax1.plot(target_scaled, color='#00ffcc', linewidth=4, label=f'TARGET: {t_asset}')
        ax1.set_title("HRF MATRIX ENGINE — STRUCTURAL DATA RECONSTRUCTION", color='#ffffff', fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax1.legend()
        st.pyplot(fig_frac)

# --- ENGINE MODULE 2 (STRUCTURAL CAPITULATION WAVE) ---
else:
    st.header("⏱️ Path-Dependent Peak-To-Trough Reset Wave")
    vol_input = st.sidebar.text_input("Volatility Reset Threshold (%)", value="-40.0")
    
    df_daily = fetch_legacy_market_data("Bitcoin (BTC)", "1d")
    if df_daily is not None:
        df_daily['return'] = ((df_daily['close'] - df_daily['open']) / df_daily['open']) * 100.0
        thresh = float(vol_input)
        days_since_tracker = []
        current_accumulator = 0
        
        running_peak = df_daily['high'].iloc[0]
        for idx in range(len(df_daily)):
            current_high = df_daily['high'].iloc[idx]
            current_low = df_daily['low'].iloc[idx]
            if current_high > running_peak: running_peak = current_high
            drawdown_pct = ((current_low - running_peak) / running_peak) * 100.0
            if drawdown_pct <= thresh:
                days_since_tracker.append(0)
                current_accumulator = 0
                running_peak = current_high
            else:
                current_accumulator += 1
                days_since_tracker.append(current_accumulator)
        
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_daily.index, days_since_tracker, color='#00ffcc')
        ax.fill_between(df_daily.index, 0, days_since_tracker, color='#00ffcc', alpha=0.06)
        st.pyplot(fig)
        st.metric("Current Days Stretched Under Matrix Threshold", f"{days_since_tracker[-1]} Days")

st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
