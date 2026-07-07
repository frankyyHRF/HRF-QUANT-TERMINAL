import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# --- UNIFIED SYSTEM INITIALIZATION ---
st.set_page_config(page_title="HRF QUANT PLATFORM", layout="wide")

# Inject deep dark-mode styling variables for mobile UI clarity
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

st.title("🏛️ HRF QUANT MASTER PLATFORM V4.1")
st.caption("Unblocked CoinGecko Macro Core Engine (Max Historical Archive) — Model By HRF")
st.divider()

# --- NAVIGATION ---
app_mode = st.sidebar.selectbox("🚀 Choose Analysis Core Engine", ["Algorithmic Fractal Scan", "Structural Capitulation Wave"])

# CoinGecko API unique asset identifiers for max historical pull
ticker_map = {
    'Bitcoin (BTC)': 'bitcoin',
    'Ethereum (ETH)': 'ethereum',
    'Ripple (XRP)': 'ripple',
    'Binance Coin (BNB)': 'binancecoin',
    'Solana (SOL)': 'solana',
    'Cardano (ADA)': 'cardano',
    'Dogecoin (DOGE)': 'dogecoin'
}

def percentage_return_scale(series):
    arr = np.array(series, dtype=float).flatten()
    if len(arr) == 0 or arr[0] == 0:
        return np.zeros_like(arr)
    return ((arr - arr[0]) / arr[0]) * 100.0

def get_election_regime(year):
    if year % 4 == 0: return 'Election Year'
    elif year % 4 == 1: return 'Post-Election'
    elif year % 4 == 2: return 'Midterm Year'
    else: return 'Pre-Election'

# ==============================================================================
# UNBLOCKED MAX DATA GEOMETRY ENGINE (COINGECKO ARCHIVE)
# ==============================================================================
@st.cache_data(show_spinner=False)
def fetch_max_historical_data(asset_key, interval_choice):
    """
    Bypasses Cloudflare hosting restrictions natively by querying CoinGecko's open ledger.
    Delivers maximum possible lifetime data array strings.
    """
    cg_id = ticker_map[asset_key]
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "max",
        "interval": "daily"
    }
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, params=params, headers=headers, timeout=15)
        if res.status_code != 200:
            return None
            
        data = res.json()
        if 'prices' not in data or not data['prices']:
            return None
            
        # Parse data frames
        df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
        df_vol = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        df = pd.merge(df_price, df_vol, on='timestamp')
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('time', inplace=True)
        
        # Populate operational matrix layers smoothly
        df['open'] = df['close'].shift(1).fillna(df['close'])
        df['high'] = df[['open', 'close']].max(axis=1)
        df['low'] = df[['open', 'close']].min(axis=1)
        
        # Resample logic depending on macro views
        if interval_choice in ['1w', 'Sequence Time Frame Interval']:
            resample_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df = df.resample('W').agg(resample_rules).dropna()
        elif interval_choice == '1M':
            resample_rules = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df = df.resample('M').agg(resample_rules).dropna()
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return None

# ==============================================================================
# STRUCTURAL CORRELATION ENGINE MODULE
# ==============================================================================
def calculate_rolling_correlation(series_a, series_b, lookback_window):
    arr_a = np.array(series_a, dtype=float).flatten()
    arr_b = np.array(series_b, dtype=float).flatten()
    min_len = min(len(arr_a), len(arr_b))
    if min_len == 0: return []
    df = pd.DataFrame({'Target_Curve': arr_a[:min_len], 'Match_Curve': arr_b[:min_len]})
    df['Ret_A'] = df['Target_Curve'].pct_change()
    df['Ret_C'] = df['Match_Curve'].pct_change()
    return df['Ret_A'].rolling(window=int(lookback_window)).corr(df['Ret_C']).fillna(0.0).tolist()

# ==============================================================================
# MAIN CORE: ENGINE MODULE 1 (FRACTAL SCANNER)
# ==============================================================================
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 Cloud-Unblocked Fractal Core Scanner")
    
    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", ["All Assets"] + list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1M', '1w', '1d'], index=2)
    f_mode = st.sidebar.selectbox("Framework Processing Mode", ["Calculate Fractals", "Manual Compare"], index=0)
    
    st.sidebar.markdown("### 📊 Inter-Asset Correlation Layer")
    enable_corr = st.sidebar.checkbox("Overlay Rolling Macro Asset Correlation", value=False)
    corr_window = st.sidebar.text_input("Correlation Lookback Window (Bars)", value="10")
    
    start_d = st.sidebar.text_input("Analysis Target Window Start", value="latest")
    end_d = st.sidebar.text_input("Analysis Target Window End", value="latest")
    
    ov_starts = st.sidebar.text_input("Manual Overlay Window Starts", value="2020-03-01, 2022-11-01")
    ov_ends = st.sidebar.text_input("Manual Overlay Window Ends", value="2020-05-01, 2023-01-01")
    
    t_bars = st.sidebar.text_input("Scanning Sequence Width (Bars)", value="30")
    f_bars = st.sidebar.text_input("Forward Projection Target (Bars)", value="15")
    n_fractals = st.sidebar.text_input("Maximum Displayed Fractals
