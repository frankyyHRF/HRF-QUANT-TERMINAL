import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval

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

st.title("🏛️ HRF QUANT MASTER PLATFORM V4.0")
st.caption("TradingView Market Core Engine — Model By HRF")
st.divider()

# --- NAVIGATION ---
app_mode = st.sidebar.selectbox("🚀 Choose Analysis Core Engine", ["Algorithmic Fractal Scan", "Structural Capitulation Wave"])

# Expanded TradingView Ticker Mapping Range
ticker_map = {
    'Bitcoin (BTC)': ('BTCUSD', 'BINANCE'),
    'Ethereum (ETH)': ('ETHUSD', 'BINANCE'),
    'S&P 500 (SPX)': ('SPX', 'SP'),
    'Binance Coin (BNB)': ('BNBUSD', 'BINANCE'),
    'Solana (SOL)': ('SOLUSD', 'BINANCE'),
    'Cardano (ADA)': ('ADAUSD', 'BINANCE'),
    'Dogecoin (DOGE)': ('DOGEUSD', 'BINANCE'),
    'TRON (TRX)': ('TRXUSD', 'BINANCE'),
    'Bitcoin Cash (BCH)': ('BCHUSD', 'BINANCE'),
    'Litecoin (LTC)': ('LTCUSD', 'BINANCE'),
    'Bittensor (TAO)': ('TAOUSD', 'BINANCE'),
    'Bitget Token (BGB)': ('BGBUSDT', 'BITGET'),
    'Internet Computer (ICP)': ('ICPUSD', 'BINANCE')
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
# RESTORED TRADINGVIEW DATA PIPELINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def fetch_legacy_market_data(asset_name, interval_str):
    symbol, exchange = ticker_map.get(asset_name, ('BTCUSD', 'BINANCE'))
    
    interval_dict = {
        '1M': Interval.in_monthly,
        '1w': Interval.in_weekly,
        '1d': Interval.in_daily,
        '4h': Interval.in_4_hour,
        '1h': Interval.in_1_hour,
        '15m': Interval.in_15_minute,
        '5m': Interval.in_5_minute,
        '1m': Interval.in_1_minute
    }
    tv_interval = interval_dict.get(interval_str, Interval.in_daily)
        
    try:
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=4000)
        if df is None or df.empty: return None
        
        df.index.name = 'time'
        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
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
# ENGINE HELPER FUNCTIONS FOR MATHEMATICAL INTEGRATION
# ==============================================================================
def calculate_wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_supertrend(df, period, multiplier):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.zeros(len(df))
    in_uptrend = True
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            in_uptrend = True
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            in_uptrend = False
            
        supertrend[i] = lower_band.iloc[i] if in_uptrend else upper_band.iloc[i]
    return pd.Series(supertrend, index=df.index)

def calculate_vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-10)

# ==============================================================================
# MAIN CORE: ENGINE MODULE 1 (FRACTAL SCANNER)
# ==============================================================================
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 TradingView Core Fractal Scanner")
    
    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", ["All Assets"] + list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1M', '1w', '1d', '4h', '1h', '15m', '5m', '1m'], index=2)
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
    n_fractals = st.sidebar.text_input("Maximum Displayed Fractals", value="5")
    iso_path = st.sidebar.text_input("Focus Highlight Mode Target (Number/All/Mean)", value="all")
    
    c_filter = st.sidebar.selectbox("US Cycle Macro Regime Filter", ['All Cycles', 'Election Year', 'Post-Election', 'Midterm Year', 'Pre-Election'], index=0)
    spec_years = st.sidebar.text_input("Restrict Matching to Specific Calendar Years", value="all")
    g_bars = st.sidebar.text_input("Temporal Separation Buffer Width (Bars)", value="20")
    v_boost = st.sidebar.text_input("Amplitude Volatility Boost Multiplier (%)", value="0")
    s_bands = st.sidebar.text_input("Standard Deviation Pipeline Bands (+/-)", value="1.0")

    try:
        target_bars_num = int(t_bars)
        forecast_bars_num = int(f_bars)
        num_fractals_num = int(n_fractals)
        gap_bars_num = int(g_bars)
        vol_boost_pct = float(v_boost)
        std_dev_multiplier = float(s_bands)
        c_win = int(corr_window)
    except ValueError:
        st.error("⚠️ Input Parse Warning: Confirm all dynamic inputs contain clean numeric integers.")
        st.stop()

    try:
        df_target = fetch_legacy_market_data(t_asset, i_choice)
        if df_target is None or df_target.empty:
            st.error("❌ Data download error from TradingView backend systems. Retry executing pipeline.")
            st.stop()
            
        close_target = df_target['close'].dropna()
        start_clean = start_d.strip().lower()
        end_clean = end_d.strip().lower()
        
        if start_clean == 'latest' or end_clean == 'latest':
            target_df = close_target.iloc[-target_bars_num:]
        else:
            try:
                target_df = close_target.loc[pd.to_datetime(start_clean):pd.to_datetime(end_clean)]
                if len(target_df) == 0:
                    target_df = close_target.iloc[-target_bars_num:]
            except:
                target_df = close_target.iloc[-target_bars_num:]
                
        target_bars_num = len(target_df)
        if target_bars_num == 0:
            st.error("Selected timeline coordinates do not contain market bar metrics.")
            st.stop()
            
        target_scaled = percentage_return_scale(target_df.tolist())
        
        plt.style.use('dark_background')
        if enable_corr:
            fig_frac, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        else:
            fig_frac, ax1 = plt.subplots(1, 1, figsize=(16, 8))
            ax2 = None
            
        ax1.plot(target_scaled, color='#00ffcc', linewidth=4, label=f'TARGET: {t_asset} (Baseline)', zorder=5)
        
        if f_mode == "Manual Compare":
            starts = [s.strip() for s in ov_starts.split(',') if s.strip()]
            ends = [e.strip() for e in ov_ends.split(',') if e.strip()]
            manual_paths_list = []
            isolate_clean = iso_path.strip().lower()
            
            for idx, (st_val, ed_val) in enumerate(zip(starts, ends)):
                try:
                    ov_df = close_target.loc[pd.to_datetime(st_val):pd.to_datetime(ed_val)]
                    if len(ov_df) == 0: continue
                    ov_scaled = percentage_return_scale(ov_df.tolist())
                    manual_paths_list.append(ov_scaled)
                    
                    if isolate_clean == 'all' or (isolate_clean.isdigit() and int(isolate_clean) == idx + 1):
                        ax1.plot(ov_scaled, linewidth=2, linestyle='--', alpha=0.6, label=f'Manual #{idx+1} [{st_val}]')
                except:
                    pass
            
            if manual_paths_list and isolate_clean in ['all', 'mean']:
                max_len = max(len(p) for p in manual_paths_list)
                padded_list = [np.pad(p, (0, max_len - len(p)), 'edge') if len(p) < max_len else p for p in manual_paths_list]
                manual_matrix = np.vstack(padded_list)
                m_mean = np.mean(manual_matrix, axis=0) * (1.0 + (vol_boost_pct / 100.0))
                m_std = np.std(manual_matrix, axis=0) * (1.0 + (vol_boost_pct / 100.0))
                
                if std_dev_multiplier > 0:
                    ax1.fill_between(range(len(m_mean)), m_mean - (m_std * std_dev_multiplier), m_mean + (m_std * std_dev_multiplier), color='#ffff00', alpha=0.12)
                ax1.plot(m_mean, color='#ffff00', linewidth=4, label='MANUAL COMPOSITE MEAN TRACK', zorder=6)

        else:
            assets_to_scan = list(ticker_map.keys()) if s_pool == "All Assets" else [s_pool]
            all_discovered_results = []
            
            for asset_item in assets_to_scan:
                df_scan = fetch_legacy_market_data(asset_item, i_choice)
                if df_scan is None or df_scan.empty: continue
                close_scan = df_scan['close'].dropna()
                
                if asset_item == t_asset and (start_clean == 'latest' or end_clean == 'latest'):
                    history_pool = close_scan.iloc[:-target_bars_num].tolist()
                    history_dates = close_scan.index[:-target_bars_num]
                else:
                    history_pool = close_scan.tolist()
                    history_dates = close_scan.index
                    
                max_search_index = len(history_pool) - (target_bars_num + forecast_bars_num)
                if max_search_index <= 0: continue
                
                parsed_years = [int(y.strip()) for y in spec_years.split(',') if y.strip() and y.lower() != 'all']
                
                for i in range(max_search_index):
                    hist_year = history_dates[i].year
                    if hist_year == 2026: continue
                    hist_regime = get_election_regime(hist_year)
                    
                    if c_filter != 'All Cycles' and hist_regime != c_filter: continue
                    if parsed_years and hist_year not in parsed_years: continue
                    
                    hist_pattern = history_pool[i : i + target_bars_num]
                    hist_scaled = percentage_return_scale(hist_pattern)
                    
                    mse = float(np.mean((target_scaled - hist_scaled) ** 2))
                    corr = float(np.corrcoef(target_scaled, hist_scaled)[0, 1])
                    
                    all_discovered_results.append({
                        'asset_name': asset_item, 'start_index': i, 'end_index': i + target_bars_num,
                        'mse': mse, 'correlation': corr, 'year': hist_year,
                        'raw_prices': history_pool[i : i + target_bars_num + forecast_bars_num],
                        'date_str': history_dates[i].strftime('%Y-%m-%d %H:%M')
                    })
            
            if all_discovered_results:
                match_df = pd.DataFrame(all_discovered_results).sort_values(by='mse', ascending=True)
                unique_matches, seen_clusters = [], set()
                for _, row in match_df.iterrows():
                    s, e, a = row['start_index'], row['end_index'], row['asset_name']
                    if not any(max(s, es - gap_bars_num) < min(e, ee + gap_bars_num) for es, ee, aa in seen_clusters if aa == a):
                        unique_matches.append(row)
                        seen_clusters.add((s, e, a))
                    if len(unique_matches) >= num_fractals_num: break
                
                all_scaled_paths_list = []
                for row in unique_matches:
                    raw_full = row['raw_prices']
                    scaled_full = ((np.array(raw_full) - raw_full[0]) / raw_full[0]) * 100.0 if raw_full[0] != 0 else np.zeros_like(raw_full)
                    all_scaled_paths_list.append(scaled_full)
                    
                    mean_path_array = np.mean(np.vstack(all_scaled_paths_list), axis=0) * (1.0 + (vol_boost_pct / 100.0)) if all_scaled_paths_list else None
                isolate_clean = iso_path.strip().lower()
                
                if isolate_clean != 'mean':
                    for idx, row in enumerate(unique_matches):
                        if isolate_clean.isdigit() and int(isolate_clean) != idx + 1: continue
                        raw_full = row['raw_prices']
                        scaled_full = ((np.array(raw_full) - raw_full[0]) / raw_full[0]) * 100.0
                        ax1.plot(scaled_full, linestyle='--', alpha=0.5, label=f"#{idx+1} {row['asset_name']} ({row['date_str']})")
                
                if mean_path_array is not None and isolate_clean in ['all', 'mean']:
                    if std_dev_multiplier > 0:
                        std_path = np.std(np.vstack(all_scaled_paths_list), axis=0) * (1.0 + (vol_boost_pct / 100.0))
                        ax1.fill_between(range(len(mean_path_array)), mean_path_array - (std_path * std_dev_multiplier), mean_path_array + (std_path * std_dev_multiplier), color='#ffff00', alpha=0.1)
                    ax1.plot(mean_path_array, color='#ffff00', linewidth=4, label='COMPOSITE FRACTAL MEAN (Excluding 2026)', zorder=6)
                ax1.axvline(x=target_bars_num - 1, color='#ffffff', linestyle=':', alpha=0.5)

                if enable_corr and ax2 is not None and mean_path_array is not None:
                    r_wave = calculate_rolling_correlation(target_scaled, mean_path_array, c_win)
                    ax2.plot(r_wave, color='#ffff00', linewidth=2.5, label=f"Rolling {c_win}-Bar Correlation")
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) >= 0), color='#00ff88', alpha=0.15)
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) < 0), color='#ff0055', alpha=0.15)

        ax1.axhline(y=0.0, color='#555555', linestyle='-', linewidth=1.2)
        ax1.set_title("HRF MATRIX ENGINE — UNIFIED CRYPTO CORE CANVAS", color='#ffffff', fontsize=12, fontweight='bold')
        ax1.set_ylabel("Percentage Performance Shift (%)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax1.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
        
        if enable_corr and ax2 is not None:
            ax2.axhline(0.0, color='#ffffff', linestyle='-', linewidth=0.8, alpha=0.5)
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Correlation R-Scale")
            ax2.grid(True, color='#222222')
        
        plt.tight_layout()
        st.pyplot(fig_frac)
    except Exception as e:
        st.error(f"Ecosystem halted: {e}")

# ==============================================================================
# ENGINE MODULE 2: STRUCTURAL CAPITULATION WAVE (RESTORED & EXPANDED PLAYGROUND)
# ==============================================================================
else:
    st.header("🧮 Structural Indicator Capitulation Wave Playground")
    
    playground_asset = st.sidebar.selectbox("Asset Target Matrix", list(ticker_map.keys()), index=0)
    playground_interval = st.sidebar.selectbox("Time Interval Scale", ['1M', '1w', '1d', '4h', '1h', '15m', '5m', '1m'], index=4)
    
    st.sidebar.markdown("### 📅 Precise Temporal Window")
    start_time_str = st.sidebar.text_input("Start Time (YYYY-MM-DD HH:MM)", value="2026-01-01 00:00")
    end_time_str = st.sidebar.text_input("End Time (YYYY-MM-DD HH:MM)", value="2026-07-14 00:00")
    
    # NEW FEATURE 1: Asset simulated Percentage shift factor
    st.sidebar.markdown("### ⚡ Capitulation Shift Engine")
    pct_shift_str = st.sidebar.text_input("Simulate Asset Percentage Shift (e.g. -70% or +50%)", value="0%")
    
    # NEW FEATURE 2: Candle Custom Sequence Engine
    st.sidebar.markdown("### 🕯️ Candlestick Pattern Lookup & Lookahead")
    candle_pattern_input = st.sidebar.text_input("Target Color String Sequence (e.g., ggrrgg)", value="ggrrgg").strip().lower()
    lookahead_candles = st.sidebar.number_input("Forward Lookahead Candle Window", min_value=1, max_value=200, value=10)
    
    st.sidebar.markdown("### 📈 Indicator Core Selection")
    selected_indicator = st.sidebar.selectbox("Target Indicator", [
        "Relative Strength Index (RSI)",
        "MACD Line",
        "Bollinger Bands (Middle)",
        "SuperTrend",
        "Average True Range (ATR)",
        "Volume Weighted Average Price (VWAP)",
        "Simple Moving Average (SMA)", 
        "Exponential Moving Average (EMA)", 
        "Weighted Moving Average (WMA)"
    ])
    
    selected_op = st.sidebar.selectbox("Mathematical Operational Engine", [
        "Divide Asset by Indicator (/) ", 
        "Multiply Asset by Indicator (*)", 
        "Subtract Indicator from Asset (-)", 
        "Add Asset and Indicator (+)",
        "Raise Asset to Power of Indicator (^)",
        "Square Root of Asset Matrix (sqrt)"
    ])
    
    st.sidebar.markdown("### ⚙️ Indicator Tuning Matrix")
    indicator_period = st.sidebar.number_input("Indicator Lookback Window Length", min_value=1, max_value=500, value=14)
    bb_std_dev = st.sidebar.number_input("Bollinger Band Standard Deviation Multiplier", min_value=0.1, max_value=5.0, value=2.0)
    st_mult = st.sidebar.number_input("SuperTrend Volatility Multiplier", min_value=0.1, max_value=10.0, value=3.0)

    try:
        # Load structural asset data stream
        df_play = fetch_legacy_market_data(playground_asset, playground_interval)
        if df_play is None or df_play.empty:
            st.error("❌ Target dataset empty. Verify connection or increase data bar limits.")
            st.stop()
            
        df_play = df_play.copy().sort_index()
        
        # Parse simulated percentage structural shift modifier
        try:
            clean_pct = pct_shift_str.replace('%', '').strip()
            shift_multiplier = 1.0 + (float(clean_pct) / 100.0)
        except:
            shift_multiplier = 1.0
            st.sidebar.warning("⚠️ Invalid % entry. Shift tracking defaulted to 0%.")

        # Process standard indicators across raw stream base layer
        if "Simple Moving Average" in selected_indicator or "(SMA)" in selected_indicator:
            ind_series = df_play['close'].rolling(window=int(indicator_period)).mean()
        elif "Exponential Moving Average" in selected_indicator or "(EMA)" in selected_indicator:
            ind_series = df_play['close'].ewm(span=int(indicator_period), adjust=False).mean()
        elif "Weighted Moving Average" in selected_indicator or "(WMA)" in selected_indicator:
            ind_series = calculate_wma(df_play['close'], int(indicator_period))
        elif "Relative Strength Index" in selected_indicator or "(RSI)" in selected_indicator:
            ind_series = calculate_rsi(df_play['close'], int(indicator_period))
        elif "MACD Line" in selected_indicator:
            ind_series = df_play['close'].ewm(span=12, adjust=False).mean() - df_play['close'].ewm(span=26, adjust=False).mean()
        elif "Bollinger Bands" in selected_indicator:
            ind_series = df_play['close'].rolling(window=int(indicator_period)).mean()
        elif "SuperTrend" in selected_indicator:
            ind_series = calculate_supertrend(df_play, int(indicator_period), st_mult)
        elif "Average True Range" in selected_indicator or "(ATR)" in selected_indicator:
            ind_series = calculate_atr(df_play, int(indicator_period))
        elif "Volume Weighted Average Price" in selected_indicator or "(VWAP)" in selected_indicator:
            ind_series = calculate_vwap(df_play)
            
        df_play['INDICATOR_VAL'] = ind_series
        
        # Apply precise timestamp coordinate window filtering 
        t_start = pd.to_datetime(start_time_str.strip())
        t_end = pd.to_datetime(end_time_str.strip())
        df_filtered = df_play.loc[t_start:t_end].copy()
        
        if df_filtered.empty:
            st.warning("⚠️ Filter Window Warning: The parsed date coordinates do not match any retrieved market timestamps inside the TradingView engine.")
            st.stop()
            
        # Execute mathematical shift logic directly across the target frame layer
        asset_close = df_filtered['close'] * shift_multiplier
        indicator_val = df_filtered['INDICATOR_VAL']
        
        if "Divide" in selected_op:
            math_result = asset_close / (indicator_val + 1e-10)
            op_label = f"Asset (Shifted) / Indicator Ratio"
        elif "Multiply" in selected_op:
            math_result = asset_close * indicator_val
            op_label = f"Asset (Shifted) * Indicator Matrix"
        elif "Subtract" in selected_op:
            math_result = asset_close - indicator_val
            op_label = f"Asset (Shifted) - Indicator Spread"
        elif "Add" in selected_op:
            math_result = asset_close + indicator_val
            op_label = f"Asset (Shifted) + Indicator Cumulative"
        elif "Raise" in selected_op:
            math_result = asset_close ** indicator_val
            op_label = f"Asset (Shifted) ^ Indicator Exponential Power"
        elif "Square Root" in selected_op:
            math_result = np.sqrt(asset_close)
            op_label = f"√Asset (Shifted) Matrix Scaling"

        # --- RE-INTEGRATED ORIGINAL PATH DEPENDENT STATISTICAL DRAWDOWN RESTORATION CORE ---
        running_max = -np.inf
        calculated_drawdown_series = []
        
        for price_point in asset_close:
            if price_point > running_max:
                running_max = price_point
                calculated_drawdown_series.append(0.0)
            else:
                dd_val = ((running_max - price_point) / running_max) * 100.0
                calculated_drawdown_series.append(dd_val)
                
        df_filtered['HRF_RESTORED_DRAWDOWN'] = calculated_drawdown_series

        # --- CANDLESTICK COLOR PATTERN SEARCH ENGINE ---
        df_filtered['candle_color'] = np.where(df_filtered['close'] >= df_filtered['open'], 'g', 'r')
        color_string_stream = "".join(df_filtered['candle_color'].tolist())
        
        match_indices = []
        pattern_len = len(candle_pattern_input)
        if pattern_len > 0:
            start_pos = 0
            while True:
                pos = color_string_stream.find(candle_pattern_input, start_pos)
                if pos == -1:
                    break
                match_indices.append(pos + pattern_len - 1) # Capture exact index completion index edge
                start_pos = pos + 1

        # Matplotlib visualization rendering zone using custom color scheme layout guidelines (Red/Purple focus)
        plt.style.use('dark_background')
        fig_play, (ax_main, ax_op, ax_dd) = plt.subplots(3, 1, figsize=(16, 13), sharex=True, 
                                                         gridspec_kw={'height_ratios': [2, 1.5, 1.5]})
        
        # Upper Canvas Segment: Target Price Action Baseline
        ax_main.plot(df_filtered.index, asset_close, color='#8a2be2', linewidth=2.5, label=f"{playground_asset} Close Price (Shifted x{shift_multiplier:.2f})")
        if "ATR" not in selected_indicator and "RSI" not in selected_indicator and "MACD" not in selected_indicator:
            ax_main.plot(df_filtered.index, indicator_val, color='#ff00ff', linewidth=1.5, linestyle='--', label=f"Overlayed {selected_indicator}")
            
        # Highlight match zones and lookahead regions
        for m_idx in match_indices:
            if m_idx < len(df_filtered):
                ax_main.axvline(df_filtered.index[m_idx], color='#00ffcc', linestyle='-', alpha=0.7, linewidth=1.5)
                # Compute projection frame boundary
                end_proj_idx = min(m_idx + int(lookahead_candles), len(df_filtered) - 1)
                ax_main.axvspan(df_filtered.index[m_idx], df_filtered.index[end_proj_idx], color='#00ffcc', alpha=0.1)

        ax_main.set_title(f"PRIMARY TRACKER: {playground_asset} ({playground_interval} Interval)", fontsize=11, fontweight='bold', color='#ffffff')
        ax_main.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
        ax_main.grid(True, color='#222222')
        
        # Middle Canvas Segment: Mathematical Engine Result
        ax_op.plot(df_filtered.index, math_result, color='#ff0055', linewidth=3, label=f"Calculated: {op_label}")
        ax_op.fill_between(df_filtered.index, math_result, math_result.mean(), color='#ff0055', alpha=0.08)
        ax_op.axhline(math_result.mean(), color='#ffffff', linestyle=':', alpha=0.6, label=f"Mean Baseline ({math_result.mean():.2f})")
        ax_op.set_title(f"QUANT ENGINE OUTPUT: {op_label}", fontsize=11, fontweight='bold', color='#ffffff')
        ax_op.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
        ax_op.grid(True, color='#222222')
        
        # Lower Canvas Segment: Reintegrated Restored Conditional High Reset Drawdown Framework 
        ax_dd.plot(df_filtered.index, df_filtered['HRF_RESTORED_DRAWDOWN'], color='#a100ff', linewidth=2.5, label="HRF Path-Dependent Drawdown Engine Tracker")
        ax_dd.fill_between(df_filtered.index, df_filtered['HRF_RESTORED_DRAWDOWN'], 0, color='#a100ff', alpha=0.15)
        ax_dd.invert_yaxis()  # Standard risk matrix layout rendering format
        ax_dd.set_ylabel("Drawdown Percentage Amplitude Scale (%)")
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'-{y:.1f}%'))
        ax_dd.set_title("⚠️ SYSTEM CORE RISK PROFILE LAYER (RESTORED RUNNING FRAMEWORK)", fontsize=11, fontweight='bold', color='#ffffff')
        ax_dd.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
        ax_dd.grid(True, color='#222222')
        
        plt.tight_layout(pad=3.0)
        st.pyplot(fig_play)
        
        # Performance matrix monitoring deck
        st.subheader("📊 Quant Wave Performance Matrix")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Formula Coordinate", f"{math_result.iloc[-1]:,.4f}")
        col2.metric("Matrix Peak High", f"{math_result.max():,.4f}")
        col3.metric("Max Calculated Structural Drawdown", f"-{df_filtered['HRF_RESTORED_DRAWDOWN'].max():.2f}%")
        col4.metric("Identified Sequence Clusters Found", f"{len(match_indices)} matches")
        
    except Exception as ex:
        st.error(f"Playground Compilation Fault: {ex}")

st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
