import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tvDatafeed import TvDatafeed, Interval

# ==============================================================================
# --- UNIFIED SYSTEM INITIALIZATION ---
# ==============================================================================
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

st.title("🏛️ HRF QUANT MASTER PLATFORM V5.0")
st.caption("The Complete Unified Multi-Engine Ecosystem — Model By HRF")
st.divider()

# --- CENTRAL HUB NAVIGATION ---
app_mode = st.sidebar.selectbox(
    "🚀 Choose Analysis Core Engine", 
    ["Algorithmic Fractal Scan", "Structural Capitulation Wave", "Statistical Framework Core"]
)

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
# TRADINGVIEW DATA PIPELINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def fetch_legacy_market_data(asset_name, interval_str):
    symbol, exchange = ticker_map.get(asset_name, ('BTCUSD', 'BINANCE'))
    
    interval_dict = {
        '1month': Interval.in_monthly,
        '1 week': Interval.in_weekly,
        '1day': Interval.in_daily,
        '4h': Interval.in_4_hour,
        '1h': Interval.in_1_hour,
        '15m': Interval.in_15_minute,
        '5m': Interval.in_5_minute,
        '1m': Interval.in_1_minute
    }
    tv_interval = interval_dict.get(interval_str, Interval.in_daily)
    
    # Scale depth request specifically to capture 2023-Today historical window
    if interval_str == '5m':
        bars_limit = 450000
    elif interval_str == '15m':
        bars_limit = 150000
    elif interval_str == '1h':
        bars_limit = 35000
    else:
        bars_limit = 4000
        
    try:
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=bars_limit)
        if df is None or df.empty: return None
        
        df.index.name = 'time'
        df.index = pd.to_datetime(df.index)
        df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return None

# ==============================================================================
# NATIVE VECTORIZED MATH AND INDICATORS CALCULATOR
# ==============================================================================
def calculate_custom_indicators(df, cfg):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    
    # Moving Averages
    df['SMA'] = close.rolling(window=cfg['sma_p']).mean()
    df['EMA'] = close.ewm(span=cfg['ema_p'], adjust=False).mean()
    df['WMA'] = close.rolling(window=cfg['wma_p']).apply(
        lambda x: np.dot(x, np.arange(1, cfg['wma_p'] + 1)) / np.arange(1, cfg['wma_p'] + 1).sum(), raw=True
    )
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=cfg['rsi_p']).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=cfg['rsi_p']).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    # MACD
    ema_fast = close.ewm(span=cfg['macd_f'], adjust=False).mean()
    ema_slow = close.ewm(span=cfg['macd_s'], adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    
    # Bollinger Bands
    df['BB_Mid'] = close.rolling(window=cfg['bb_p']).mean()
    bb_std = close.rolling(window=cfg['bb_p']).std()
    df['BB_Upper'] = df['BB_Mid'] + (cfg['bb_std'] * bb_std)
    df['BB_Lower'] = df['BB_Mid'] - (cfg['bb_std'] * bb_std)
    
    # ATR
    h_l = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=cfg['atr_p']).mean()
    
    # VWAP
    cum_vol_price = (close * vol).cumsum()
    cum_vol = vol.cumsum()
    df['VWAP'] = cum_vol_price / (cum_vol + 1e-10)
    
    # SuperTrend
    atr = df['ATR'].fillna(0)
    hl2 = (high + low) / 2.0
    upper_band = hl2 + (cfg['st_m'] * atr)
    lower_band = hl2 - (cfg['st_m'] * atr)
    
    st_arr = np.zeros(len(df))
    in_uptrend = True
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            in_uptrend = True
        elif close.iloc[i] < lower_band.iloc[i-1]:
            in_uptrend = False
            
        st_arr[i] = lower_band.iloc[i] if in_uptrend else upper_band.iloc[i]
    df['SuperTrend'] = st_arr
    
    return df

def calculate_rolling_correlation(series_a, series_b, lookback_window):
    arr_a = np.array(series_a, dtype=float).flatten()
    arr_b = np.array(series_b, dtype=float).flatten()
    min_len = min(len(arr_a), len(arr_b))
    if min_len == 0: return []
    df = pd.DataFrame({'Target_Curve': arr_a[:min_len], 'Match_Curve': arr_b[:min_len]})
    df['Ret_A'] = df['Target_Curve'].pct_change()
    df['Ret_C'] = df['Match_Curve'].pct_change()
    return df['Ret_A'].rolling(window=int(lookback_window)).corr(df['Ret_C']).fillna(0.0).tolist()

# PATH-DEPENDENT LOCAL HIGH RESET DRAWDOWN LOGIC
def calculate_hrf_path_drawdown(price_series):
    prices = np.array(price_series)
    n = len(prices)
    drawdowns = np.zeros(n)
    if n == 0: return drawdowns
    current_local_high = prices[0]
    for i in range(1, n):
        if prices[i] >= current_local_high:
            current_local_high = prices[i]
            drawdowns[i] = 0.0
        else:
            drawdowns[i] = (prices[i] - current_local_high) / current_local_high
    return drawdowns

# ==============================================================================
# ENGINE MODULE 1: ALGORITHMIC FRACTAL SCANNER
# ==============================================================================
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 TradingView Core Fractal Scanner")
    
    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", ["All Assets"] + list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1month', '1 week', '1day', '4h', '1h', '15m', '5m', '1m'], index=2)
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
            st.error("❌ Data download error from TradingView backend systems.")
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
            
        ax1.plot(target_scaled, color='#00ffcc', linewidth=4, label=f'TARGET: {t_asset} ({i_choice})', zorder=5)
        
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
                manual_matrix_boosted = manual_matrix * (1.0 + (vol_boost_pct / 100.0))
                m_mean = np.mean(manual_matrix_boosted, axis=0)
                m_std = np.std(manual_matrix_boosted, axis=0)
                
                if std_dev_multiplier > 0:
                    ax1.fill_between(range(len(m_mean)), m_mean - (m_std * std_dev_multiplier), m_mean + (m_std * std_dev_multiplier), color='#ff00ff', alpha=0.12)
                ax1.plot(m_mean, color='#ff00ff', linewidth=4, label='MANUAL COMPOSITE MEAN TRACK', zorder=6)

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
                    
                if all_scaled_paths_list:
                    stacked_paths = np.vstack(all_scaled_paths_list)
                    stacked_paths_boosted = stacked_paths * (1.0 + (vol_boost_pct / 100.0))
                    mean_path_array = np.mean(stacked_paths_boosted, axis=0)
                else:
                    mean_path_array = None
                    
                isolate_clean = iso_path.strip().lower()
                
                if isolate_clean != 'mean':
                    for idx, row in enumerate(unique_matches):
                        if isolate_clean.isdigit() and int(isolate_clean) != idx + 1: continue
                        raw_full = row['raw_prices']
                        scaled_full = ((np.array(raw_full) - raw_full[0]) / raw_full[0]) * 100.0
                        scaled_full_boosted = scaled_full * (1.0 + (vol_boost_pct / 100.0))
                        ax1.plot(scaled_full_boosted, linestyle='--', alpha=0.5, label=f"#{idx+1} {row['asset_name']} ({row['date_str']})")
                
                # --- FIXED LINE BELOW (CLEANED SYNTAX CONFLICT) ---
                if mean_path_array is not None and isolate_clean in ['all', 'mean']:
                    if std_dev_multiplier > 0:
                        std_path = np.std(stacked_paths * (1.0 + (vol_boost_pct / 100.0)), axis=0)
                        ax1.fill_between(range(len(mean_path_array)), mean_path_array - (std_path * std_dev_multiplier), mean_path_array + (std_path * std_dev_multiplier), color='#ff00ff', alpha=0.1)
                    ax1.plot(mean_path_array, color='#ff00ff', linewidth=4, label='COMPOSITE FRACTAL MEAN (Excluding 2026)', zorder=6)
                ax1.axvline(x=target_bars_num - 1, color='#ffffff', linestyle=':', alpha=0.5)

                if enable_corr and ax2 is not None and mean_path_array is not None:
                    r_wave = calculate_rolling_correlation(target_scaled, mean_path_array, c_win)
                    ax2.plot(r_wave, color='#ff00ff', linewidth=2.5, label=f"Rolling {c_win}-Bar Correlation")
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) >= 0), color='#00ff88', alpha=0.15)
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) < 0), color='#ff0055', alpha=0.15)

        ax1.axhline(y=0.0, color='#555555', linestyle='-', linewidth=1.2)
        ax1.set_title("HRF MATRIX ENGINE — Model By HRF", color='#ffffff', fontsize=12, fontweight='bold')
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
# ENGINE MODULE 2: STRUCTURAL CAPITULATION & MATHEMATICAL OPERATIONS ENGINE
# ==============================================================================
elif app_mode == "Structural Capitulation Wave":
    st.header("⏱️ Structural Capitulation Wave & Mathematical Matrix")
    
    stat_asset = st.sidebar.selectbox("Asset", list(ticker_map.keys()), index=0)
    stat_interval = st.sidebar.selectbox("Time Interval", ['1m', '5m', '15m', '1h', '4h', '1day', '1 week', '1month'], index=3)
    
    start_time_str = st.sidebar.text_input("Time Frame Start (YYYY-MM-DD:HH:MM)", value="2026-01-01:00:00")
    end_time_str = st.sidebar.text_input("Time Frame End (YYYY-MM-DD:HH:MM)", value="2026-07-14:00:00")
    
    target_indicator = st.sidebar.selectbox(
        "Indicators Matrix Selection", 
        ["Moving Average (Simple)", "Moving Average (Exponential)", "Moving Average (Weighted)", "RSI", "MACD", "Bollinger Band (Middle)", "Super Trend", "ATR", "VWAP"]
    )
    
    target_operation = st.sidebar.selectbox(
        "Mathematical Programmatic Operations", 
        ["None (Just Plot Asset & Indicator)", "Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)", "Power (^)", "Square Root (√)"]
    )
    
    with st.sidebar.expander("⚙️ Indicator Core Settings Parameters"):
        sma_p = st.number_input("Simple Moving Average Length", value=20, min_value=1)
        ema_p = st.number_input("Exponential Moving Average Length", value=20, min_value=1)
        wma_p = st.number_input("Weighted Moving Average Length", value=20, min_value=1)
        rsi_p = st.number_input("RSI Window Lookback", value=14, min_value=1)
        macd_f = st.number_input("MACD Fast Loop EMA Length", value=12, min_value=1)
        macd_s = st.number_input("MACD Slow Loop EMA Length", value=26, min_value=1)
        bb_p = st.number_input("Bollinger Bands Simple Period", value=20, min_value=1)
        bb_std = st.number_input("Bollinger Bands Dev Multiplier", value=2.0, min_value=0.1)
        atr_p = st.number_input("ATR Volatility Bands Period", value=14, min_value=1)
        st_m = st.number_input("SuperTrend Factor Multiplier", value=3.0, min_value=0.1)

    indicator_config = {
        'sma_p': int(sma_p), 'ema_p': int(ema_p), 'wma_p': int(wma_p),
        'rsi_p': int(rsi_p), 'macd_f': int(macd_f), 'macd_s': int(macd_s),
        'bb_p': int(bb_p), 'bb_std': float(bb_std), 'atr_p': int(atr_p), 'st_m': float(st_m)
    }

    try:
        df_raw = fetch_legacy_market_data(stat_asset, stat_interval)
        if df_raw is None or df_raw.empty:
            st.error("❌ Deep data stream failed to initiate for the selected parameters.")
            st.stop()
            
        df_calculated = calculate_custom_indicators(df_raw.copy(), indicator_config)
        
        ind_key_map = {
            "Moving Average (Simple)": "SMA", "Moving Average (Exponential)": "EMA", "Moving Average (Weighted)": "WMA",
            "RSI": "RSI", "MACD": "MACD", "Bollinger Band (Middle)": "BB_Mid", "Super Trend": "SuperTrend",
            "ATR": "ATR", "VWAP": "VWAP"
        }
        selected_ind_col = ind_key_map[target_indicator]
        
        def parse_custom_date(dt_str):
            try:
                parts = dt_str.strip().split(':')
                date_part = parts[0]
                hour = parts[1] if len(parts) > 1 else "00"
                minute = parts[2] if len(parts) > 2 else "00"
                return pd.to_datetime(f"{date_part} {hour}:{minute}:00")
            except:
                return pd.to_datetime(dt_str)
                
        start_dt = parse_custom_date(start_time_str)
        end_dt = parse_custom_date(end_time_str)
        
        df_filtered = df_calculated.loc[start_dt:end_dt].copy()
        if df_filtered.empty:
            st.error("⚠️ Filter Error: No timeline entries found matching those specific coordinate blocks.")
            st.stop()
            
        asset_val = df_filtered['close']
        ind_val = df_filtered[selected_ind_col]
        
        op_title = "Raw Plot Analysis"
        output_curve = asset_val.copy()
        
        if target_operation == "Add (+)":
            output_curve = asset_val + ind_val
            op_title = f"Sum Layer: [Asset + {selected_ind_col}]"
        elif target_operation == "Subtract (-)":
            output_curve = asset_val - ind_val
            op_title = f"Spread Layer: [Asset - {selected_ind_col}]"
        elif target_operation == "Multiply (*)":
            output_curve = asset_val * ind_val
            op_title = f"Product Matrix: [Asset * {selected_ind_col}]"
        elif target_operation == "Divide (/)":
            output_curve = asset_val / (ind_val + 1e-10)
            op_title = f"Ratio Signal Shift: [Asset / {selected_ind_col}]"
        elif target_operation == "Power (^)":
            output_curve = asset_val ** ind_val
            op_title = f"Exponential Growth: [Asset ^ {selected_ind_col}]"
        elif target_operation == "Square Root (√)":
            output_curve = np.sqrt(asset_val.abs())
            op_title = "Absolute Asset Vector Square Root (√)"

        plt.style.use('dark_background')
        
        if target_operation == "None (Just Plot Asset & Indicator)":
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))
            ax1.plot(df_filtered.index, asset_val, color='#9400d3', linewidth=2.5, label=f"Asset Close ({stat_asset})")
            ax1.plot(df_filtered.index, ind_val, color='#ff007f', linewidth=2.0, linestyle="--", label=f"Indicator ({target_indicator})")
            ax1.set_title(f"STRUCTURAL VISUAL METRIC MATRIX — Model By HRF", color='#ff007f', fontweight='bold', fontsize=12)
            ax1.grid(True, color='#222222', alpha=0.5)
            ax1.legend(loc='upper left', facecolor='#111111')
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1.5]})
            
            ax1.plot(df_filtered.index, asset_val, color='#9400d3', linewidth=2.5, label=f"Asset Close ({stat_asset})")
            ax1.plot(df_filtered.index, ind_val, color='#ffffff', linewidth=1.5, linestyle=":", label=f"Indicator ({target_indicator})")
            ax1.set_title(f"STRUCTURAL CAPITULATION WAVE COMPONENTS", color='#ff007f', fontweight='bold', fontsize=11)
            ax1.grid(True, color='#222222', alpha=0.4)
            ax1.legend(loc='upper left', facecolor='#111111')
            
            ax2.plot(df_filtered.index, output_curve, color='#ff0055', linewidth=3.0, label=op_title)
            ax2.fill_between(df_filtered.index, output_curve, output_curve.mean(), color='#9400d3', alpha=0.12)
            ax2.set_title(f"ALGEBRAIC OUTPUT COEFFICIENT WAVE", color='#ffffff', fontweight='bold', fontsize=10)
            ax2.grid(True, color='#222222', alpha=0.4)
            ax2.legend(loc='upper left', facecolor='#111111')
            
        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        
        st.subheader(f"📊 Matrix Metrics Sheet ({stat_asset} Engine Processing)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Asset Execution Price", f"${asset_val.iloc[-1]:,.2f}")
        c2.metric(f"Active {selected_ind_col} Terminal", f"{ind_val.iloc[-1]:,.4f}")
        c3.metric("Final Transformed Yield Value", f"{output_curve.iloc[-1]:,.4f}")
        
    except Exception as ex:
        st.error(f"Execution Error inside Statistical Pipeline Horizon: {ex}")

# ==============================================================================
# ENGINE MODULE 3: STATISTICAL FRAMEWORK CORE (PATH RESET DRAWDOWN)
# ==============================================================================
elif app_mode == "Statistical Framework Core":
    st.header("📈 Path-Dependent Statistical Framework Engine")
    
    stat_asset = st.sidebar.selectbox("Select Target Framework Asset", list(ticker_map.keys()), index=0)
    stat_interval = st.sidebar.selectbox("Framework Time Interval", ['1m', '5m', '15m', '1h', '4h', '1day', '1 week', '1month'], index=5)
    
    start_time_str = st.sidebar.text_input("Frame Start (YYYY-MM-DD)", value="2025-01-01")
    end_time_str = st.sidebar.text_input("Frame End (YYYY-MM-DD)", value="2026-07-14")
    
    try:
        df_raw = fetch_legacy_market_data(stat_asset, stat_interval)
        if df_raw is None or df_raw.empty:
            st.error("❌ Deep data stream failed to initiate for the statistical framework.")
            st.stop()
            
        start_dt = pd.to_datetime(start_time_str)
        end_dt = pd.to_datetime(end_time_str)
        
        df_filtered = df_raw.loc[start_dt:end_dt].copy()
        if df_filtered.empty:
            st.error("⚠️ Filter Error: No timeline records match your frame filters.")
            st.stop()
            
        # Run custom conditional path drawdown logic
        df_filtered['HRF_Drawdown'] = calculate_hrf_path_drawdown(df_filtered['close'])
        
        # Build Interactive Multi-Axis Chart Suite
        fig = go.Figure()
        
        # Main Price Track (Vibrant Purple)
        fig.add_trace(go.Scatter(
            x=df_filtered.index, 
            y=df_filtered['close'],
            mode='lines',
            name=f'{stat_asset} Price',
            line=dict(color='#8A2BE2', width=2.5)
        ))
        
        # Path Drawdown Layer (Vibrant Red)
        fig.add_trace(go.Scatter(
            x=df_filtered.index, 
            y=df_filtered['HRF_Drawdown'] * 100,
            mode='lines',
            name='Path-Dependent Drawdown (%)',
            line=dict(color='#FF0055', width=1.5),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"<b>{stat_asset} Framework - Model By HRF</b>",
            xaxis=dict(title="Timeline Series", gridcolor='#222222'),
            yaxis=dict(title="Asset Price ($)", side="left", gridcolor='#222222'),
            yaxis2=dict(title="Conditional Drawdown (%)", side="right", overlaying="y", showgrid=False, range=[-100, 5]),
            template="plotly_dark",
            paper_bgcolor='#0d0d11',
            plot_bgcolor='#0d0d11',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Micro Summary Block
        st.subheader("📊 Path Analysis Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Current Asset Close", f"${df_filtered['close'].iloc[-1]:,.2f}")
        m2.metric("Max Trapped Path Drawdown", f"{df_filtered['HRF_Drawdown'].min() * 100:.2f}%")
        
    except Exception as ex:
        st.error(f"Ecosystem failed inside statistical block calculations: {ex}")

# Global Watermark Footer
st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
