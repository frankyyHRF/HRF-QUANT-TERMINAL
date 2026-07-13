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

st.title("🏛️ HRF QUANT MASTER PLATFORM V5.0")
st.caption("TradingView Core Multi-Engine — Model By HRF")
st.divider()

# --- NAVIGATION ---
app_mode = st.sidebar.selectbox(
    "🚀 Choose Analysis Core Engine", 
    ["Algorithmic Fractal Scan", "Structural Capitulation Wave", "Technical Indicators Matrix"]
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
# ENGINE MODULE 1: ALGORITHMIC FRACTAL SCANNER
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
# ENGINE MODULE 2: REBUILT STATISTICAL ENGINE (CAPITULATION & RESET WAVE)
# ==============================================================================
elif app_mode == "Structural Capitulation Wave":
    st.header("⏱️ Path-Dependent Peak-To-Trough Reset Wave")
    
    # NEW REBUILT CORE SELECTION CAPABILITIES
    stat_asset = st.sidebar.selectbox("Statistical Focus Target Asset", list(ticker_map.keys()), index=0)
    stat_interval = st.sidebar.selectbox("Statistical Horizon Time Interval", ['1M', '1w', '1d', '4h', '1h', '15m', '5m', '1m'], index=2)
    
    vol_input = st.sidebar.text_input("Volatility Reset Threshold (%)", value="-40.0")
    pattern_input = st.sidebar.text_input("Candle Sequence Pattern (G/R)", value="RRGG")
    
    # NEW % CHANGE N-BAR CONFIGURATION
    n_bars_pct = st.sidebar.text_input("Calculate % Change Over N-Bars", value="1")
    stat_year_filter = st.sidebar.text_input("Restrict Statistical View to Specific Year (or 'all')", value="all")

    try:
        thresh = float(vol_input)
        n_pct = int(n_bars_pct)
    except ValueError:
        st.error("⚠️ Dynamic Variable Error: Verify parameters contain clean mathematical integers.")
        st.stop()
        
    @st.cache_data(show_spinner="🔄 Fetching Selected Assets Over Active Pipeline Horizon...")
    def load_dynamic_statistical_data(asset_name, interval_str, n_bars_lookback):
        df = fetch_legacy_market_data(asset_name, interval_str)
        if df is not None:
            # Custom % Change over N-bars formula
            df['return'] = df['close'].pct_change(periods=n_bars_lookback) * 100.0
            df['is_midterm'] = (df.index.year % 4 == 2)
        return df

    try:
        df_stat_raw = load_dynamic_statistical_data(stat_asset, stat_interval, n_pct)
        if df_stat_raw is None or df_stat_raw.empty:
            st.error("❌ Deep data stream failed to initiate for the selected parameters.")
            st.stop()
            
        df_stat = df_stat_raw.dropna(subset=['return']).copy()
        
        # Apply strict calendar year boundaries if specified
        clean_yr = stat_year_filter.strip().lower()
        if clean_yr != 'all' and clean_yr.isdigit():
            df_stat = df_stat[df_stat.index.year == int(clean_yr)]
            if df_stat.empty:
                st.warning(f"No pricing historical points available for year matching: {clean_yr}")
                st.stop()

        days_since_tracker = []
        current_accumulator = 0
        
        # --- CONDITIONAL PATH-DEPENDENT RESET LOGIC ---
        if thresh < 0:
            running_peak = df_stat['high'].iloc[0]
            for idx in range(len(df_stat)):
                current_high = df_stat['high'].iloc[idx]
                current_low = df_stat['low'].iloc[idx]
                
                if current_high > running_peak: 
                    running_peak = current_high
                    
                drawdown_pct = ((current_low - running_peak) / running_peak) * 100.0
                
                if drawdown_pct <= thresh:
                    days_since_tracker.append(0)
                    current_accumulator = 0
                    running_peak = current_high  
                else:
                    current_accumulator += 1
                    days_since_tracker.append(current_accumulator)
        else:
            running_trough = df_stat['low'].iloc[0]
            for idx in range(len(df_stat)):
                current_high = df_stat['high'].iloc[idx]
                current_low = df_stat['low'].iloc[idx]
                
                if current_low < running_trough: 
                    running_trough = current_low
                    
                expansion_pct = ((current_high - running_trough) / running_trough) * 100.0
                
                if expansion_pct >= thresh:
                    days_since_tracker.append(0)
                    current_accumulator = 0
                    running_trough = current_low  
                else:
                    current_accumulator += 1
                    days_since_tracker.append(current_accumulator)
        
        current_live_gap = days_since_tracker[-1] if days_since_tracker else 0
        clean_pattern = pattern_input.strip().upper()
        candle_string_sequence = "".join(['G' if r >= 0 else 'R' for r in df_stat['return'].tolist()])
        match_indices = [i + len(clean_pattern) for i in range(len(candle_string_sequence) - len(clean_pattern)) if candle_string_sequence[i : i + len(clean_pattern)] == clean_pattern]
        
        # VIBRANT VISUAL THEME (RED & PURPLE MATRIX COLORS)
        plt.style.use('dark_background')
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.hist(df_stat['return'].values, bins=100, color='#9400d3', alpha=0.6, edgecolor='#ff007f', label=f'{stat_asset} Ret Distribution')
        ax1.set_title(f"HISTOGRAM: {n_bars_pct}-BAR DISTRIBUTION SHIFT", color='#ff007f', fontweight='bold', fontsize=10)
        ax1.grid(True, color='#222222', alpha=0.5)
        
        ax3.plot(df_stat.index, days_since_tracker, color='#ff0055', linewidth=1.5, label='Bars Since Reset')
        ax3.fill_between(df_stat.index, 0, days_since_tracker, color='#9400d3', alpha=0.15)
        ax3.set_title("CHART: PATH-DEPENDENT DYNAMIC TIMELINE WAVE", color='#ff007f', fontweight='bold', fontsize=10)
        ax3.grid(True, color='#222222', alpha=0.5)
        
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)
        
        st.subheader(f"📊 Performance Summary Matrix ({stat_asset} @ {stat_interval})")
        metrics_data = {
            "Statistical Set Grouping": [f"Selected Series (Lookback: {n_bars_pct} Bars)", "US Midterm Cycle Periods"],
            "Calculated Average Returns": [f"{df_stat['return'].mean():+.4f}%", f"{df_stat[df_stat['is_midterm']]['return'].mean():+.4f}%"],
            "Standard Deviation Variance": [f"{df_stat['return'].std():.4f}%", f"{df_stat[df_stat['is_midterm']]['return'].std():.4f}%"]
        }
        st.table(pd.DataFrame(metrics_data))
        
        col1, col2 = st.columns(2)
        col1.metric("Current Window Elongation Run", f"{current_live_gap} Bars")
        col2.metric(f"Sequential Pattern '{clean_pattern}' Identifications", f"{len(match_indices)} Matches")
    except Exception as ex:
        st.error(f"Execution Error inside Statistical Pipeline: {ex}")

# ==============================================================================
# ENGINE MODULE 3: RESTORED TECHNICAL INDICATORS MATRIX
# ==============================================================================
else:
    st.header("⚡ Core Technical Analysis Indicators Matrix")
    
    ind_asset = st.sidebar.selectbox("Indicators Target Asset Focus", list(ticker_map.keys()), index=0)
    ind_interval = st.sidebar.selectbox("Indicators Framework Interval", ['1M', '1w', '1d', '4h', '1h', '15m', '5m', '1m'], index=2)
    
    ma_fast = st.sidebar.text_input("Fast Moving Average Parameter", value="20")
    ma_slow = st.sidebar.text_input("Slow Moving Average Parameter", value="50")
    rsi_window = st.sidebar.text_input("Relative Strength Index Lookback", value="14")
    
    try:
        fast_w = int(ma_fast)
        slow_w = int(ma_slow)
        rsi_w = int(rsi_window)
    except ValueError:
        st.error("Input Warning: Verify moving average fields contain numeric context strings.")
        st.stop()
        
    try:
        df_ind = fetch_legacy_market_data(ind_asset, ind_interval)
        if df_ind is None or df_ind.empty:
            st.error("❌ Data retrieval breakdown within TradingView endpoints framework.")
            st.stop()
            
        # Calculation Layers
        df_ind['Fast_MA'] = df_ind['close'].rolling(window=fast_w).mean()
        df_ind['Slow_MA'] = df_ind['close'].rolling(window=slow_w).mean()
        
        # Relative Strength Formula
        delta = df_ind['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_w).mean()
        rs = gain / loss
        df_ind['RSI'] = 100 - (100 / (1 + rs))
        
        plt.style.use('dark_background')
        fig_ind, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        
        # Panel 1 - Price Action Tracks
        ax_p.plot(df_ind.index, df_ind['close'], color='#ffffff', linewidth=2.0, label=f'Price Close ({ind_asset})')
        ax_p.plot(df_ind.index, df_ind['Fast_MA'], color='#ff00ff', linewidth=1.5, linestyle='--', label=f'Fast MA ({fast_w})')
        ax_p.plot(df_ind.index, df_ind['Slow_MA'], color='#8a2be2', linewidth=1.5, label=f'Slow MA ({slow_w})')
        ax_p.set_title("TECHNICAL PRICE METRICS CORE TRACKS", fontweight='bold', color='#ffffff')
        ax_p.set_ylabel("Asset Unit Denomination Value")
        ax_p.legend(loc='upper left', facecolor='#111111')
        ax_p.grid(True, color='#222222')
        
        # Panel 2 - RSI Track
        ax_r.plot(df_ind.index, df_ind['RSI'], color='#ff0055', linewidth=1.8, label=f'RSI ({rsi_w})')
        ax_r.axhline(70, color='#ff0055', linestyle=':', alpha=0.7)
        ax_r.axhline(30, color='#00ffcc', linestyle=':', alpha=0.7)
        ax_r.fill_between(df_ind.index, 70, 30, color='#8a2be2', alpha=0.04)
        ax_r.set_ylabel("RSI Gauge Bounds")
        ax_r.set_ylim(10, 90)
        ax_r.legend(loc='upper left', facecolor='#111111')
        ax_r.grid(True, color='#222222')
        
        st.pyplot(fig_ind)
        
        # Display localized dashboard overview states
        latest_row = df_ind.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Market Value", f"${latest_row['close']:.2f}")
        c2.metric(f"Active Fast MA ({fast_w})", f"${latest_row['Fast_MA']:.2f}")
        c3.metric(f"Relative Strength Gauge ({rsi_w})", f"{latest_row['RSI']:.2f}")
        
    except Exception as ex:
        st.error(f"Framework Engine Interrupted: {ex}")

st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
