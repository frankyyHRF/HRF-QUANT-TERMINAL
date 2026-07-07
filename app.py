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

st.title("🏛️ HRF QUANT MASTER PLATFORM V2.1")
st.caption("Unified Quantitative Ecosystem with Rolling Inter-Asset Correlation Matrices — Model By HRF")
st.divider()

# --- DUAL-ENGINE NAVIGATION ---
app_mode = st.sidebar.selectbox("🚀 Choose Analysis Core Engine", ["Algorithmic Fractal Scan", "Structural Capitulation Wave"])

# Global Setup Configurations
ticker_map = {
    'Bitcoin (BTC)': ('BTCUSDT', 'BINANCE'),
    'Ethereum (ETH)': ('ETHUSDT', 'BINANCE'),
    'Ripple (XRP)': ('XRPUSDT', 'BINANCE'),
    'Binance Coin (BNB)': ('BNBUSDT', 'BINANCE'),
    'Solana (SOL)': ('SOLUSDT', 'BINANCE'),
    'S&P 500 Index (SPY)': ('SPY', 'AMEX'),
    'Nasdaq 100 (QQQ)': ('QQQ', 'NASDAQ'),
    'Hang Seng Index (HSI)': ('HSI', 'INDEX'),
    'Gold (GLD)': ('GLD', 'AMEX'),
    'Silver (SLV)': ('SLV', 'AMEX'),
    'Palladium Futures': ('PALL', 'AMEX'),
    'EUR/USD': ('EURUSD', 'FX_IDC'),
    'USD/JPY': ('USDJPY', 'FX_IDC'),
    'GBP/USD': ('GBPUSD', 'FX_IDC')
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
# FIXED INTER-ASSET SEQUENTIAL CORRELATION MATH PIPELINE
# ==============================================================================
def calculate_rolling_correlation(series_a, series_b, lookback_window):
    """
    Computes a point-by-point rolling correlation based on chart index positions,
    completely bypassing calendar date mismatch conflicts.
    """
    arr_a = np.array(series_a, dtype=float).flatten()
    arr_b = np.array(series_b, dtype=float).flatten()
    
    min_len = min(len(arr_a), len(arr_b))
    if min_len == 0:
        return []
        
    arr_a = arr_a[:min_len]
    arr_b = arr_b[:min_len]
    
    df = pd.DataFrame({
        'Target_Curve': arr_a,
        'Match_Curve': arr_b
    })
    
    df['Ret_A'] = df['Target_Curve'].pct_change()
    df['Ret_C'] = df['Match_Curve'].pct_change()
    
    df['Rolling_R'] = df['Ret_A'].rolling(window=int(lookback_window)).corr(df['Ret_C'])
    return df['Rolling_R'].fillna(0.0).tolist()

# ==============================================================================
# ENGINE MODULE 1: ALGORITHMIC FRACTAL GRID SCANNER
# ==============================================================================
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 Algorithmic Fractal, Structural Sandbox & Correlation Grid")
    
    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", ["All Assets"] + list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1M', '1w', '1d', '1h', '15m', '5m'], index=2)
    f_mode = st.sidebar.selectbox("Framework Processing Mode", ["Calculate Fractals", "Manual Compare"], index=0)
    
    st.sidebar.markdown("### 📊 Inter-Asset Correlation Layer")
    enable_corr = st.sidebar.checkbox("Overlay Rolling Macro Asset Correlation", value=False)
    corr_window = st.sidebar.text_input("Correlation Lookback Window (Bars)", value="10")
    
    start_d = st.sidebar.text_input("Analysis Target Window Start", value="latest")
    end_d = st.sidebar.text_input("Analysis Target Window End", value="latest")
    
    ov_starts = st.sidebar.text_input("Manual Overlay Window Starts (Comma Separated)", value="2020-03-01, 2022-11-01")
    ov_ends = st.sidebar.text_input("Manual Overlay Window Ends (Comma Separated)", value="2020-05-01, 2023-01-01")
    
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

    interval_map = {'1m': Interval.in_1_minute, '5m': Interval.in_5_minute, '15m': Interval.in_15_minute, '1h': Interval.in_1_hour, '1d': Interval.in_daily, '1w': Interval.in_weekly, '1M': Interval.in_monthly}
    chosen_interval = interval_map.get(i_choice.lower(), Interval.in_daily)
    max_bars = 16000 if i_choice in ['1d', '1w', '1M'] else 4900

    tv = TvDatafeed()
    sym, exch = ticker_map[t_asset]
    
    try:
        df_target = tv.get_hist(symbol=sym, exchange=exch, interval=chosen_interval, n_bars=max_bars)
        df_target.index = pd.to_datetime(df_target.index).tz_localize(None)
        close_target = df_target['close'].dropna()
        
        start_clean = start_d.strip().lower()
        end_clean = end_d.strip().lower()
        
        if start_clean == 'latest' or end_clean == 'latest':
            target_df = close_target.iloc[-target_bars_num:]
        else:
            target_df = close_target.loc[pd.to_datetime(start_clean):pd.to_datetime(end_clean)]
            target_bars_num = len(target_df)
            
        if len(target_df) == 0:
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
        
        # --- MANUAL COMPARE MODE ---
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

                # Execute dynamic sequential correlation between Cyan curve and Yellow curve
                if enable_corr and ax2 is not None:
                    r_wave = calculate_rolling_correlation(target_scaled, m_mean, c_win)
                    ax2.plot(r_wave, color='#ffff00', linewidth=2.5, label=f"Rolling {c_win}-Bar Correlation (Cyan vs Yellow Line)")
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) >= 0), color='#00ff88', alpha=0.15)
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) < 0), color='#ff0055', alpha=0.15)

        # --- CALCULATE FRACTALS MODE ---
        else:
            assets_to_scan = list(ticker_map.keys()) if s_pool == "All Assets" else [s_pool]
            all_discovered_results = []
            
            for asset_item in assets_to_scan:
                s_sym, s_exch = ticker_map[asset_item]
                try:
                    df_scan = tv.get_hist(symbol=s_sym, exchange=s_exch, interval=chosen_interval, n_bars=max_bars)
                    if df_scan is None or df_scan.empty: continue
                    df_scan.index = pd.to_datetime(df_scan.index).tz_localize(None)
                    close_scan = df_scan['close'].dropna()
                except:
                    continue
                
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
                        'date_str': history_dates[i].strftime('%Y-%m-%d')
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
                    base_p = raw_full[0]
                    scaled_full = ((np.array(raw_full) - base_p) / base_p) * 100.0 if base_p != 0 else np.zeros_like(raw_full)
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

                # In calculation scan mode, correlate the Target Cyan line vs the Composite Mean Yellow line
                if enable_corr and ax2 is not None and mean_path_array is not None:
                    r_wave = calculate_rolling_correlation(target_scaled, mean_path_array, c_win)
                    ax2.plot(r_wave, color='#ffff00', linewidth=2.5, label=f"Rolling {c_win}-Bar Correlation (Cyan vs Mean Fractal)")
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) >= 0), color='#00ff88', alpha=0.15)
                    ax2.fill_between(range(len(r_wave)), r_wave, 0, where=(np.array(r_wave) < 0), color='#ff0055', alpha=0.15)

        ax1.axhline(y=0.0, color='#555555', linestyle='-', linewidth=1.2)
        ax1.set_title("HRF MATRIX ENGINE — UNIFIED MULTI-ASSET CORE CANVAS", color='#ffffff', fontsize=12, fontweight='bold')
        ax1.set_ylabel("Percentage Performance Shift (%)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax1.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
        
        if enable_corr and ax2 is not None:
            ax2.axhline(0.0, color='#ffffff', linestyle='-', linewidth=0.8, alpha=0.5)
            ax2.axhline(0.7, color='#00ff88', linestyle=':', alpha=0.8, label="High Correlation (> 0.7)")
            ax2.axhline(-0.7, color='#ff0055', linestyle=':', alpha=0.8, label="Inverse Separation (< -0.7)")
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Correlation R-Scale")
            ax2.grid(True, color='#222222')
            ax2.legend(loc='lower left', facecolor='#111111', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_frac)

    except Exception as e:
        st.error(f"Ecosystem loading block halted: {e}")

# ==============================================================================
# ENGINE MODULE 2: CAPITULATION & DAYS SINCE RESET ENGINE
# ==============================================================================
else:
    st.header("⏱️ Path-Dependent Peak-To-Trough Reset Wave")
    
    vol_input = st.sidebar.text_input("Volatility Reset Threshold (%)", value="-40.0")
    pattern_input = st.sidebar.text_input("Candle Sequence Pattern (G/R)", value="RRGG")
    
    @st.cache_data(show_spinner="🔄 Streaming historical data streams...")
    def load_capitulation_data():
        tv = TvDatafeed()
        df_d = tv.get_hist(symbol="BLX", exchange="BNC", interval=Interval.in_daily, n_bars=18000)
        df_d.index = pd.to_datetime(df_d.index).tz_localize(None)
        df_w = tv.get_hist(symbol="BLX", exchange="BNC", interval=Interval.in_weekly, n_bars=2000)
        df_w.index = pd.to_datetime(df_w.index).tz_localize(None)
        df_h = tv.get_hist(symbol="BTCUSDT", exchange="BINANCE", interval=Interval.in_1_hour, n_bars=15000)
        df_h.index = pd.to_datetime(df_h.index).tz_localize(None)
        
        for df in [df_d, df_w, df_h]:
            df['return'] = ((df['close'] - df['open']) / df['open']) * 100.0
            df['is_midterm'] = (df.index.year % 4 == 2)
        return df_d, df_w, df_h

    try:
        df_daily_raw, df_weekly, df_hourly = load_capitulation_data()
        df_daily = df_daily_raw.copy()
        
        thresh = float(vol_input)
        days_since_tracker, historical_gaps = [], []
        current_accumulator = 0
        
        if thresh < 0:
            running_peak = df_daily['high'].iloc[0]
            for idx in range(len(df_daily)):
                current_high = df_daily['high'].iloc[idx]
                current_low = df_daily['low'].iloc[idx]
                if current_high > running_peak: running_peak = current_high
                drawdown_pct = ((current_low - running_peak) / running_peak) * 100.0
                if drawdown_pct <= thresh:
                    days_since_tracker.append(0)
                    if current_accumulator > 0: historical_gaps.append(current_accumulator)
                    current_accumulator = 0
                    running_peak = current_high
                else:
                    current_accumulator += 1
                    days_since_tracker.append(current_accumulator)
        else:
            running_trough = df_daily['low'].iloc[0]
            for idx in range(len(df_daily)):
                current_high = df_daily['high'].iloc[idx]
                current_low = df_daily['low'].iloc[idx]
                if current_low < running_trough: running_trough = current_low
                expansion_pct = ((current_high - running_trough) / running_trough) * 100.0
                if expansion_pct >= thresh:
                    days_since_tracker.append(0)
                    if current_accumulator > 0: historical_gaps.append(current_accumulator)
                    current_accumulator = 0
                    running_trough = current_low
                else:
                    current_accumulator += 1
                    days_since_tracker.append(current_accumulator)
        
        current_live_gap = days_since_tracker[-1]
        
        clean_pattern = pattern_input.strip().upper()
        candle_string_sequence = "".join(['G' if r >= 0 else 'R' for r in df_daily['return'].tolist()])
        match_indices = [i + len(clean_pattern) for i in range(len(candle_string_sequence) - len(clean_pattern)) if candle_string_sequence[i : i + len(clean_pattern)] == clean_pattern]
        
        post_pattern_returns = [df_daily['return'].iloc[m] for m in match_indices if m < len(df_daily)]
        
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        ax1.hist(df_daily['return'].dropna().values, bins=100, range=(-15, 15), color='#8a2be2', alpha=0.6, label='Daily Lifetime')
        ax1.set_title("HISTOGRAM 1: TOTAL DAILY RETURN FREQUENCY", fontweight='bold', fontsize=10)
        ax1.legend(facecolor='#111111')
        
        ax2.hist(df_daily[df_daily['is_midterm']]['return'].dropna().values, bins=100, range=(-15, 15), color='#ff0055', alpha=0.6, label='Midterm Regime')
        ax2.set_title("HISTOGRAM 2: MIDTERM YEAR CYCLE FREQUENCY", fontweight='bold', fontsize=10)
        ax2.legend(facecolor='#111111')
        
        ax3.plot(df_daily.index, days_since_tracker, color='#00ffcc', linewidth=1.2, label='Days Since')
        ax3.fill_between(df_daily.index, 0, days_since_tracker, color='#00ffcc', alpha=0.06)
        ax3.set_title("CHART 3: STRUCTURAL RESETS TIMELINE WAVE", fontweight='bold', fontsize=10)
        ax3.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2026-07-01'))
        
        if post_pattern_returns:
            ax4.hist(post_pattern_returns, bins=30, range=(-15, 15), color='#ffff00', alpha=0.6, label=f"Post '{clean_pattern}'")
        ax4.set_title("HISTOGRAM 4: FORWARD EDGE PERFORMANCE", fontweight='bold', fontsize=10)
        ax4.legend(facecolor='#111111')
        
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)
        
        st.subheader("📊 Performance Summary Matrix")
        metrics_data = {
            "Timeframe Segment": ["Daily Changes", "Weekly Changes", "Hourly Changes"],
            "Lifetime History Average": [f"{df_daily['return'].mean():+.4f}%", f"{df_weekly['return'].mean():+.4f}%", f"{df_hourly['return'].mean():+.4f}%"],
            "US Midterm Cycles Only": [f"{df_daily[df_daily['is_midterm']]['return'].mean():+.4f}%", f"{df_weekly[df_weekly['is_midterm']]['return'].mean():+.4f}%", f"{df_hourly[df_hourly['is_midterm']]['return'].mean():+.4f}%"]
        }
        st.table(pd.DataFrame(metrics_data))
        
        col1, col2 = st.columns(2)
        col1.metric("Current Days Stretched Under Matrix", f"{current_live_gap} Days")
        col2.metric(f"Historical '{clean_pattern}' Frequency Discoveries", f"{len(match_indices)} Matches")
        
    except Exception as ex:
        st.error(f"Execution Error: {ex}")

st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
               
