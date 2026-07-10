import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io
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

st.title("🏛️ HRF QUANT MASTER PLATFORM V4.6")
st.caption("Zip Archive Multi-Asset Aggregator Engine — Model By HRF")
st.divider()

# --- GOOGLE DRIVE / ZIP DATA INGESTION LAYER ---
st.sidebar.markdown("### 💾 Core Data Ingestion Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Combined .ZIP Archive", type=["csv", "zip"])

# New feature: Remote streaming link input box placed directly in the ingestion flow
drive_url_input = st.sidebar.text_input("Or Paste Shared Google Drive CSV Link:", value="")

# Read, merge, and cache multiple CSVs from a single ZIP file dynamically
@st.cache_data(show_spinner=False)
def load_drive_dataset(file_obj, url_string=None):
    # Process remote link if provided and no manual file is uploaded
    if file_obj is None and url_string:
        try:
            if "id=" in url_string:
                file_id = url_string.split("id=")[1].split("&")[0]
            elif "file/d/" in url_string:
                file_id = url_string.split("file/d/")[1].split("/")[0]
            else:
                file_id = url_string
            
            direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            df = pd.read_csv(direct_download_url)
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                df.sort_index(inplace=True)
            return df
        except Exception as e:
            st.sidebar.error(f"Google Drive Link Stream Error: {e}")
            return None

    if file_obj is None:
        return None
    try:
        # Scenario A: User uploaded a combined ZIP file
        if file_obj.name.endswith('.zip'):
            combined_df = pd.DataFrame()
            with zipfile.ZipFile(file_obj) as z:
                for file_name in z.namelist():
                    if file_name.endswith('.csv') and not file_name.startswith('__MACOSX'):
                        with z.open(file_name) as f:
                            df = pd.read_csv(f)
                            df.columns = [c.strip().lower() for c in df.columns]
                            
                            # Clean up time/date indexing
                            date_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
                            if date_col:
                                df[date_col] = pd.to_datetime(df[date_col])
                                df.set_index(date_col, inplace=True)
                                
                            # Identify the price column
                            close_col = next((c for c in df.columns if 'close' in c or 'price' in c), None)
                            if close_col:
                                # Clean asset label from file name (e.g., "BTCUSDT_1d.csv" -> "btcusdt")
                                asset_label = file_name.lower().split('.')[0].split('_')[0]
                                temp_series = df[[close_col]].rename(columns={close_col: asset_label})
                                if combined_df.empty:
                                    combined_df = temp_series
                                else:
                                    combined_df = combined_df.join(temp_series, how='outer')
            if not combined_df.empty:
                combined_df.sort_index(inplace=True)
                return combined_df
                
        # Scenario B: User uploaded a single standalone CSV file
        else:
            df = pd.read_csv(file_obj)
            df.columns = [c.strip().lower() for c in df.columns]
            date_col = next((c for c in df.columns if 'time' in c or 'date' in c), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                df.sort_index(inplace=True)
            return df
    except Exception as e:
        st.sidebar.error(f"Engine parsing error: {e}")
        return None

# Combined link + file integration pass
drive_df = load_drive_dataset(uploaded_file, drive_url_input)

if drive_df is not None:
    st.sidebar.success(f"✅ Archive Loaded! Discovered columns: {', '.join(list(drive_df.columns)[:5])}...")

# --- NAVIGATION & TICKER MAP ---
app_mode = st.sidebar.selectbox("🚀 Choose Analysis Core Engine", ["Algorithmic Fractal Scan", "Structural Capitulation Wave"])

ticker_map = {
    'Bitcoin (BTC)': ('BTCUSD', 'BINANCE'),
    'Ethereum (ETH)': ('ETHUSD', 'BINANCE'),
    'S&P 500 (SPX)': ('SPX', 'SP'),
    'Nasdaq 100 (QQQ)': ('QQQ', 'NASDAQ'),
    'Binance Coin (BNB)': ('BNBUSD', 'BINANCE'),
    'Solana (SOL)': ('SOLUSD', 'BINANCE'),
    'Cardano (ADA)': ('ADAUSD', 'BINANCE'),
    'Dogecoin (DOGE)': ('DOGEUSD', 'BINANCE'),
    'TRON (TRX)': ('TRXUSD', 'BINANCE'),
    'Bitcoin Cash (BCH)': ('BCHUSD', 'BINANCE'),
    'Litecoin (LTC)': ('LTCUSD', 'BINANCE'),
    'Bittensor (TAO)': ('TAOUSD', 'BINANCE')
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

# --- HYBRID RECONSTRUCTION SOURCE INTERCEPT ENGINE ---
@st.cache_data(show_spinner=False)
def fetch_legacy_market_data(asset_name, interval_str):
    is_crypto = "SPX" not in asset_name and "QQQ" not in asset_name
    
    if is_crypto and drive_df is not None:
        try:
            clean_name = asset_name.lower()
            # Try to catch tokens matching file sub-strings (e.g., "btc" inside "bitcoin (btc)")
            matching_col = next((c for c in drive_df.columns if c in clean_name or clean_name in c), None)
            
            if matching_col:
                df_slice = drive_df[[matching_col]].dropna().copy()
                df_slice.columns = ['close']
                df_slice['open'] = df_slice['close']
                df_slice['high'] = df_slice['close']
                df_slice['low'] = df_slice['close']
                df_slice['volume'] = 0
                return df_slice
        except:
            pass

    symbol, exchange = ticker_map.get(asset_name, ('BTCUSD', 'BINANCE'))
    interval_dict = {
        '1M': Interval.in_monthly, '1w': Interval.in_weekly, '1d': Interval.in_daily,
        '4h': Interval.in_4_hour, '1h': Interval.in_1_hour
    }
    tv_interval = interval_dict.get(interval_str, Interval.in_daily)
        
    try:
        tv = TvDatafeed()
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=tv_interval, n_bars=3000)
        if df is None or df.empty: return None
        df.index.name = 'time'
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return None

# --- ENGINE MODULE 1 (FRACTAL SCANNER) ---
if app_mode == "Algorithmic Fractal Scan":
    st.header("🎯 System Core Fractal Scanner")
    
    t_asset = st.sidebar.selectbox("Baseline Target Asset", list(ticker_map.keys()), index=0)
    s_pool = st.sidebar.selectbox("Scan Matching Pool Range", ["All Assets"] + list(ticker_map.keys()), index=0)
    i_choice = st.sidebar.selectbox("Sequence Time Frame Interval", ['1M', '1w', '1d', '4h', '1h'], index=2)
    f_mode = st.sidebar.selectbox("Framework Processing Mode", ["Calculate Fractals", "Manual Compare"], index=0)
    
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
    except ValueError:
        st.error("⚠️ Input Parse Warning: Confirm all dynamic inputs contain clean numeric integers.")
        st.stop()

    try:
        df_target = fetch_legacy_market_data(t_asset, i_choice)
        if df_target is None or df_target.empty:
            st.error("❌ Data source stream blank or unavailable. Check variables.")
            st.stop()
            
        close_target = df_target['close'].dropna()
        target_df = close_target.iloc[-target_bars_num:]
        target_bars_num = len(target_df)
        target_scaled = percentage_return_scale(target_df.tolist())
        
        plt.style.use('dark_background')
        fig_frac, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        ax1.plot(target_scaled, color='#00ffcc', linewidth=4, label=f'TARGET: {t_asset}', zorder=5)
        
        if f_mode != "Manual Compare":
            assets_to_scan = list(ticker_map.keys()) if s_pool == "All Assets" else [s_pool]
            all_discovered_results = []
            
            for asset_item in assets_to_scan:
                df_scan = fetch_legacy_market_data(asset_item, i_choice)
                if df_scan is None or df_scan.empty: continue
                close_scan = df_scan['close'].dropna()
                
                if asset_item == t_asset:
                    history_pool = close_scan.iloc[:-target_bars_num].tolist()
                    history_dates = close_scan.index[:-target_bars_num]
                else:
                    history_pool = close_scan.tolist()
                    history_dates = close_scan.index
                    
                max_search_index = len(history_pool) - (target_bars_num + forecast_bars_num)
                if max_search_index <= 0: continue
                
                parsed_years = [int(y.strip()) for y in spec_years.split(',') if y.strip() and y.lower() != 'all']
                
                for i in range(max_search_index):
                    try:
                        hist_year = history_dates[i].year
                        if hist_year == 2026: continue
                        hist_regime = get_election_regime(hist_year)
                    except:
                        hist_year = 0
                        hist_regime = 'All Cycles'
                    
                    if c_filter != 'All Cycles' and hist_regime != c_filter: continue
                    if parsed_years and hist_year not in parsed_years: continue
                    
                    hist_pattern = history_pool[i : i + target_bars_num]
                    hist_scaled = percentage_return_scale(hist_pattern)
                    
                    mse = float(np.mean((target_scaled - hist_scaled) ** 2))
                    corr = float(np.corrcoef(target_scaled, hist_scaled)[0, 1]) if len(target_scaled) > 1 else 0.0
                    
                    all_discovered_results.append({
                        'asset_name': asset_item, 'start_index': i, 'end_index': i + target_bars_num,
                        'mse': mse, 'correlation': corr, 'year': hist_year,
                        'raw_prices': history_pool[i : i + target_bars_num + forecast_bars_num],
                        'date_str': history_dates[i].strftime('%Y-%m-%d') if hasattr(history_dates[i], 'strftime') else f"Idx {i}"
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

        ax1.axhline(y=0.0, color='#555555', linestyle='-', linewidth=1.2)
        ax1.set_title("HRF MATRIX ENGINE — MULTI-ASSET COMPRESSION LAYER", color='#ffffff', fontsize=12, fontweight='bold')
        ax1.set_ylabel("Percentage Performance Shift (%)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
        ax1.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
        
        plt.tight_layout()
        st.pyplot(fig_frac)
    except Exception as e:
        st.error(f"Ecosystem halted: {e}")

# --- ENGINE MODULE 2 (STRUCTURAL CAPITULATION WAVE) ---
else:
    st.header("⏱️ Path-Dependent Peak-To-Trough Reset Wave")
    vol_input = st.sidebar.text_input("Volatility Reset Threshold (%)", value="-40.0")
    pattern_input = st.sidebar.text_input("Candle Sequence Pattern (G/R)", value="RRGG")
    
    try:
        df_daily_raw = fetch_legacy_market_data("Bitcoin (BTC)", "1d")
        if df_daily_raw is None or df_daily_raw.empty:
            st.error("❌ High-density historical data streams unreachable.")
            st.stop()
            
        df_daily = df_daily_raw.copy()
        df_daily['return'] = ((df_daily['close'] - df_daily['open']) / df_daily['open']) * 100.0
        try:
            df_daily['is_midterm'] = (df_daily.index.year % 4 == 2)
        except:
            df_daily['is_midterm'] = False

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
        
        current_live_gap = days_since_tracker[-1]
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(df_daily.index, days_since_tracker, color='#00ffcc', linewidth=1.2)
        ax.fill_between(df_daily.index, 0, days_since_tracker, color='#00ffcc', alpha=0.06)
        ax.set_title("CHART: STRUCTURAL RESETS TIMELINE WAVE (PATH DEPENDENT)", fontweight='bold')
        st.pyplot(fig)
        
        st.metric("Current Days Stretched Under Matrix Threshold", f"{current_live_gap} Days")
    except Exception as ex:
        st.error(f"Execution Error: {ex}")

st.divider()
st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
