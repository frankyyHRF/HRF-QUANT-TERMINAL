# --- STEP 1: CONFIGURE SYSTEM WORKSPACE DEPENDENCIES ---
import sys
import subprocess

print("📥 Checking package dependencies... installing structural calculation modules...")
dependencies = ["fastdtw", "pandas", "numpy", "matplotlib", "ipywidgets"]
for package in dependencies:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Specific GitHub install rule for updated TradingView data engine stream connections
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "git+https://github.com/rongardF/tvDatafeed.git"])
print("✅ Environment compiled successfully. Initializing platform modules...")

# --- STEP 2: LOAD MASTER ENGINE PLATFORM ---
import ipywidgets as widgets
from ipywidgets import interact_manual
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from fastdtw import fastdtw

# Initialize public node connection
tv = TvDatafeed()

def percentage_return_scale(series):
    """Transforms a price series into percentage returns relative to its starting element."""
    arr = np.array(series, dtype=float).flatten()
    if len(arr) == 0 or arr[0] == 0:
        return np.zeros_like(arr)
    return ((arr - arr[0]) / arr[0]) * 100.0

def get_election_regime(year):
    """Categorizes any historical year into its exact US Election Cycle branch."""
    if year % 4 == 0:
        return 'Election Year'
    elif year % 4 == 1:
        return 'Post-Election'
    elif year % 4 == 2:
        return 'Midterm Year'
    else:
        return 'Pre-Election'

# Asset Data Dictionary Lookups
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

def run_hrf_dashboard(Target_Asset, Scan_Assets, Interval_Choice, Mode, Sort_Priority, Min_Correlation, Max_MSE, Start_Date, End_Date, Overlay_Starts, Overlay_Ends, Target_Bars, Forecast_Bars, Number_Of_Fractals, Isolate_Path, Cycle_Filter, Specific_Years, Exclude_Years, Gap_Bars, Vol_Boost, Std_Dev_Bands):
    """
    HRF QUANT PLATFORM — MASTER CORE MATRIX V4
    Includes Multi-Asset Selection Lists, Priority Engine Filters, Threshold Cutoffs, and Core Stacking.
    """
    try:
        target_bars_num = int(Target_Bars)
        forecast_bars_num = int(Forecast_Bars)
        num_fractals_num = int(Number_Of_Fractals)
        gap_bars_num = int(Gap_Bars)
        vol_boost_pct = float(Vol_Boost)
        std_dev_multiplier = float(Std_Dev_Bands)
        min_corr_val = float(Min_Correlation)
        max_mse_val = float(Max_MSE)
    except ValueError:
        print("⚠️ Parameter Configuration Error: Clean your numeric variables.")
        return

    interval_map = {
        '1M': Interval.in_monthly, '1w': Interval.in_weekly, '1d': Interval.in_daily, 
        '1h': Interval.in_1_hour, '15m': Interval.in_15_minute, '5m': Interval.in_5_minute
    }
    chosen_interval = interval_map.get(Interval_Choice, Interval.in_daily)
    max_bars = 16000 if Interval_Choice in ['1d', '1w', '1M'] else 4900

    print(f"\n🔄 Syncing structural source data line for target asset {Target_Asset}...")
    sym, exch = ticker_map[Target_Asset]
    try:
        df_target = tv.get_hist(symbol=sym, exchange=exch, interval=chosen_interval, n_bars=max_bars)
    except:
        print("❌ Network Connection Error: Could not stream dataset pipeline.")
        return
        
    if df_target is None or df_target.empty:
        print("❌ Data Query Empty.")
        return

    df_target.index = pd.to_datetime(df_target.index).tz_localize(None)
    close_target = df_target['close'].dropna()

    start_clean = Start_Date.strip().lower()
    end_clean = End_Date.strip().lower()
    
    if start_clean == 'latest' or end_clean == 'latest':
        target_df = close_target.iloc[-target_bars_num:]
    else:
        target_df = close_target.loc[pd.to_datetime(start_clean):pd.to_datetime(end_clean)]
        if len(target_df) == 0:
            print("❌ Selection Range Exception: Target range contains 0 ticks.")
            return
        target_bars_num = len(target_df)

    target_scaled = percentage_return_scale(target_df.tolist())

    plt.style.use('dark_background')
    plt.figure(figsize=(16, 9))
    plt.plot(target_scaled, color='#00ffcc', linewidth=4, label=f'TARGET: {Target_Asset} (Baseline)', zorder=5)

    # --- SANDBOX MODE CORE RUNTIME ---
    if Mode == "Manual Compare":
        print("⚡ [SANDBOX MODE]: Extracting and Stacking Custom History Overlays...")
        starts = [s.strip() for s in Overlay_Starts.split(',') if s.strip()]
        ends = [e.strip() for e in Overlay_Ends.split(',') if e.strip()]
        
        if len(starts) != len(ends):
            print("❌ Matrix Misalignment: Start entries must match total end entries.")
            return
            
        manual_paths_list = []
        cmap = plt.colormaps.get_cmap('spring')
        color_steps = np.linspace(0, 0.7, max(len(starts), 2))

        for idx, (st, ed) in enumerate(zip(starts, ends)):
            try:
                ov_df = close_target.loc[pd.to_datetime(st):pd.to_datetime(ed)]
                if len(ov_df) == 0: continue
                ov_scaled = percentage_return_scale(ov_df.tolist())
                manual_paths_list.append(ov_scaled)
                
                isolate_clean = Isolate_Path.strip().lower()
                if isolate_clean == 'all' or (isolate_clean.isdigit() and int(isolate_clean) == idx + 1):
                    plt.plot(ov_scaled, color=cmap(color_steps[idx]), linewidth=2, linestyle='--', alpha=0.6, label=f'Manual #{idx+1} [{st}]')
            except:
                continue

        if manual_paths_list and Isolate_Path.strip().lower() in ['all', 'mean']:
            max_len = max(len(p) for p in manual_paths_list)
            padded_list = [np.pad(p, (0, max_len - len(p)), 'edge') if len(p) < max_len else p for p in manual_paths_list]
            manual_matrix = np.vstack(padded_list)
            m_mean = np.mean(manual_matrix, axis=0) * (1.0 + (vol_boost_pct / 100.0))
            m_std = np.std(manual_matrix, axis=0) * (1.0 + (vol_boost_pct / 100.0))
            
            if std_dev_multiplier > 0:
                plt.fill_between(range(len(m_mean)), m_mean - (m_std * std_dev_multiplier), m_mean + (m_std * std_dev_multiplier), color='#ffff00', alpha=0.12, label='Manual Stacking Deviation Channel')
            plt.plot(m_mean, color='#ffff00', linewidth=4, label='MANUAL COMPOSITE MEAN TRACK', zorder=6)

    # --- CROSS-ASSET MATH ENGINE SCANNERS ---
    else:
        if not Scan_Assets:
            print("⚠️ User Matrix Alert: Please select at least one historical database tracker from the multi-asset list.")
            return
            
        print(f"🎯 [ALGO SCAN]: Scanning across your selected assets: {list(Scan_Assets)}...")
        all_discovered_results = []
        
        include_years = [int(y.strip()) for y in Specific_Years.split(',') if y.strip() and y.lower() != 'all']
        exclude_years = [int(y.strip()) for y in Exclude_Years.split(',') if y.strip() and y.lower() != 'none']

        for asset_item in Scan_Assets:
            s_sym, s_exch = ticker_map[asset_item]
            try:
                df_scan = tv.get_hist(symbol=s_sym, exchange=s_exch, interval=chosen_interval, n_bars=max_bars)
                if df_scan is None or df_scan.empty: continue
                df_scan.index = pd.to_datetime(df_scan.index).tz_localize(None)
                close_scan = df_scan['close'].dropna()
            except:
                continue

            if asset_item == Target_Asset and (start_clean == 'latest' or end_clean == 'latest'):
                history_pool = close_scan.iloc[:-target_bars_num].tolist()
                history_dates = close_scan.index[:-target_bars_num]
            else:
                history_pool = close_scan.tolist()
                history_dates = close_scan.index
                
            max_search_index = len(history_pool) - (target_bars_num + forecast_bars_num)
            if max_search_index <= 0: continue

            for i in range(max_search_index):
                hist_year = history_dates[i].year
                hist_regime = get_election_regime(hist_year)
                
                if Cycle_Filter != 'All Cycles' and hist_regime != Cycle_Filter: continue
                if include_years and hist_year not in include_years: continue
                if hist_year in exclude_years or hist_year == 2026: continue
                
                hist_pattern = history_pool[i : i + target_bars_num]
                hist_scaled = percentage_return_scale(hist_pattern)
                
                # Metric calculation nodes
                mse = float(np.mean((target_scaled - hist_scaled) ** 2))
                corr_matrix = np.corrcoef(target_scaled, hist_scaled)
                corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else -1.0
                
                # Filter validation check boundaries
                if corr < min_corr_val or mse > max_mse_val:
                    continue
                    
                dtw_dist, _ = fastdtw(target_scaled, hist_scaled, radius=2, dist=2)
                
                all_discovered_results.append({
                    'asset_name': asset_item, 'start_index': i, 'end_index': i + target_bars_num,
                    'mse': mse, 'correlation': corr, 'dtw': float(dtw_dist), 'year': hist_year,
                    'raw_prices': history_pool[i : i + target_bars_num + forecast_bars_num],
                    'date_str': history_dates[i].strftime('%Y-%m-%d')
                })

        if not all_discovered_results:
            print("❌ Zero matches satisfied the requested hard thresholds.")
            return
            
        raw_match_df = pd.DataFrame(all_discovered_results)
        
        # Calculate scores for sorting configurations
        max_mse = raw_match_df['mse'].max() if raw_match_df['mse'].max() != 0 else 1
        max_dtw = raw_match_df['dtw'].max() if raw_match_df['dtw'].max() != 0 else 1
        raw_match_df['omni_score'] = (
            (raw_match_df['correlation'] + 1) / 2 * 0.40 + 
            (1 - (raw_match_df['mse'] / max_mse)) * 0.30 + 
            (1 - (raw_match_df['dtw'] / max_dtw)) * 0.30
        )
        
        # --- FLEXIBLE PRIORITY SORT ENGINE ---
        if Sort_Priority == 'Highest Correlation':
            sorted_match_df = raw_match_df.sort_values(by='correlation', ascending=False)
        elif Sort_Priority == 'Lowest MSE':
            sorted_match_df = raw_match_df.sort_values(by='mse', ascending=True)
        else: # Omni-Score Default Option Block
            sorted_match_df = raw_match_df.sort_values(by='omni_score', ascending=False)
        
        # Cluster distance mapping loop
        unique_matches = []
        seen_clusters = set()
        for _, row in sorted_match_df.iterrows():
            s, e, a = row['start_index'], row['end_index'], row['asset_name']
            if not any(max(s, es - gap_bars_num) < min(e, ee + gap_bars_num) for es, ee, aa in seen_clusters if aa == a):
                unique_matches.append(row)
                seen_clusters.add((s, e, a))
            if len(unique_matches) >= num_fractals_num:
                break

        # Generate structural rolling statistics maps
        all_scaled_paths_list = []
        for row in unique_matches:
            raw_full = row['raw_prices']
            base_p = raw_full[0]
            scaled_full = ((np.array(raw_full) - base_p) / base_p) * 100.0 if base_p != 0 else np.zeros_like(raw_full)
            all_scaled_paths_list.append(scaled_full)

        mean_path_array = None
        if all_scaled_paths_list:
            matrix_data = np.vstack(all_scaled_paths_list)
            mean_path_array = np.mean(matrix_data, axis=0) * (1.0 + (vol_boost_pct / 100.0))
            if std_dev_multiplier > 0:
                std_path = np.std(matrix_data, axis=0) * (1.0 + (vol_boost_pct / 100.0))
                plt.fill_between(range(len(mean_path_array)), mean_path_array - (std_path * std_dev_multiplier), mean_path_array + (std_path * std_dev_multiplier), color='#ffff00', alpha=0.12, label='Cross-Asset Variance Cloud')

        # Visualizer core presentation pipeline
        isolate_clean = Isolate_Path.strip().lower()
        cmap = plt.colormaps.get_cmap('plasma')
        color_steps = np.linspace(0, 0.8, max(len(unique_matches), 2))
        
        if isolate_clean != 'mean':
            for idx, row in enumerate(unique_matches):
                if isolate_clean.isdigit() and int(isolate_clean) != idx + 1: continue
                raw_full = row['raw_prices']
                base_p = raw_full[0]
                scaled_full = ((np.array(raw_full) - base_p) / base_p) * 100.0
                lbl = f"#{idx+1} {row['asset_name']} ({row['date_str']}) [Corr: {row['correlation']:.2f} | MSE: {row['mse']:.4f}]"
                plt.plot(scaled_full, color=cmap(color_steps[idx % len(color_steps)]), linestyle='--', alpha=0.6, label=lbl)

        if mean_path_array is not None and isolate_clean in ['all', 'mean']:
            plt.plot(mean_path_array, color='#ffff00', linewidth=4, label='COMPOSITE FRACTAL MEAN BASELINE', zorder=6)
            
        plt.axvline(x=target_bars_num - 1, color='#ffffff', linestyle=':', alpha=0.5, linewidth=1.5)

    # UI Presentation Parameters
    plt.axhline(y=0.0, color='#555555', linestyle='-', linewidth=1.2)
    plt.title(f"HRF QUANT PLATFORM — MASTER TERMINAL V4 ENGINE", color='#ffffff', fontsize=13, fontweight='bold', pad=15)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}%'))
    plt.ylabel("Normalized Return Performance Scale", color='#ffffff', fontsize=11)
    plt.xlabel("Timeline Index Sequence (Bars)", color='#ffffff', fontsize=11)
    plt.grid(True, color='#222222', linestyle='-')
    plt.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')
    plt.tight_layout()
    plt.show()
    print("\n--- Model By HRF ---")

# --- CUSTOMIZED USER FORM LAYOUT BUILDERS ---
assets_list = list(ticker_map.keys())

interact_manual(
    run_hrf_dashboard,
    Target_Asset=widgets.Dropdown(options=assets_list, value='Bitcoin (BTC)', description='Target Asset:'),
    Scan_Assets=widgets.SelectMultiple(options=assets_list, value=['Bitcoin (BTC)'], description='Scan Pools:', rows=6),
    Interval_Choice=['1M', '1w', '1d', '1h', '15m', '5m'],
    Mode=['Calculate Fractals', 'Manual Compare'],
    Sort_Priority=['Highest Correlation', 'Lowest MSE', 'Omni-Score Combination'],
    Min_Correlation=widgets.FloatSlider(value=-1.0, min=-1.0, max=1.0, step=0.05, description='Min Corr:'),
    Max_MSE=widgets.FloatSlider(value=1000.0, min=0.1, max=1000.0, step=10.0, description='Max MSE:'),
    Start_Date=widgets.Text(value='latest', description='Start Date:'),
    End_Date=widgets.Text(value='latest', description='End Date:'),
    Overlay_Starts=widgets.Text(value='2020-03-01, 2022-11-01', description='Overlay Starts:'),
    Overlay_Ends=widgets.Text(value='2020-05-01, 2023-01-01', description='Overlay Ends:'),
    Target_Bars=widgets.Text(value='30', description='Scan Bars:'),
    Forecast_Bars=widgets.Text(value='15', description='Forecast Bars:'),
    Number_Of_Fractals=widgets.Text(value='5', description='Num Paths:'),
    Isolate_Path=widgets.Text(value='all', description='Isolate (#):'),
    Cycle_Filter=['All Cycles', 'Election Year', 'Post-Election', 'Midterm Year', 'Pre-Election'],
    Specific_Years=widgets.Text(value='all', description='Years Only:'),
    Exclude_Years=widgets.Text(value='none', description='Exclude Years:'),
    Gap_Bars=widgets.Text(value='20', description='Gap Bars:'),
    Vol_Boost=widgets.Text(value='0', description='Vol Boost (%):'),
    Std_Dev_Bands=widgets.Text(value='1.0', description='Std Dev (+-):')
);
