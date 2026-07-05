import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval

# --- APP CONFIGURATION & STYLING ---
st.set_page_config(page_title="HRF QUANT PLATFORM", layout="wide")

# Inject deep dark-mode styling variables for mobile UI clarity
st.markdown("""
    <style>
    .reportview-container { background: #0d0d11; }
    .sidebar .sidebar-content { background: #13131a; }
    h1, h2, h3, p { color: #ffffff !important; }
    div.stButton > button:first-child {
        background-color: #8a2be2; color: white; border-radius: 8px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ HRF QUANT TERMINAL — V6 PLATFORM")
st.caption("Professional Multi-Regime Micro-Structure Analysis Engine | Bitcoin Data Sourced via BNC Core Nodes")
st.hr()

# --- SIDEBAR CONTROL UNIT ---
st.sidebar.header("🎛️ Terminal Parameters")

vol_input = st.sidebar.text_input("Volatility Reset Threshold (%)", value="-40.0")
pattern_input = st.sidebar.text_input("Candle Sequence Pattern (G/R)", value="RRGG")

st.sidebar.info("💡 Pro-Tip: For negative thresholds (like -40), the engine tracks peak-to-trough drawdowns from wicks. For positive thresholds (like 50), it tracks expansion from local low troughs.")

# --- MEMORY-CACHED STREAMLIT DATA LIFELINE ---
@st.cache_data(show_spinner="🔄 Establishing live matrix streaming from TradingView nodes...")
def fetch_and_cache_market_history():
    tv = TvDatafeed()
    # Fetch maximum chronological arrays across distinct timeframe intervals
    df_d = tv.get_hist(symbol="BLX", exchange="BNC", interval=Interval.in_daily, n_bars=18000)
    df_d.index = pd.to_datetime(df_d.index).tz_localize(None)
    
    df_w = tv.get_hist(symbol="BLX", exchange="BNC", interval=Interval.in_weekly, n_bars=2000)
    df_w.index = pd.to_datetime(df_w.index).tz_localize(None)
    
    df_h = tv.get_hist(symbol="BTCUSDT", exchange="BINANCE", interval=Interval.in_1_hour, n_bars=15000)
    df_h.index = pd.to_datetime(df_h.index).tz_localize(None)
    
    # Pre-calculate intraday session velocity matrices
    for df in [df_d, df_w, df_h]:
        df['return'] = ((df['close'] - df['open']) / df['open']) * 100.0
        df['is_midterm'] = (df.index.year % 4 == 2)
        
    return df_d, df_w, df_h

# Execute secure pull or reference memory cache node instantly
try:
    df_daily_raw, df_weekly, df_hourly = fetch_and_cache_market_history()
    df_daily = df_daily_raw.copy()
    data_loaded = True
except Exception as e:
    st.error(f"Failed to stream market data archives: {e}")
    data_loaded = False

if data_loaded:
    # --- CORE MATH CALCULATIONS UNIT ---
    try:
        thresh = float(vol_input)
    except ValueError:
        st.error("⚠️ Setup Parameter Error: Volatility input must be a valid numeric calculation factor.")
        st.stop()

    # 1. Path-Dependent Accumulator Engine Logic
    days_since_tracker = []
    historical_gaps = []
    current_accumulator = 0

    if thresh < 0:
        running_peak = df_daily['high'].iloc[0]
        for idx in range(len(df_daily)):
            current_high = df_daily['high'].iloc[idx]
            current_low = df_daily['low'].iloc[idx]
            if current_high > running_peak:
                running_peak = current_high
            drawdown_pct = ((current_low - running_peak) / running_peak) * 100.0
            if drawdown_pct <= thresh:
                days_since_tracker.append(0)
                if current_accumulator > 0:
                    historical_gaps.append(current_accumulator)
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
            if current_low < running_trough:
                running_trough = current_low
            expansion_pct = ((current_high - running_trough) / running_trough) * 100.0
            if expansion_pct >= thresh:
                days_since_tracker.append(0)
                if current_accumulator > 0:
                    historical_gaps.append(current_accumulator)
                current_accumulator = 0
                running_trough = current_low
            else:
                current_accumulator += 1
                days_since_tracker.append(current_accumulator)

    current_live_gap = days_since_tracker[-1]

    # 2. Combinatorial N-Gram Candle Sequence Analytics
    clean_pattern = pattern_input.strip().upper()
    candle_string_sequence = "".join(['G' if r >= 0 else 'R' for r in df_daily['return'].tolist()])
    match_indices = []
    pattern_length = len(clean_pattern)

    for i in range(len(candle_string_sequence) - pattern_length):
        if candle_string_sequence[i : i + pattern_length] == clean_pattern:
            match_indices.append(i + pattern_length)

    total_sequence_matches = len(match_indices)
    forward_green_count = 0
    post_pattern_returns = []

    if total_sequence_matches > 0:
        for m_idx in match_indices:
            if m_idx < len(df_daily):
                next_return = df_daily['return'].iloc[m_idx]
                post_pattern_returns.append(next_return)
                if next_return >= 0:
                    forward_green_count += 1
        forward_probability_green = (forward_green_count / total_sequence_matches) * 100.0
        forward_probability_red = 100.0 - forward_probability_green
    else:
        forward_probability_green = forward_probability_red = 0.0

    # --- VISUAL RENDERING CANVAS ---
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    all_daily_returns = df_daily['return'].dropna().values
    midterm_daily_returns = df_daily[df_daily['is_midterm'] == True]['return'].dropna().values

    # Chart 1 & 2: Return Distributions
    ax1.hist(all_daily_returns, bins=100, range=(-15, 15), color='#8a2be2', alpha=0.6, edgecolor='#111111', label='Daily Lifetime (2010+)')
    ax1.set_title("HISTOGRAM 1: TOTAL DAILY OPEN-TO-CLOSE RETURN FREQUENCY", fontweight='bold', fontsize=10)
    ax1.grid(True, color='#222222', alpha=0.5); ax1.legend(facecolor='#111111', edgecolor='#333333')

    ax2.hist(midterm_daily_returns, bins=100, range=(-15, 15), color='#ff0055', alpha=0.6, edgecolor='#111111', label='Midterm Regime Only')
    ax2.set_title("HISTOGRAM 2: MIDTERM YEAR CYCLE OPEN-TO-CLOSE FREQUENCY", fontweight='bold', fontsize=10)
    ax2.grid(True, color='#222222', alpha=0.5); ax2.legend(facecolor='#111111', edgecolor='#333333')

    # Chart 3: Chronological Sawtooth Wave Timeframe (2010-2026)
    ax3.plot(df_daily.index, days_since_tracker, color='#00ffcc', linewidth=1.2, label=f'Days Elapsed to Reach {thresh}%')
    ax3.fill_between(df_daily.index, 0, days_since_tracker, color='#00ffcc', alpha=0.06)
    ax3.set_title("CHART 3: STRUCTURAL RESETS CHRONOLOGICAL TIMELINE WAVE", fontweight='bold', fontsize=10)
    ax3.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2026-07-01'))
    ax3.grid(True, color='#222222', alpha=0.5); ax3.legend(loc='upper left', facecolor='#111111', edgecolor='#333333')

    # Chart 4: Forward Edge Distribution Outcome
    if post_pattern_returns:
        ax4.hist(post_pattern_returns, bins=30, range=(-15, 15), color='#ffff00', alpha=0.6, edgecolor='#111111', label=f"Day After '{clean_pattern}'")
    ax4.set_title(f"HISTOGRAM 4: FORWARD EDGE PERFORMANCE POST-{clean_pattern} SEQUENCE", fontweight='bold', fontsize=10)
    ax4.grid(True, color='#222222', alpha=0.5); ax4.legend(facecolor='#111111', edgecolor='#333333')

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)

    # --- DATA OUTPUT INTERFACE TABLES ---
    st.subheader("📊 Session Change Velocity Matrix")
    metrics_data = {
        "Timeframe Segment": ["Daily Open-to-Close changes", "Weekly Open-to-Close changes", "Hourly Open-to-Close changes"],
        "Lifetime History Average (2010+)": [f"{df_daily['return'].mean():+.4f}%", f"{df_weekly['return'].mean():+.4f}%", f"{df_hourly['return'].mean():+.4f}%"],
        "US Midterm Election Cycles Only": [f"{df_daily[df_daily['is_midterm']]['return'].mean():+.4f}%", f"{df_weekly[df_weekly['is_midterm']]['return'].mean():+.4f}%", f"{df_hourly[df_hourly['is_midterm']]['return'].mean():+.4f}%"]
    }
    st.table(pd.DataFrame(metrics_data))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⏱️ Path-Dependent Chronological Metrics")
        st.metric(label=f"Current Continuous Gap Since Last {thresh}% Move", value=f"{current_live_gap} Days")
        if historical_gaps:
            st.write(f"📈 **Historical Average Rest Interval:** {np.mean(historical_gaps):.1f} days")
            st.write(f"📊 **Historical Median Rest Interval:** {np.median(historical_gaps):.1f} days")
            st.write(f"⚠️ **Maximum Historical Stress Stretch:** {np.max(historical_gaps)} days")

    with col2:
        st.subheader(f"🔮 Forward Edge Sequence Probabilities: '{clean_pattern}'")
        st.metric(label="Total Historical Matches Found", value=f"{total_sequence_matches}")
        if total_sequence_matches > 0:
            st.write(f"🟢 **Probability of Next Candle printing GREEN (G):** {forward_probability_green:.2f}%")
            st.write(f"🔴 **Probability of Next Candle printing RED (R):** {forward_probability_red:.2f}%")
            st.write(f"🎯 **Expected Next Candle Mean Session Return:** {np.mean(post_pattern_returns):+.3f}%")

    st.hr()
    st.markdown("<p style='text-align: center; color: #555555;'>--- Model By HRF ---</p>", unsafe_allow_html=True)
