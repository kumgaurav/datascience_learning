import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Optional legacy model imports (handled independently from chatbot)
try:
    from stock_selector_v2 import get_top_stocks, get_stock_analysis  # legacy (optional)
    from stock_selector import calculate_confidence_score, calculate_risk_score  # legacy (optional)
except Exception:
    def get_top_stocks(*args, **kwargs):
        return pd.DataFrame()

    def get_stock_analysis(*args, **kwargs):
        return {}

    def calculate_confidence_score(*args, **kwargs):
        return None

    def calculate_risk_score(*args, **kwargs):
        return None

# Chatbot import handled separately so it's available even if legacy modules are missing
try:
    from chatbot import create_chatbot_agent
except Exception:
    def create_chatbot_agent(*args, **kwargs):
        return None
import os
import joblib
from dotenv import load_dotenv
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Load environment variables from .env file
load_dotenv()

# Project root (one level above uidev/), used for robust relative file resolution
PRJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def create_stock_price_chart(ticker, stock_data, featured_data):
    """
    Create a 3-week price chart with resistance break and momentum highlights.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (pd.DataFrame): Historical price data for the stock
        featured_data (pd.Series): Featured data containing technical indicators
    
    Returns:
        plotly.graph_objects.Figure: Interactive price chart
    """
    try:
        # Filter data for the specific ticker
        ticker_data = stock_data[stock_data['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            return None
        
        # Convert date column to datetime
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        
        # Sort by date and get last 3 weeks (21 trading days)
        ticker_data = ticker_data.sort_values('date')
        last_date = ticker_data['date'].max()
        start_date = last_date - timedelta(days=30)  # 30 days to ensure we get 21 trading days
        
        # Filter for last 3 weeks
        recent_data = ticker_data[ticker_data['date'] >= start_date].copy()
        
        if len(recent_data) < 5:  # Need at least 5 days for meaningful chart
            return None
        
        # Get technical indicators from featured data
        resistance_level = featured_data.get('resistance_20d', None)
        support_level = featured_data.get('support_20d', None)
        broke_resistance = featured_data.get('broke_resistance', False)
        momentum_5d = featured_data.get('momentum_5d', 0)
        momentum_10d = featured_data.get('momentum_10d', 0)
        rsi_14d = featured_data.get('rsi_14d', 50)
        
        # Create the main price chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=recent_data['date'],
            open=recent_data['open'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['close'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ))
        
        # Add resistance line if available
        if resistance_level is not None and not pd.isna(resistance_level):
            fig.add_hline(
                y=resistance_level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance: ${resistance_level:.2f}",
                annotation_position="top right"
            )
            
            # Highlight resistance break if it occurred
            if broke_resistance:
                # Find the first day price closed above resistance
                break_data = recent_data[recent_data['close'] > resistance_level]
                if not break_data.empty:
                    break_date = break_data.iloc[0]['date']
                    break_price = break_data.iloc[0]['close']
                    
                    fig.add_annotation(
                        x=break_date,
                        y=break_price,
                        text=f"🟢 Resistance Break<br>${break_price:.2f}<br>Resistance: ${resistance_level:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="green",
                        bgcolor="green",
                        bordercolor="white",
                        borderwidth=2
                    )
        
        # Add support line if available
        if support_level is not None and not pd.isna(support_level):
            fig.add_hline(
                y=support_level,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support: ${support_level:.2f}",
                annotation_position="bottom right"
            )
        
        # Add momentum indicators
        if momentum_5d > 0 and momentum_10d > 0:
            # Find the day with strongest momentum
            momentum_data = recent_data.tail(5)  # Last 5 days
            if not momentum_data.empty:
                max_momentum_idx = momentum_data['close'].pct_change().idxmax()
                if not pd.isna(max_momentum_idx):
                    momentum_date = recent_data.loc[max_momentum_idx, 'date']
                    momentum_price = recent_data.loc[max_momentum_idx, 'close']
                    
                    fig.add_annotation(
                        x=momentum_date,
                        y=momentum_price,
                        text=f"🚀 Momentum<br>${momentum_price:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="orange",
                        bgcolor="orange",
                        bordercolor="white",
                        borderwidth=2
                    )
        
        # Add RSI indicator
        if rsi_14d is not None and not pd.isna(rsi_14d):
            rsi_color = "green" if rsi_14d > 70 else "red" if rsi_14d < 30 else "gray"
            rsi_text = f"RSI: {rsi_14d:.1f}"
            
            # Add RSI annotation on the last day
            last_date = recent_data['date'].iloc[-1]
            last_price = recent_data['close'].iloc[-1]
            
            fig.add_annotation(
                x=last_date,
                y=last_price,
                text=rsi_text,
                showarrow=False,
                bgcolor=rsi_color,
                bordercolor="white",
                borderwidth=1,
                xanchor="left",
                yanchor="bottom"
            )
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - 3-Week Price Chart with Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Add volume as subplot
        fig.add_trace(go.Bar(
            x=recent_data['date'],
            y=recent_data['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Update layout to include volume subplot
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart for {ticker}: {str(e)}")
        return None

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Configuration ---
st.sidebar.title("🎯 Stock Screener Settings")

# Filter controls
st.sidebar.header("📊 Filter Settings")
min_confidence = st.sidebar.slider("Minimum Confidence Score", 0, 100, 30, help="Higher = more confident predictions")
max_risk = st.sidebar.slider("Maximum Risk Score", 0, 100, 70, help="Lower = less risky stocks")
diversify = st.sidebar.checkbox("Apply Portfolio Diversification", True, help="Limit exposure per sector/industry")
# Default to 25 stocks
num_stocks = st.sidebar.selectbox("Number of Stocks", [10, 15, 20, 25, 30], index=3)

# API Key Status
st.sidebar.header("🔑 API Configuration")
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    st.sidebar.success("✅ Google API Key Found")
    st.sidebar.info(f"Key: {api_key[:10]}...")
else:
    st.sidebar.error("❌ Google API Key Missing")
    st.sidebar.markdown("""
    **To enable AI chatbot:**
    1. Create a `.env` file in the project root
    2. Add: `GOOGLE_API_KEY=your_actual_api_key`
    3. Restart the app
    """)

# (Momentum Winners sidebar removed; now available as its own page under left navigation)

# --- App Title ---
st.title("📈 AI-Powered Stock Screener Pro")
st.write("Advanced stock analysis using XGBoost, technical indicators, and fundamental data.")

# --- Main Logic ---
# Use session state to store data and avoid re-running analysis on every interaction
if 'top_stocks_df' not in st.session_state:
    st.session_state.top_stocks_df = pd.DataFrame()
if 'agent' not in st.session_state:
    st.session_state.agent = None

# "Run Analysis" button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Find Top Stocks for This Week", type="primary", use_container_width=True):
        with st.spinner("Loading ensemble results..."):
            # Always load from ensemble_weekly_output.csv for consistency and speed
            _ens = pd.DataFrame()
            try:
                try:
                    from uidev.data_loader import load_ensemble_weekly, default_sort
                except ModuleNotFoundError:
                    import os as _os, sys as _sys
                    _sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
                    from uidev.data_loader import load_ensemble_weekly, default_sort
                _ens = load_ensemble_weekly()
            except Exception:
                _ens = pd.DataFrame()

            if _ens is not None and not _ens.empty:
                # Apply confidence filter as ANY-model confidence >= threshold
                _conf_candidates = [
                    'xgb_confidence_score', 'lstm_confidence_score',
                    'confidence_score_xgb', 'confidence_score_lstm', 'confidence_score'
                ]
                _present = [c for c in _conf_candidates if c in _ens.columns]
                if _present:
                    _conf_any = pd.concat([
                        pd.to_numeric(_ens[c], errors='coerce').fillna(0) for c in _present
                    ], axis=1).max(axis=1)
                    _ens = _ens[_conf_any >= min_confidence]
                # Sort and clip to desired count
                try:
                    _ens = default_sort(_ens)
                except Exception:
                    # Basic fallback if sorter fails
                    _ens = _ens.sort_values(by=[c for c in _ens.columns if c != 'ticker'][0], ascending=False)
                _ens = _ens.head(num_stocks)
                # Ensure a unified predicted_return_pct column for downstream charts
                try:
                    _dfu = _ens.copy()
                    # Candidate prediction columns (in preferred order)
                    _pred_cols = [
                        'predicted_return_pct',
                        'xgb_predicted_return_pct', 'lstm_predicted_return_pct',
                        'xgb_pred', 'lstm_pred'
                    ]
                    _present = [c for c in _pred_cols if c in _dfu.columns]
                    if _present:
                        import numpy as _np
                        _vals = []
                        for c in _present:
                            _vals.append(pd.to_numeric(_dfu[c], errors='coerce'))
                        _stack = _np.vstack([s.fillna(_np.nan).to_numpy() for s in _vals])
                        _pred = _np.nanmean(_stack, axis=0)
                        # If values look like fractions (< 1 in magnitude), convert to %
                        with _np.errstate(invalid='ignore'):
                            _med = _np.nanmedian(_np.abs(_pred))
                        if _med is not _med or _med <= 1:
                            _pred = _pred * 100.0
                        _dfu['predicted_return_pct'] = _pred
                        _ens = _dfu
                except Exception:
                    pass
                st.session_state.top_stocks_df = _ens.copy()
                try:
                    st.caption('Using ensemble_weekly_output.csv')
                except Exception:
                    pass
            else:
                st.session_state.top_stocks_df = pd.DataFrame()
            
            # Debug information
            st.info(f"📊 Found {len(st.session_state.top_stocks_df)} stocks matching your criteria")
            if not st.session_state.top_stocks_df.empty:
                # Top stock summary (support multiple prediction columns)
                _row = st.session_state.top_stocks_df.iloc[0]
                _pred_cols = ['predicted_return_pct', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct']
                _pred_val = None
                for _c in _pred_cols:
                    if _c in st.session_state.top_stocks_df.columns:
                        try:
                            import re
                            _pred_val = float(re.sub(r"[^0-9.+-]", "", str(_row.get(_c, ''))))
                            break
                        except Exception:
                            continue
                if _pred_val is None or _pred_val != _pred_val:
                    _pred_txt = "n/a"
                else:
                    _pred_txt = f"{_pred_val:.2f}%"
                st.write(f"Top stock: {_row.get('ticker','')} with {_pred_txt} predicted return")
            
            # After getting stocks, create the chatbot agent if data is available
            if not st.session_state.top_stocks_df.empty:
                try:
                    st.session_state.agent = create_chatbot_agent(st.session_state.top_stocks_df)
                    if hasattr(st.session_state.agent, 'invoke'):
                        st.success("✅ Analysis complete! AI Chatbot initialized successfully!")
                    else:
                        st.warning("⚠️ Analysis complete! Using simplified analysis mode (no AI).")
                except Exception as e:
                    st.error(f"❌ Failed to initialize chatbot: {str(e)}")
                    st.session_state.agent = None

# --- Display Results ---
if not st.session_state.top_stocks_df.empty:

    
    # --- Summary Metrics ---
    st.header("📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(st.session_state.top_stocks_df))
    with col2:
        # Metrics summary with fallbacks
        def _series_num(df, cols):
            import re
            for c in cols:
                if c in df.columns:
                    return pd.to_numeric(df[c].astype(str).str.replace(r'[^0-9.+-]+','', regex=True), errors='coerce')
            return pd.Series([], dtype=float)
        avg_return = _series_num(st.session_state.top_stocks_df, ['predicted_return_pct','xgb_predicted_return_pct','lstm_predicted_return_pct','lstm_pred']).mean()
        st.metric("Avg Predicted Return", f"{0.0 if pd.isna(avg_return) else avg_return:.2f}%")
    with col3:
        # Prefer ensemble score
        ens_series = pd.to_numeric(st.session_state.top_stocks_df.get('ensemble_score'), errors='coerce') if 'ensemble_score' in st.session_state.top_stocks_df.columns else pd.Series([], dtype=float)
        avg_ens = float(ens_series.mean()) if len(ens_series) else float('nan')
        st.metric("Avg Ensemble Score", f"{0.0 if pd.isna(avg_ens) else avg_ens:.3f}")
    with col4:
        # Risk fallback
        risk_cols = [c for c in ['risk_score','lstm_risk_score'] if c in st.session_state.top_stocks_df.columns]
        if risk_cols:
            r = pd.to_numeric(st.session_state.top_stocks_df[risk_cols[0]], errors='coerce')
            avg_risk = float(r.mean())
        else:
            avg_risk = float('nan')
        st.metric("Avg Risk Score", f"{0.0 if pd.isna(avg_risk) else avg_risk:.1f}/100")
    
    # --- Top Pick Highlight ---
    st.header("🏆 Top Pick")
    best_stock = st.session_state.top_stocks_df.iloc[0]
    # Safe predicted return text
    import re
    def _fmt_ret(val):
        try:
            v = float(re.sub(r'[^0-9.+-]', '', str(val)))
            return f"{v:+.2f}%"
        except Exception:
            return str(val)
    pr_txt = None
    for c in ['predicted_return_pct','xgb_predicted_return_pct','lstm_predicted_return_pct']:
        if c in st.session_state.top_stocks_df.columns:
            pr_txt = _fmt_ret(best_stock.get(c))
            break
    pr_txt = pr_txt or 'n/a'
    st.success(f"**{best_stock['ticker']}** - Predicted Return: **{pr_txt}**")
    # Confidence/Risk display with fallbacks
    try:
        import re
        def _numfmt(x, fmt=".1f"):
            try:
                v = float(re.sub(r'[^0-9.+-]', '', str(x)))
                return format(v, fmt)
            except Exception:
                return "n/a"
        conf_val = None
        # Prefer ensemble as overall confidence proxy if present
        if 'ensemble_score' in st.session_state.top_stocks_df.columns:
            conf_val = _numfmt(best_stock.get('ensemble_score'), ".3f")
            conf_label = "Ensemble"
        elif 'confidence_score' in st.session_state.top_stocks_df.columns:
            conf_val = _numfmt(best_stock.get('confidence_score'), ".1f")
            conf_label = "Confidence"
        elif 'lstm_confidence_score' in st.session_state.top_stocks_df.columns:
            conf_val = _numfmt(best_stock.get('lstm_confidence_score'), ".1f")
            conf_label = "LSTM Conf"
        else:
            conf_label = "Confidence"
            conf_val = "n/a"
        # Risk fallback
        risk_val = "n/a"
        if 'risk_score' in st.session_state.top_stocks_df.columns:
            risk_val = _numfmt(best_stock.get('risk_score'), ".1f")
        elif 'lstm_risk_score' in st.session_state.top_stocks_df.columns:
            risk_val = _numfmt(best_stock.get('lstm_risk_score'), ".1f")
        st.write(f"{conf_label}: {conf_val}/100 | Risk: {risk_val}/100")
    except Exception:
        pass
    with col2:
        # Current price (robust parsing)
        try:
            import re
            cval = float(re.sub(r'[^0-9.+-]','', str(best_stock.get('close',''))))
            st.metric("Current Price", f"${cval:.2f}")
        except Exception:
            st.metric("Current Price", str(best_stock.get('close','n/a')))
    with col3:
        # Predicted change: compute if not present
        try:
            import re, math
            if 'predicted_change' in best_stock.index:
                pc = float(re.sub(r'[^0-9.+-]','', str(best_stock.get('predicted_change',''))))
            else:
                # derive from predicted return pct and close
                pr_num = None
                for c in ['predicted_return_pct','xgb_predicted_return_pct','lstm_predicted_return_pct','lstm_pred']:
                    if c in best_stock.index:
                        try:
                            pr_num = float(re.sub(r'[^0-9.+-]','', str(best_stock.get(c,''))))
                            break
                        except Exception:
                            continue
                cnum = float(re.sub(r'[^0-9.+-]','', str(best_stock.get('close','')))) if 'close' in best_stock.index else float('nan')
                pc = (pr_num / 100.0) * cnum if pr_num is not None and math.isfinite(cnum) else float('nan')
            st.metric("Predicted Change", f"${pc:+.2f}" if pc == pc else "n/a")
        except Exception:
            st.metric("Predicted Change", "n/a")
    
    # --- Detailed Stock Analysis ---
    if st.button("🔍 Analyze Top Pick"):
        analysis = get_stock_analysis(best_stock['ticker'])
        if analysis:
            st.subheader(f"📋 Detailed Analysis: {best_stock['ticker']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Technical Signals**")
                def _fmt_float(_v, _nd=2):
                    try:
                        return f"{float(_v):.{_nd}f}"
                    except Exception:
                        return str(_v)
                for signal, value in analysis['technical_signals'].items():
                    if isinstance(value, bool):
                        status = "✅" if value else "❌"
                        st.write(f"{status} {signal.replace('_', ' ').title()}")
                    else:
                        st.write(f"📊 {signal.replace('_', ' ').title()}: {_fmt_float(value, 2)}")
            
            with col2:
                st.write("**Fundamental Signals**")
                for signal, value in analysis['fundamental_signals'].items():
                    if isinstance(value, bool):
                        status = "✅" if value else "❌"
                        st.write(f"{status} {signal.replace('_', ' ').title()}")
                    else:
                        st.write(f"📊 {signal.replace('_', ' ').title()}: {_fmt_float(value, 2)}")
            
            with col3:
                st.write("**Risk Metrics**")
                for metric, value in analysis['risk_metrics'].items():
                    st.write(f"📊 {metric.replace('_', ' ').title()}: {_fmt_float(value, 2)}")
    
    # --- Stock Price Chart Analysis ---
    st.header("📈 3-Week Price Chart Analysis")
    
    # Load historical price data
    step_ctx = "read_prices_csv"
    try:
        def _pick_prices_csv():
            env_path = os.getenv('STOCK_PRICES_CSV')
            candidates = []
            if env_path:
                candidates.append(env_path)
            candidates += [
                'data/input/stock_prices_with_clean_data.csv',
                'data/input/stock_prices_filtered.csv',
                'data/input/stock_prices.csv',
                'data/stock_prices.csv',
            ]
            for _p in candidates:
                # Resolve relative paths against project root
                _abs = _p if os.path.isabs(_p) else os.path.join(PRJ_DIR, _p)
                if os.path.exists(_abs):
                    return _abs
            return None

        stock_prices_path = _pick_prices_csv()
        if not stock_prices_path:
            raise FileNotFoundError("No historical prices CSV found. Set STOCK_PRICES_CSV or place clean prices at data/input/stock_prices_with_clean_data.csv")
        stock_data = pd.read_csv(stock_prices_path)
        # Ensure 'ticker' column exists even if source uses 'symbol'
        if 'ticker' not in stock_data.columns and 'symbol' in stock_data.columns:
            try:
                stock_data['ticker'] = stock_data['symbol'].astype(str).str.upper()
            except Exception:
                pass
        # Helper: validate and coerce numeric columns; log offending values
        step_ctx = "coerce_prices_numeric"
        def _validate_numeric_columns(df, cols, label):
            issues = []
            for _c in cols:
                if _c in df.columns:
                    coerced = pd.to_numeric(df[_c], errors='coerce')
                    # Identify non-numeric strings (original non-null that became NaN)
                    bad_mask = coerced.isna() & df[_c].notna()
                    if bool(bad_mask.any()):
                        samples = df.loc[bad_mask, _c].astype(str).unique().tolist()[:5]
                        issues.append(f"{_c}: {bad_mask.sum()} non-numeric (e.g., {samples})")
                    df[_c] = coerced
            if issues:
                import textwrap as _tw
                msg = "\n".join(_tw.wrap("; ".join(issues), width=120))
                try:
                    st.warning(f"Data validation for {label}: detected non-numeric values → {msg}")
                except Exception:
                    print(f"[UI VALIDATION] {label}: {msg}")
            return df
        stock_data = _validate_numeric_columns(stock_data, ['open','high','low','close','volume'], 'stock_prices.csv')
        st.success("✅ Historical price data loaded successfully")
        try:
            st.caption(f"Using prices from: {stock_prices_path}")
        except Exception:
            pass
        
        # Create stock selector
        available_tickers = st.session_state.top_stocks_df['ticker'].tolist()
        selected_ticker = st.selectbox(
            "Select a stock to analyze:",
            available_tickers,
            index=0,
            help="Choose a stock to view its 3-week price chart with technical analysis"
        )
        
        if selected_ticker:
            # Load complete featured data to get all technical indicators
            try:
                # Prefer full technical features if available, then model outputs, then legacy featured
                step_ctx = "load_weekly_output"
                candidates = [
                    os.path.join(PRJ_DIR, 'data', 'features', 'stock_features_clean.csv'),
                    os.path.join(PRJ_DIR, 'data', 'xgboost', 'xgboost_weekly_output.csv'),
                    os.path.join(PRJ_DIR, 'data', 'ensemble', 'ensemble_weekly_output.csv'),
                ]
                env_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
                legacy = [env_path, 'data/top/featured_stocks_top.csv', 'data/featured_stocks_top.csv']
                candidates += [p if os.path.isabs(p) else os.path.join(PRJ_DIR, p) for p in legacy]
                picked = None
                for _p in candidates:
                    if os.path.exists(_p):
                        picked = _p
                        break
                if picked is None:
                    raise FileNotFoundError("No featured/feature dataset found for technicals")
                complete_featured_data = pd.read_csv(picked)
                step_ctx = "normalize_featured_cols"
                if 'ticker' in complete_featured_data.columns:
                    complete_featured_data['ticker'] = complete_featured_data['ticker'].astype(str).str.upper()
                # Coerce common numeric fields to numeric to avoid f-string format errors downstream
                try:
                    _num_cols = [
                        'close','open','high','low','volume',
                        'support_20d','resistance_20d','rsi_14d',
                        'momentum_5d','momentum_10d','momentum_20d','momentum_30d','momentum_60d',
                        'predicted_return_pct','confidence_score','risk_score','composite_score'
                    ]
                    for _c in _num_cols:
                        if _c in complete_featured_data.columns:
                            complete_featured_data[_c] = pd.to_numeric(complete_featured_data[_c], errors='coerce')
                except Exception:
                    pass
                stock_featured_data = complete_featured_data[complete_featured_data['ticker'] == selected_ticker].iloc[0]
                # Backfill support/resistance if missing from weekly output
                step_ctx = "backfill_support_resistance"
                try:
                    need_sr = []
                    for _c in ['support_20d', 'resistance_20d']:
                        if _c not in stock_featured_data.index or pd.isna(stock_featured_data.get(_c)):
                            need_sr.append(_c)
                    if need_sr:
                        # Try from featured_stocks_top.csv
                        try:
                            _feat_top_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
                            _feat_top = pd.read_csv(_feat_top_path)
                            if 'ticker' in _feat_top.columns:
                                _feat_top['ticker'] = _feat_top['ticker'].astype(str).str.upper()
                                _row = _feat_top[_feat_top['ticker'] == selected_ticker]
                                if not _row.empty:
                                    _row = _row.iloc[0]
                                    for _c in need_sr:
                                        if _c in _row.index and not pd.isna(_row.get(_c)):
                                            stock_featured_data[_c] = _row.get(_c)
                        except Exception:
                            pass
                        # If still missing, compute from last 20 closes
                        still_missing = [
                            _c for _c in ['support_20d', 'resistance_20d']
                            if _c not in stock_featured_data.index or pd.isna(stock_featured_data.get(_c))
                        ]
                        if still_missing:
                            try:
                                _td = stock_data[stock_data['ticker'] == selected_ticker].copy()
                                _td['date'] = pd.to_datetime(_td['date'], errors='coerce')
                                _td = _td.sort_values('date').tail(20)
                                if len(_td) > 0:
                                    if 'support_20d' in still_missing:
                                        stock_featured_data['support_20d'] = float(_td['close'].min())
                                    if 'resistance_20d' in still_missing:
                                        stock_featured_data['resistance_20d'] = float(_td['close'].max())
                            except Exception:
                                pass
                except Exception:
                    pass
                
                # Backfill RSI and momentum from price history if missing
                try:
                    need_ta = []
                    for _c in ['rsi_14d', 'momentum_5d', 'momentum_10d']:
                        if _c not in stock_featured_data.index or pd.isna(stock_featured_data.get(_c)):
                            need_ta.append(_c)
                    if need_ta:
                        _td2 = stock_data[stock_data['ticker'] == selected_ticker].copy()
                        if not _td2.empty and 'close' in _td2.columns:
                            _td2['date'] = pd.to_datetime(_td2['date'], errors='coerce')
                            _td2 = _td2.sort_values('date')
                            _price = pd.to_numeric(_td2['close'], errors='coerce')
                            if 'rsi_14d' in need_ta and _price.notna().sum() >= 14:
                                _delta = _price.diff(1)
                                _gain = _delta.where(_delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
                                _loss = (-_delta.where(_delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
                                _rs = _gain / _loss
                                _rsi = 100 - (100 / (1 + _rs))
                                try:
                                    stock_featured_data['rsi_14d'] = float(_rsi.iloc[-1])
                                except Exception:
                                    pass
                            if 'momentum_5d' in need_ta and len(_price) > 5:
                                try:
                                    stock_featured_data['momentum_5d'] = float(_price.pct_change(5).iloc[-1])
                                except Exception:
                                    pass
                            if 'momentum_10d' in need_ta and len(_price) > 10:
                                try:
                                    stock_featured_data['momentum_10d'] = float(_price.pct_change(10).iloc[-1])
                                except Exception:
                                    pass
                except Exception:
                    pass

                # Get predicted return and confidence data from filtered stocks if available (robust columns)
                if not st.session_state.top_stocks_df.empty:
                    filtered_stock_data = st.session_state.top_stocks_df[st.session_state.top_stocks_df['ticker'] == selected_ticker]
                    if not filtered_stock_data.empty:
                        filtered_row = filtered_stock_data.iloc[0]
                        # Predicted return: prefer combined/xgb/lstm
                        _pred_candidates = [
                            'predicted_return_pct', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct',
                            'xgb_pred', 'lstm_pred'
                        ]
                        for _pc in _pred_candidates:
                            if _pc in filtered_row.index and pd.notna(filtered_row.get(_pc)):
                                stock_featured_data['predicted_return_pct'] = filtered_row.get(_pc)
                                break
                        # Confidence: max across available confidence columns
                        _conf_candidates = [
                            'confidence_score', 'xgb_confidence_score', 'lstm_confidence_score',
                            'confidence_score_xgb', 'confidence_score_lstm'
                        ]
                        _confs = []
                        for _cc in _conf_candidates:
                            if _cc in filtered_row.index:
                                try:
                                    _confs.append(float(str(filtered_row.get(_cc)).replace('%','')))
                                except Exception:
                                    continue
                        if _confs:
                            stock_featured_data['confidence_score'] = max(_confs)
                        # Other optional fields
                        for _k in ['predicted_change', 'risk_score', 'composite_score']:
                            if _k in filtered_row.index and pd.notna(filtered_row.get(_k)):
                                stock_featured_data[_k] = filtered_row.get(_k)
                
                st.success(f"✅ Complete technical data loaded for {selected_ticker}")
            except Exception as e:
                # Fallback to filtered data if complete data not available
                stock_featured_data = st.session_state.top_stocks_df[st.session_state.top_stocks_df['ticker'] == selected_ticker].iloc[0]
                st.warning(f"⚠️ Using filtered data for {selected_ticker} (some technical indicators may be missing)")
            
            # Create tabs for different chart views
            candlestick_tab, price_graph_tab, earnings_tab = st.tabs(["📊 Candlestick Chart", "📈 Price Graph", "📋 Earnings History"])
            
            with candlestick_tab:
                # Create the price chart (current candlestick view)
                chart = create_stock_price_chart(selected_ticker, stock_data, stock_featured_data)
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    st.subheader(f"📊 Technical Analysis Summary: {selected_ticker}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        import re as _re
                        def _num(x):
                            try:
                                return float(_re.sub(r'[^0-9.+-]', '', str(x)))
                            except Exception:
                                return float('nan')
                        _cp = _num(stock_featured_data.get('close'))
                        _res = _num(stock_featured_data.get('resistance_20d'))
                        _sup = _num(stock_featured_data.get('support_20d'))
                        st.metric("Current Price", f"${_cp:.2f}" if _cp == _cp else str(stock_featured_data.get('close','N/A')))
                        if not pd.isna(_res):
                            st.metric("Resistance Level", f"${_res:.2f}")
                        if not pd.isna(_sup):
                            st.metric("Support Level", f"${_sup:.2f}")
                    
                    with col2:
                        rsi_value = stock_featured_data.get('rsi_14d')
                        rsi_display = f"{rsi_value:.1f}" if rsi_value is not None and not pd.isna(rsi_value) else "N/A"
                        st.metric("RSI (14d)", rsi_display)
                        
                        momentum_5d = stock_featured_data.get('momentum_5d')
                        momentum_5d_display = f"{momentum_5d:.2f}" if momentum_5d is not None and not pd.isna(momentum_5d) else "N/A"
                        st.metric("Momentum (5d)", momentum_5d_display)
                        
                        momentum_10d = stock_featured_data.get('momentum_10d')
                        momentum_10d_display = f"{momentum_10d:.2f}" if momentum_10d is not None and not pd.isna(momentum_10d) else "N/A"
                        st.metric("Momentum (10d)", momentum_10d_display)
                    
                    with col3:
                        resistance_status = "🟢 BROKEN" if stock_featured_data.get('broke_resistance', False) else "🔴 HOLDING"
                        st.metric("Resistance Status", resistance_status)
                        # Predicted Return
                        _pr = stock_featured_data.get('predicted_return_pct')
                        try:
                            _prn = float(_pr)
                            st.metric("Predicted Return", f"{_prn:.2f}%")
                        except Exception:
                            st.metric("Predicted Return", str(_pr) if _pr is not None else "N/A")
                        # Confidence Score
                        _cs = stock_featured_data.get('confidence_score')
                        try:
                            _csn = float(_cs)
                            st.metric("Confidence Score", f"{_csn:.1f}/100")
                        except Exception:
                            st.metric("Confidence Score", str(_cs) if _cs is not None else "N/A")
                    
                    # Add key price levels section
                    st.subheader("🎯 Key Price Levels & Validation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Resistance break price
                        if stock_featured_data.get('broke_resistance', False):
                            resistance_level = _num(stock_featured_data.get('resistance_20d'))
                            current_price = _num(stock_featured_data.get('close'))
                            if resistance_level == resistance_level:
                                price_above_resistance = current_price - resistance_level
                                st.metric(
                                    "🟢 Resistance Break Price", 
                                    f"${resistance_level:.2f}",
                                    delta=f"+${price_above_resistance:.2f} above resistance"
                                )
                                st.info(f"**Resistance Level**: ${resistance_level:.2f} was broken. Current price ${current_price:.2f} is ${price_above_resistance:.2f} above resistance.")
                        else:
                            resistance_level = _num(stock_featured_data.get('resistance_20d'))
                            current_price = _num(stock_featured_data.get('close'))
                            if resistance_level == resistance_level:
                                distance_to_resistance = resistance_level - current_price
                                st.metric(
                                    "🔴 Resistance Level", 
                                    f"${resistance_level:.2f}",
                                    delta=f"${distance_to_resistance:.2f} to break"
                                )
                                st.info(f"**Resistance Level**: ${resistance_level:.2f}. Current price ${current_price:.2f} needs to rise ${distance_to_resistance:.2f} to break resistance.")
                            else:
                                st.metric("🔴 Resistance Break Price", "Not broken yet")
                        
                        # Support level
                        support_level = _num(stock_featured_data.get('support_20d'))
                        if support_level == support_level:
                            current_price = _num(stock_featured_data.get('close'))
                            distance_from_support = current_price - support_level
                            st.metric(
                                "🟢 Support Level", 
                                f"${support_level:.2f}",
                                delta=f"+${distance_from_support:.2f} above support"
                            )
                            st.info(f"**Support Level**: ${support_level:.2f}. Current price ${current_price:.2f} is ${distance_from_support:.2f} above support.")
                        else:
                            st.metric("🔴 Support Level", "Not available")
                    
                    with col2:
                        # Momentum identification price
                        momentum_5d = stock_featured_data.get('momentum_5d')
                        momentum_10d = stock_featured_data.get('momentum_10d')
                        if momentum_5d is not None and momentum_10d is not None and momentum_5d > 0 and momentum_10d > 0:
                            current_price = stock_featured_data['close']
                            # Calculate momentum strength
                            momentum_strength = (momentum_5d + momentum_10d) / 2
                            st.metric(
                                "🚀 Momentum Identified", 
                                f"${current_price:.2f}",
                                delta=f"+{momentum_strength:.1f}% momentum"
                            )
                        else:
                            st.metric("📊 Momentum Status", "Mixed/Weak")
                        
                        # RSI interpretation
                        rsi_value = stock_featured_data.get('rsi_14d')
                        if rsi_value is not None and not pd.isna(rsi_value):
                            if rsi_value > 70:
                                st.metric("⚠️ RSI Status", "Overbought", delta=f"{rsi_value:.1f}")
                            elif rsi_value < 30:
                                st.metric("🟢 RSI Status", "Oversold", delta=f"{rsi_value:.1f}")
                            else:
                                st.metric("📊 RSI Status", "Neutral", delta=f"{rsi_value:.1f}")
                        else:
                            st.metric("📊 RSI Status", "N/A")
                    
                    # Add resistance price details
                    st.subheader("🎯 Resistance Price Analysis")
                    
                    resistance_level = _num(stock_featured_data.get('resistance_20d'))
                    current_price = _num(stock_featured_data.get('close'))
                    broke_resistance = stock_featured_data.get('broke_resistance', False)
                    
                    if resistance_level == resistance_level:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if broke_resistance:
                                price_above_resistance = current_price - resistance_level
                                st.success(f"**🟢 RESISTANCE BROKEN!**")
                                st.metric(
                                    "Resistance Price", 
                                    f"${resistance_level:.2f}",
                                    delta=f"BROKEN by ${price_above_resistance:.2f}"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                                st.write(f"**Price Above Resistance**: ${price_above_resistance:.2f}")
                            else:
                                distance_to_resistance = resistance_level - current_price
                                st.warning(f"**🔴 RESISTANCE NOT BROKEN**")
                                st.metric(
                                    "Resistance Price", 
                                    f"${resistance_level:.2f}",
                                    delta=f"${distance_to_resistance:.2f} to break"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                                st.write(f"**Distance to Break**: ${distance_to_resistance:.2f}")
                        
                        with col2:
                            # Show resistance break date if available
                            if broke_resistance:
                                st.info("**📅 Resistance Break Details**")
                                st.write("✅ **Status**: Resistance level has been successfully broken")
                                st.write("📈 **Signal**: Bullish - price is now above resistance")
                                st.write("🎯 **Next Target**: Look for continued upward momentum")
                            else:
                                st.info("**📅 Resistance Status**")
                                st.write("⏳ **Status**: Price is below resistance level")
                                st.write("📊 **Signal**: Waiting for breakout confirmation")
                                st.write("🎯 **Watch For**: Price breaking above resistance with volume")
                    else:
                        st.warning("Resistance level data not available for this stock.")
                    
                    # Add support level analysis
                    st.subheader("🎯 Support Level Analysis")
                    
                    support_level = _num(stock_featured_data.get('support_20d'))
                    current_price = _num(stock_featured_data.get('close'))
                    
                    if support_level == support_level:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            distance_from_support = current_price - support_level
                            if distance_from_support > 0:
                                st.success(f"**🟢 ABOVE SUPPORT**")
                                st.metric(
                                    "Support Price", 
                                    f"${support_level:.2f}",
                                    delta=f"${distance_from_support:.2f} above support"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Support Level**: ${support_level:.2f}")
                                st.write(f"**Distance Above Support**: ${distance_from_support:.2f}")
                            else:
                                st.error(f"**🔴 BELOW SUPPORT**")
                                st.metric(
                                    "Support Price", 
                                    f"${support_level:.2f}",
                                    delta=f"${abs(distance_from_support):.2f} below support"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Support Level**: ${support_level:.2f}")
                                st.write(f"**Distance Below Support**: ${abs(distance_from_support):.2f}")
                        
                        with col2:
                            if distance_from_support > 0:
                                st.info("**📅 Support Status**")
                                st.write("✅ **Status**: Price is above support level")
                                st.write("📈 **Signal**: Bullish - price has support below")
                                st.write("🎯 **Next Target**: Look for continued upward movement")
                            else:
                                st.warning("**📅 Support Status**")
                                st.write("⚠️ **Status**: Price is below support level")
                                st.write("📉 **Signal**: Bearish - price may continue falling")
                                st.write("🎯 **Watch For**: Price bouncing back above support")
                    else:
                        st.warning("Support level data not available for this stock.")
                    
                    # Add detailed technical insights
                    st.subheader("🔍 Technical Insights")
                    
                    insights = []
                    
                    # Resistance analysis
                    if stock_featured_data.get('broke_resistance', False):
                        insights.append("🟢 **Resistance Break**: Price has successfully broken above the 20-day resistance level, indicating bullish momentum.")
                    else:
                        insights.append("🔴 **Resistance Test**: Price is testing the resistance level. A break above could signal upward momentum.")
                    
                    # RSI analysis
                    rsi = stock_featured_data.get('rsi_14d')
                    if rsi is not None and not pd.isna(rsi):
                        if rsi > 70:
                            insights.append("⚠️ **Overbought**: RSI above 70 suggests the stock may be overbought and could face resistance.")
                        elif rsi < 30:
                            insights.append("🟢 **Oversold**: RSI below 30 suggests the stock may be oversold and could bounce back.")
                        else:
                            insights.append("📊 **Neutral RSI**: RSI in neutral territory, no extreme overbought/oversold conditions.")
                    else:
                        insights.append("📊 **RSI**: RSI data not available for analysis.")
                    
                    # Momentum analysis
                    momentum_5d = stock_featured_data.get('momentum_5d')
                    momentum_10d = stock_featured_data.get('momentum_10d')
                    
                    if momentum_5d is not None and momentum_10d is not None and not pd.isna(momentum_5d) and not pd.isna(momentum_10d):
                        if momentum_5d > 0 and momentum_10d > 0:
                            insights.append("🚀 **Positive Momentum**: Both 5-day and 10-day momentum are positive, indicating upward price movement.")
                        elif momentum_5d < 0 and momentum_10d < 0:
                            insights.append("📉 **Negative Momentum**: Both 5-day and 10-day momentum are negative, indicating downward pressure.")
                        else:
                            insights.append("📊 **Mixed Momentum**: Short-term and medium-term momentum are mixed, suggesting consolidation.")
                    else:
                        insights.append("📊 **Momentum**: Momentum data not available for analysis.")
                    
                    # Display insights
                    for insight in insights:
                        st.write(insight)
                    
                    # Add comprehensive chart interpretation guide
                    st.subheader("📚 How to Interpret the Chart for Investment Decisions")
                    
                    # Create expandable sections for different signal types
                    with st.expander("🟢 Bullish Signals (Good to Buy)", expanded=False):
                        st.markdown("""
                        **Resistance Break**: Price closes above the red resistance line
                        - Look for the 🟢 Resistance Break annotation on the chart
                        - This indicates strong buying pressure and potential upward movement
                        
                        **Green Candlesticks**: Multiple consecutive green days
                        - Shows consistent buying pressure
                        - Each green candle means the day closed higher than it opened
                        
                        **High Volume**: Volume increases with price gains
                        - Check the gray volume bars at the bottom
                        - Higher bars with price increases confirm strong buying interest
                        
                        **RSI 30-70**: Stock is not overbought
                        - RSI in the neutral range allows for continued upside
                        - Avoid stocks with RSI > 70 (overbought)
                        
                        **Momentum Point**: Recent strong upward movement
                        - Look for the 🚀 Momentum annotation on the chart
                        - Shows the peak of recent buying pressure
                        """)
                    
                    with st.expander("🔴 Bearish Signals (Avoid or Sell)", expanded=False):
                        st.markdown("""
                        **Support Break**: Price closes below the green support line
                        - This indicates potential further downside
                        - Support level acts as a floor for the stock price
                        
                        **Red Candlesticks**: Multiple consecutive red days
                        - Shows consistent selling pressure
                        - Each red candle means the day closed lower than it opened
                        
                        **High Volume Down**: High volume with price declines
                        - High volume bars with price drops indicate strong selling
                        - This confirms bearish sentiment
                        
                        **RSI >70**: Stock is overbought and may pull back
                        - RSI above 70 suggests the stock may be due for a correction
                        - Consider taking profits or waiting for a pullback
                        
                        **Low Volume**: Lack of buying interest
                        - Small volume bars indicate lack of conviction
                        - Price movements without volume are less reliable
                        """)
                    
                    with st.expander("📊 Neutral/Consolidation", expanded=False):
                        st.markdown("""
                        **Small Candlesticks**: Price moving sideways
                        - Small candle bodies indicate indecision
                        - Stock is consolidating before next move
                        
                        **Low Volume**: Lack of strong directional movement
                        - Low volume suggests lack of conviction
                        - Wait for volume confirmation before making decisions
                        
                        **RSI 40-60**: Neutral momentum
                        - RSI in middle range indicates balanced buying/selling
                        - Stock is not overbought or oversold
                        """)
                    
                    with st.expander("📋 Technical Analysis Summary Section", expanded=False):
                        st.markdown("""
                        **Current Price Metrics**
                        - **Current Price**: Latest closing price from the chart
                        - **Resistance Level**: Price level to watch for breakouts (red line)
                        - **Support Level**: Price level to watch for breakdowns (green line)
                        
                        **Momentum Indicators**
                        - **RSI (14d)**:
                          - 0-30: Oversold (potential buy opportunity)
                          - 30-70: Normal trading range
                          - 70-100: Overbought (potential sell signal)
                        - **Momentum (5d)**: Short-term price change percentage
                        - **Momentum (10d)**: Medium-term price change percentage
                        
                        **Status Indicators**
                        - **Resistance Status**:
                          - 🟢 BROKEN: Bullish signal - price above resistance
                          - 🔴 HOLDING: Price still below resistance level
                        - **Predicted Return**: AI model's forecast for price movement
                        - **Confidence Score**: How confident the model is (0-100 scale)
                        """)
                    
                    with st.expander("🔍 Technical Insights Section", expanded=False):
                        st.markdown("""
                        **Resistance Analysis**
                        - 🟢 **Resistance Break**: "Price has successfully broken above the 20-day resistance level, indicating bullish momentum"
                        - 🔴 **Resistance Test**: "Price is testing the resistance level. A break above could signal upward momentum"
                        
                        **RSI Analysis**
                        - ⚠️ **Overbought**: "RSI above 70 suggests the stock may be overbought and could face resistance"
                        - 🟢 **Oversold**: "RSI below 30 suggests the stock may be oversold and could bounce back"
                        - 📊 **Neutral RSI**: "RSI in neutral territory, no extreme overbought/oversold conditions"
                        
                        **Momentum Analysis**
                        - 🚀 **Positive Momentum**: "Both 5-day and 10-day momentum are positive, indicating upward price movement"
                        - 📉 **Negative Momentum**: "Both 5-day and 10-day momentum are negative, indicating downward pressure"
                        - 📊 **Mixed Momentum**: "Short-term and medium-term momentum are mixed, suggesting consolidation"
                        """)
                    
                    with st.expander("💡 Practical Investment Strategy", expanded=False):
                        st.markdown("""
                        **For Buying (Entry Points)**
                        1. **Wait for resistance break** with high volume
                        2. **Buy on pullbacks** to support level
                        3. **Look for oversold RSI** (<30) for bounce-back opportunities
                        4. **Confirm with positive momentum** (5d and 10d both positive)
                        
                        **For Selling (Exit Points)**
                        1. **Sell on resistance rejection** (price fails to break above resistance)
                        2. **Exit on support break** (price closes below support)
                        3. **Take profits on overbought RSI** (>70)
                        4. **Watch for negative momentum** (both 5d and 10d negative)
                        
                        **Risk Management**
                        - **Set stop-loss** below support level
                        - **Take partial profits** at resistance levels
                        - **Don't chase overbought stocks** (RSI >70)
                        - **Use volume confirmation** for major moves
                        """)
                    
            with price_graph_tab:
                st.subheader("📈 3-Week Price Graph")
                st.write("This tab will show a simple line chart of the 3-week price movement.")
                
                # Create a simple line chart for the 3-week period
                try:
                    # Filter data for the specific ticker
                    ticker_data = stock_data[stock_data['ticker'] == selected_ticker].copy()
                    
                    if not ticker_data.empty:
                        # Convert date column to datetime
                        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                        
                        # Sort by date and get last 3 weeks (21 trading days)
                        ticker_data = ticker_data.sort_values('date')
                        last_date = ticker_data['date'].max()
                        start_date = last_date - timedelta(days=30)  # 30 days to ensure we get 21 trading days
                        
                        # Filter for last 3 weeks
                        recent_data = ticker_data[ticker_data['date'] >= start_date].copy()
                        
                        if len(recent_data) >= 5:  # Need at least 5 days for meaningful chart
                            # Create line chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=recent_data['date'],
                                y=recent_data['close'],
                                mode='lines+markers',
                                name='Close Price',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6)
                            ))
                            
                            # Add moving averages
                            if len(recent_data) >= 20:
                                recent_data['MA20'] = recent_data['close'].rolling(window=20).mean()
                                fig.add_trace(go.Scatter(
                                    x=recent_data['date'],
                                    y=recent_data['MA20'],
                                    mode='lines',
                                    name='20-Day MA',
                                    line=dict(color='orange', width=1, dash='dash')
                                ))
                            
                            if len(recent_data) >= 50:
                                recent_data['MA50'] = recent_data['close'].rolling(window=50).mean()
                                fig.add_trace(go.Scatter(
                                    x=recent_data['date'],
                                    y=recent_data['MA50'],
                                    mode='lines',
                                    name='50-Day MA',
                                    line=dict(color='red', width=1, dash='dash')
                                ))
                            
                            fig.update_layout(
                                title=f"{selected_ticker} - 3-Week Price Movement",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add price statistics
                            st.subheader("📊 Price Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Start Price", f"${recent_data['close'].iloc[0]:.2f}")
                            with col2:
                                st.metric("End Price", f"${recent_data['close'].iloc[-1]:.2f}")
                            with col3:
                                price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
                                price_change_pct = (price_change / recent_data['close'].iloc[0]) * 100
                                st.metric("Price Change", f"${price_change:.2f}", delta=f"{price_change_pct:.2f}%")
                            with col4:
                                st.metric("Highest Price", f"${recent_data['close'].max():.2f}")
                            
                            # Add volume analysis
                            st.subheader("📈 Volume Analysis")
                            if 'volume' in recent_data.columns:
                                avg_volume = recent_data['volume'].mean()
                                current_volume = recent_data['volume'].iloc[-1]
                                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Volume", f"{avg_volume:,.0f}")
                                with col2:
                                    st.metric("Current Volume", f"{current_volume:,.0f}")
                                with col3:
                                    st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                                
                                # Volume trend
                                if volume_ratio > 1.5:
                                    st.success("📈 **High Volume**: Current volume is significantly above average, indicating strong interest.")
                                elif volume_ratio < 0.5:
                                    st.warning("📉 **Low Volume**: Current volume is below average, indicating weak interest.")
                                else:
                                    st.info("📊 **Normal Volume**: Current volume is within normal range.")
                        else:
                            st.warning("Insufficient data for 3-week analysis. Need at least 5 trading days.")
                    else:
                        st.error(f"No data found for {selected_ticker}")
                except Exception as e:
                    st.error(f"Error creating price graph: {str(e)}")
            
            with earnings_tab:
                st.subheader("📋 Quarterly Earnings & Revenue History")
                st.write("This tab shows how the company has performed against earnings and revenue expectations over the last year.")
                
                # Load earnings history data
                try:
                    earnings_history_path = os.getenv('EARNINGS_HISTORY_CSV', 'data/earnings_history.csv')
                    earnings_data = pd.read_csv(earnings_history_path)
                    earnings_ticker_data = earnings_data[earnings_data['ticker'] == selected_ticker]
                    
                    if not earnings_ticker_data.empty:
                        # Convert earnings_date to datetime
                        earnings_ticker_data['earnings_date'] = pd.to_datetime(earnings_ticker_data['earnings_date'])
                        
                        # Sort by date (most recent first)
                        earnings_ticker_data = earnings_ticker_data.sort_values('earnings_date', ascending=False)
                        
                        # Display earnings history table
                        st.subheader("📊 Earnings Performance History")
                        
                        # Format the data for display
                        display_data = earnings_ticker_data.copy()
                        display_data['earnings_date'] = display_data['earnings_date'].dt.strftime('%Y-%m-%d')
                        display_data['reported_eps'] = display_data['reported_eps'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        display_data['estimate_eps'] = display_data['estimate_eps'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        display_data['surprise_percentage'] = display_data['surprise_percentage'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        
                        # Create earnings surprise chart
                        st.subheader("📈 Earnings Surprise Trend")
                        
                        # Filter out rows with missing data
                        chart_data = earnings_ticker_data.dropna(subset=['surprise_percentage'])
                        
                        if not chart_data.empty:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=chart_data['earnings_date'].dt.strftime('%Y-%m'),
                                y=chart_data['surprise_percentage'],
                                name='Earnings Surprise %',
                                marker_color=['green' if x > 0 else 'red' for x in chart_data['surprise_percentage']]
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_ticker} - Earnings Surprise Percentage",
                                xaxis_title="Quarter",
                                yaxis_title="Surprise Percentage (%)",
                                height=400,
                                showlegend=False
                            )
                            
                            # Add horizontal line at 0
                            fig.add_hline(y=0, line_dash="dash", line_color="black")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Earnings analysis
                            st.subheader("📊 Earnings Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                positive_surprises = len(chart_data[chart_data['surprise_percentage'] > 0])
                                total_quarters = len(chart_data)
                                beat_rate = (positive_surprises / total_quarters * 100) if total_quarters > 0 else 0
                                st.metric("Beat Rate", f"{beat_rate:.1f}%")
                            
                            with col2:
                                avg_surprise = chart_data['surprise_percentage'].mean()
                                st.metric("Avg Surprise", f"{avg_surprise:.2f}%")
                            
                            with col3:
                                max_surprise = chart_data['surprise_percentage'].max()
                                st.metric("Best Surprise", f"{max_surprise:.2f}%")
                            
                            with col4:
                                min_surprise = chart_data['surprise_percentage'].min()
                                st.metric("Worst Surprise", f"{min_surprise:.2f}%")
                            
                            # Earnings consistency analysis
                            st.subheader("🎯 Earnings Consistency Analysis")
                            
                            if beat_rate >= 75:
                                st.success("🏆 **Excellent Track Record**: Company consistently beats earnings expectations.")
                            elif beat_rate >= 50:
                                st.info("📈 **Good Track Record**: Company beats expectations more often than not.")
                            elif beat_rate >= 25:
                                st.warning("⚠️ **Mixed Track Record**: Company has difficulty consistently beating expectations.")
                            else:
                                st.error("📉 **Poor Track Record**: Company rarely beats earnings expectations.")
                            
                            # Recent performance
                            if len(chart_data) >= 2:
                                recent_surprise = chart_data.iloc[0]['surprise_percentage']
                                previous_surprise = chart_data.iloc[1]['surprise_percentage']
                                
                                st.subheader("📅 Recent Performance")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Latest Quarter", f"{recent_surprise:.2f}%")
                                with col2:
                                    st.metric("Previous Quarter", f"{previous_surprise:.2f}%")
                                
                                if recent_surprise > previous_surprise:
                                    st.success("📈 **Improving**: Latest quarter showed better performance than the previous quarter.")
                                elif recent_surprise < previous_surprise:
                                    st.warning("📉 **Declining**: Latest quarter showed worse performance than the previous quarter.")
                                else:
                                    st.info("📊 **Stable**: Performance has remained consistent between quarters.")
                        else:
                            st.warning("No earnings surprise data available for analysis.")
                    else:
                        st.warning(f"No earnings history data found for {selected_ticker}")
                        
                        # Show available tickers
                        available_tickers = earnings_data['ticker'].unique()
                        st.info(f"Available tickers in earnings data: {', '.join(available_tickers)}")
                except Exception as e:
                    st.error(f"Error loading earnings data: {str(e)}")
                    st.info("Make sure the earnings_history.csv file is available in the data folder.")
                
                resistance_level = _num(stock_featured_data.get('resistance_20d'))
                current_price = _num(stock_featured_data.get('close'))
                broke_resistance = stock_featured_data.get('broke_resistance', False)
                
                if resistance_level == resistance_level:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if broke_resistance:
                            price_above_resistance = current_price - resistance_level
                            st.success(f"**🟢 RESISTANCE BROKEN!**")
                            st.metric(
                                "Resistance Price", 
                                f"${resistance_level:.2f}",
                                delta=f"BROKEN by ${price_above_resistance:.2f}"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                            st.write(f"**Price Above Resistance**: ${price_above_resistance:.2f}")
                        else:
                            distance_to_resistance = resistance_level - current_price
                            st.warning(f"**🔴 RESISTANCE NOT BROKEN**")
                            st.metric(
                                "Resistance Price", 
                                f"${resistance_level:.2f}",
                                delta=f"${distance_to_resistance:.2f} to break"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                            st.write(f"**Distance to Break**: ${distance_to_resistance:.2f}")
                    
                    with col2:
                        # Show resistance break date if available
                        if broke_resistance:
                            st.info("**📅 Resistance Break Details**")
                            st.write("✅ **Status**: Resistance level has been successfully broken")
                            st.write("📈 **Signal**: Bullish - price is now above resistance")
                            st.write("🎯 **Next Target**: Look for continued upward momentum")
                        else:
                            st.info("**📅 Resistance Status**")
                            st.write("⏳ **Status**: Price is below resistance level")
                            st.write("📊 **Signal**: Waiting for breakout confirmation")
                            st.write("🎯 **Watch For**: Price breaking above resistance with volume")
                else:
                    st.warning("Resistance level data not available for this stock.")
                
                # Add support level analysis
                st.subheader("🎯 Support Level Analysis")
                
                support_level = _num(stock_featured_data.get('support_20d'))
                current_price = _num(stock_featured_data.get('close'))
                
                if support_level == support_level:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        distance_from_support = current_price - support_level
                        if distance_from_support > 0:
                            st.success(f"**🟢 ABOVE SUPPORT**")
                            st.metric(
                                "Support Price", 
                                f"${support_level:.2f}",
                                delta=f"${distance_from_support:.2f} above support"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Support Level**: ${support_level:.2f}")
                            st.write(f"**Distance Above Support**: ${distance_from_support:.2f}")
                        else:
                            st.error(f"**🔴 BELOW SUPPORT**")
                            st.metric(
                                "Support Price", 
                                f"${support_level:.2f}",
                                delta=f"${abs(distance_from_support):.2f} below support"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Support Level**: ${support_level:.2f}")
                            st.write(f"**Distance Below Support**: ${abs(distance_from_support):.2f}")
                    
                    with col2:
                        if distance_from_support > 0:
                            st.info("**📅 Support Status**")
                            st.write("✅ **Status**: Price is above support level")
                            st.write("📈 **Signal**: Bullish - price has support below")
                            st.write("🎯 **Next Target**: Look for continued upward movement")
                        else:
                            st.warning("**📅 Support Status**")
                            st.write("⚠️ **Status**: Price is below support level")
                            st.write("📉 **Signal**: Bearish - price may continue falling")
                            st.write("🎯 **Watch For**: Price bouncing back above support")
                else:
                    st.warning("Support level data not available for this stock.")
                    
                    # Add detailed technical insights
                    st.subheader("🔍 Technical Insights")
                    
                    insights = []
                    
                    # Resistance analysis
                    if stock_featured_data.get('broke_resistance', False):
                        insights.append("🟢 **Resistance Break**: Price has successfully broken above the 20-day resistance level, indicating bullish momentum.")
                    else:
                        insights.append("🔴 **Resistance Test**: Price is testing the resistance level. A break above could signal upward momentum.")
                    
                    # RSI analysis
                    rsi = stock_featured_data.get('rsi_14d')
                    if rsi is not None and not pd.isna(rsi):
                        if rsi > 70:
                            insights.append("⚠️ **Overbought**: RSI above 70 suggests the stock may be overbought and could face resistance.")
                        elif rsi < 30:
                            insights.append("🟢 **Oversold**: RSI below 30 suggests the stock may be oversold and could bounce back.")
                        else:
                            insights.append("📊 **Neutral RSI**: RSI in neutral territory, no extreme overbought/oversold conditions.")
                    else:
                        insights.append("📊 **RSI**: RSI data not available for analysis.")
                    
                    # Momentum analysis
                    momentum_5d = stock_featured_data.get('momentum_5d')
                    momentum_10d = stock_featured_data.get('momentum_10d')
                    
                    if momentum_5d is not None and momentum_10d is not None and not pd.isna(momentum_5d) and not pd.isna(momentum_10d):
                        if momentum_5d > 0 and momentum_10d > 0:
                            insights.append("🚀 **Positive Momentum**: Both 5-day and 10-day momentum are positive, indicating upward price movement.")
                        elif momentum_5d < 0 and momentum_10d < 0:
                            insights.append("📉 **Negative Momentum**: Both 5-day and 10-day momentum are negative, indicating downward pressure.")
                        else:
                            insights.append("📊 **Mixed Momentum**: Short-term and medium-term momentum are mixed, suggesting consolidation.")
                    else:
                        insights.append("📊 **Momentum**: Momentum data not available for analysis.")
                    
                    # Display insights
                    for insight in insights:
                        st.write(insight)
                    
                    # Add comprehensive chart interpretation guide
                    st.subheader("📚 How to Interpret the Chart for Investment Decisions")
                    
                    # Create expandable sections for different signal types
                    with st.expander("🟢 Bullish Signals (Good to Buy)", expanded=False):
                        st.markdown("""
                        **Resistance Break**: Price closes above the red resistance line
                        - Look for the 🟢 Resistance Break annotation on the chart
                        - This indicates strong buying pressure and potential upward movement
                        
                        **Green Candlesticks**: Multiple consecutive green days
                        - Shows consistent buying pressure
                        - Each green candle means the day closed higher than it opened
                        
                        **High Volume**: Volume increases with price gains
                        - Check the gray volume bars at the bottom
                        - Higher bars with price increases confirm strong buying interest
                        
                        **RSI 30-70**: Stock is not overbought
                        - RSI in the neutral range allows for continued upside
                        - Avoid stocks with RSI > 70 (overbought)
                        
                        **Momentum Point**: Recent strong upward movement
                        - Look for the 🚀 Momentum annotation on the chart
                        - Shows the peak of recent buying pressure
                        """)
                    
                    with st.expander("🔴 Bearish Signals (Avoid or Sell)", expanded=False):
                        st.markdown("""
                        **Support Break**: Price closes below the green support line
                        - This indicates potential further downside
                        - Support level acts as a floor for the stock price
                        
                        **Red Candlesticks**: Multiple consecutive red days
                        - Shows consistent selling pressure
                        - Each red candle means the day closed lower than it opened
                        
                        **High Volume Down**: High volume with price declines
                        - High volume bars with price drops indicate strong selling
                        - This confirms bearish sentiment
                        
                        **RSI >70**: Stock is overbought and may pull back
                        - RSI above 70 suggests the stock may be due for a correction
                        - Consider taking profits or waiting for a pullback
                        
                        **Low Volume**: Lack of buying interest
                        - Small volume bars indicate lack of conviction
                        - Price movements without volume are less reliable
                        """)
                    
                    with st.expander("📊 Neutral/Consolidation", expanded=False):
                        st.markdown("""
                        **Small Candlesticks**: Price moving sideways
                        - Small candle bodies indicate indecision
                        - Stock is consolidating before next move
                        
                        **Low Volume**: Lack of strong directional movement
                        - Low volume suggests lack of conviction
                        - Wait for volume confirmation before making decisions
                        
                        **RSI 40-60**: Neutral momentum
                        - RSI in middle range indicates balanced buying/selling
                        - Stock is not overbought or oversold
                        """)
                    
                    with st.expander("📋 Technical Analysis Summary Section", expanded=False):
                        st.markdown("""
                        **Current Price Metrics**
                        - **Current Price**: Latest closing price from the chart
                        - **Resistance Level**: Price level to watch for breakouts (red line)
                        - **Support Level**: Price level to watch for breakdowns (green line)
                        
                        **Momentum Indicators**
                        - **RSI (14d)**:
                          - 0-30: Oversold (potential buy opportunity)
                          - 30-70: Normal trading range
                          - 70-100: Overbought (potential sell signal)
                        - **Momentum (5d)**: Short-term price change percentage
                        - **Momentum (10d)**: Medium-term price change percentage
                        
                        **Status Indicators**
                        - **Resistance Status**:
                          - 🟢 BROKEN: Bullish signal - price above resistance
                          - 🔴 HOLDING: Price still below resistance level
                        - **Predicted Return**: AI model's forecast for price movement
                        - **Confidence Score**: How confident the model is (0-100 scale)
                        """)
                    
                    with st.expander("🔍 Technical Insights Section", expanded=False):
                        st.markdown("""
                        **Resistance Analysis**
                        - 🟢 **Resistance Break**: "Price has successfully broken above the 20-day resistance level, indicating bullish momentum"
                        - 🔴 **Resistance Test**: "Price is testing the resistance level. A break above could signal upward momentum"
                        
                        **RSI Analysis**
                        - ⚠️ **Overbought**: "RSI above 70 suggests the stock may be overbought and could face resistance"
                        - 🟢 **Oversold**: "RSI below 30 suggests the stock may be oversold and could bounce back"
                        - 📊 **Neutral RSI**: "RSI in neutral territory, no extreme overbought/oversold conditions"
                        
                        **Momentum Analysis**
                        - 🚀 **Positive Momentum**: "Both 5-day and 10-day momentum are positive, indicating upward price movement"
                        - 📉 **Negative Momentum**: "Both 5-day and 10-day momentum are negative, indicating downward pressure"
                        - 📊 **Mixed Momentum**: "Short-term and medium-term momentum are mixed, suggesting consolidation"
                        """)
                    
                    with st.expander("💡 Practical Investment Strategy", expanded=False):
                        st.markdown("""
                        **For Buying (Entry Points)**
                        1. **Wait for resistance break** with high volume
                        2. **Buy on pullbacks** to support level
                        3. **Look for oversold RSI** (<30) for bounce-back opportunities
                        4. **Confirm with positive momentum** (5d and 10d both positive)
                        
                        **For Selling (Exit Points)**
                        1. **Sell on resistance rejection** (price fails to break above resistance)
                        2. **Exit on support break** (price closes below support)
                        3. **Take profits on overbought RSI** (>70)
                        4. **Watch for negative momentum** (both 5d and 10d negative)
                        
                        **Risk Management**
                        - **Set stop-loss** below support level
                        - **Take partial profits** at resistance levels
                        - **Don't chase overbought stocks** (RSI >70)
                        - **Use volume confirmation** for major moves
                        """)
                    
        else:
            st.warning(f"Could not create chart for {selected_ticker}. Insufficient historical data.")
                
    except Exception as e:
        try:
            st.error(f"❌ Error loading historical price data at step '{step_ctx}': {e}")
        except Exception:
            print(f"[UI ERROR] price data load failed at step '{step_ctx}': {e}")
        st.info("💡 Make sure the stock price data file is available in the data folder.")
    
    # --- Visualizations ---
    st.header("📈 Portfolio Analysis")
    
    # Create tabs for different visualizations (Momentum Winners moved to its own page)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Rankings", "🎯 Risk vs Return", "📈 Technical Signals", "🏢 Sector Analysis"])
    
    with tab1:
        # Stock rankings chart (exclude low-priced stocks < $3)
        _chart_df = st.session_state.top_stocks_df.copy()
        # pick y-column dynamically
        y_col = 'predicted_return_pct'
        if y_col not in _chart_df.columns:
            y_col = 'xgb_predicted_return_pct' if 'xgb_predicted_return_pct' in _chart_df.columns else (
                'lstm_predicted_return_pct' if 'lstm_predicted_return_pct' in _chart_df.columns else None)
        # As a last resort, synthesize predicted_return_pct from available columns
        if y_col is None:
            try:
                import numpy as _np
                _cands = [
                    'predicted_return_pct', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct',
                    'xgb_pred', 'lstm_pred'
                ]
                _present = [c for c in _cands if c in _chart_df.columns]
                if _present:
                    _vals = [pd.to_numeric(_chart_df[c], errors='coerce') for c in _present]
                    _stack = _np.vstack([s.fillna(_np.nan).to_numpy() for s in _vals])
                    _pred = _np.nanmean(_stack, axis=0)
                    with _np.errstate(invalid='ignore'):
                        _med = _np.nanmedian(_np.abs(_pred))
                    if not (_med == _med) or _med <= 1:
                        _pred = _pred * 100.0
                    _chart_df['predicted_return_pct'] = _pred
                    y_col = 'predicted_return_pct'
            except Exception:
                pass
        # If still none, fall back to ensemble_score so we can render a chart
        if y_col is None and 'ensemble_score' in _chart_df.columns:
            y_col = 'ensemble_score'
        if y_col is not None:
            try:
                if 'close' in _chart_df.columns:
                    _chart_df = _chart_df[pd.to_numeric(_chart_df['close'], errors='coerce') >= 3]
            except Exception:
                pass
            import plotly.express as px
            fig = px.bar(
                _chart_df.head(num_stocks),
                x='ticker',
                y=y_col,
                color='confidence_score' if 'confidence_score' in _chart_df.columns else None,
                title="Top Stocks by Predicted Return",
                labels={y_col: 'Predicted Return (%)', 'confidence_score': 'Confidence Score'}
            )
            fig.update_layout(title=f"Top {num_stocks} Stocks by Predicted Return")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No predicted return column available to plot.')
    
    with tab2:
        # Risk vs Return scatter plot (robust column selection)
        _df_plot = st.session_state.top_stocks_df.copy()
        # Choose axes dynamically
        x_col = 'risk_score' if 'risk_score' in _df_plot.columns else ('lstm_risk_score' if 'lstm_risk_score' in _df_plot.columns else None)
        y_col = 'predicted_return_pct'
        if y_col not in _df_plot.columns:
            y_col = 'xgb_predicted_return_pct' if 'xgb_predicted_return_pct' in _df_plot.columns else (
                'lstm_predicted_return_pct' if 'lstm_predicted_return_pct' in _df_plot.columns else (
                    'lstm_pred' if 'lstm_pred' in _df_plot.columns else None))
        # Synthesize predicted_return_pct if still missing
        if y_col is None:
            try:
                import numpy as _np
                _cands = [
                    'predicted_return_pct', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct',
                    'xgb_pred', 'lstm_pred'
                ]
                _present = [c for c in _cands if c in _df_plot.columns]
                if _present:
                    _vals = [pd.to_numeric(_df_plot[c], errors='coerce') for c in _present]
                    _stack = _np.vstack([s.fillna(_np.nan).to_numpy() for s in _vals])
                    _pred = _np.nanmean(_stack, axis=0)
                    with _np.errstate(invalid='ignore'):
                        _med = _np.nanmedian(_np.abs(_pred))
                    if not (_med == _med) or _med <= 1:
                        _pred = _pred * 100.0
                    _df_plot['predicted_return_pct'] = _pred
                    y_col = 'predicted_return_pct'
            except Exception:
                pass
        # If still missing, fall back to ensemble_score
        if y_col is None and 'ensemble_score' in _df_plot.columns:
            y_col = 'ensemble_score'
        color_col = 'composite_score' if 'composite_score' in _df_plot.columns else ('lstm_composite_score' if 'lstm_composite_score' in _df_plot.columns else None)
        # Marker size from confidence where available
        if 'confidence_score' in _df_plot.columns:
            _df_plot['confidence_size'] = _df_plot['confidence_score'].abs()
        elif 'lstm_confidence_score' in _df_plot.columns:
            _df_plot['confidence_size'] = _df_plot['lstm_confidence_score'].abs()
        else:
            _df_plot['confidence_size'] = 0.0
        if x_col is None or y_col is None:
            st.info('Not enough fields to plot Risk vs Return (missing risk or predicted return column).')
        else:
            import plotly.express as px
            labels_map = {x_col: 'Risk Score', y_col: 'Predicted Return (%)', 'confidence_size': 'Confidence (|score|)'}
            if color_col:
                labels_map[color_col] = 'Composite Score'
            fig = px.scatter(
                _df_plot,
                x=x_col,
                y=y_col,
                size='confidence_size',
                color=color_col,
                hover_data=['ticker'],
                title="Risk vs Return Analysis",
                labels=labels_map
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Technical signals heatmap
        if len(st.session_state.top_stocks_df) > 0:
            tech_signals = ['broke_resistance', 'post_earnings_dip_rally', 'strong_momentum', 
                           'breakout_confirmed', 'earnings_in_3_weeks', 'last_2q_positive_surprises']
            
            signal_data = []
            for _, stock in st.session_state.top_stocks_df.iterrows():
                for signal in tech_signals:
                    if signal in stock:
                        signal_data.append({
                            'ticker': stock['ticker'],
                            'signal': signal.replace('_', ' ').title(),
                            'value': 1 if stock[signal] else 0
                        })
            
            if signal_data:
                signal_df = pd.DataFrame(signal_data)
                
                # Handle duplicate entries by aggregating values
                try:
                    # Group by ticker and signal, then take the max value (in case of duplicates)
                    signal_df_agg = signal_df.groupby(['ticker', 'signal'])['value'].max().reset_index()
                    
                    # Create pivot table
                    pivot_df = signal_df_agg.pivot(index='ticker', columns='signal', values='value')
                    
                    # Fill NaN values with 0
                    pivot_df = pivot_df.fillna(0)
                    
                    fig = px.imshow(
                        pivot_df,
                        title="Technical Signals Heatmap",
                        color_continuous_scale='RdYlGn',
                        aspect='auto'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not create heatmap due to data structure: {str(e)}")
                    # Fallback: show a simple table
                    st.write("**Technical Signals Summary:**")
                    signal_summary = st.session_state.top_stocks_df[tech_signals].sum()
                    st.write(signal_summary)
    
    with tab4:
        # Sector analysis
        if 'sector' in st.session_state.top_stocks_df.columns:
            sector_counts = st.session_state.top_stocks_df['sector'].value_counts()
            fig = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title="Portfolio Sector Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector information not available in the current dataset.")
    

    
    # --- Stock Table ---
    st.header("📋 Detailed Stock Recommendations")
    
    # Format the dataframe for better readability
    df_to_display = st.session_state.top_stocks_df.copy()
    # Optionally merge external model outputs (XGB weekly, LSTM predictions, Ensemble)
    try:
        import os as _os
        import pandas as _pd
        def _read_if_exists(path):
            try:
                return _pd.read_csv(path) if _os.path.exists(path) else None
            except Exception:
                return None
        # If a prebuilt UI dataset exists, prefer it as the display source
        ui_ds = _read_if_exists('data/top/ui_dataset.csv')
        used_ui_ds = ui_ds is not None and ('ticker' in ui_ds.columns)
        # Prefer merged weekly XGB output if present, else raw ranked (skip if using ui_ds)
        xgb_weekly = _read_if_exists('data/top/xgb_weekly_output.csv')
        use_weekly = (xgb_weekly is not None and 'ticker' in xgb_weekly.columns) and (not used_ui_ds)
        xgb_csv = _read_if_exists('data/top/xgb_ranked_output.csv') or _read_if_exists('data/top/xgb_ranked.csv')
        lstm_csv = _read_if_exists('data/top/lstm_weekly_predictions_output.csv') or _read_if_exists('data/top/lstm_weekly_predictions.csv')
        ens_csv = _read_if_exists('data/top/ensemble_scores_output.csv') or _read_if_exists('data/top/final_ensemble_scores.csv')
        if used_ui_ds:
            df_to_display = ui_ds.copy()
            df_to_display['ticker'] = df_to_display['ticker'].astype(str).str.upper()
        elif use_weekly:
            df_to_display = xgb_weekly.copy()
            df_to_display['ticker'] = df_to_display['ticker'].astype(str).str.upper()
        else:
            # Fallback to building from session state + ranked + features
            # Prefer latest per-ticker features for clean 1:1 merges; fallback to full input
            xgb_feat_csv = _read_if_exists('data/top/xgb_features_latest.csv') or _read_if_exists('data/top/xgb_features_input.csv')
            # Prepare uppercase ticker keys
            df_to_display['ticker'] = df_to_display['ticker'].astype(str).str.upper()
            if xgb_csv is not None and 'ticker' in xgb_csv.columns:
                xgb_csv['ticker'] = xgb_csv['ticker'].astype(str).str.upper()
                # Disambiguate columns to avoid overwriting core fields
                xgb_ren = {}
                if 'confidence_score' in xgb_csv.columns:
                    xgb_ren['confidence_score'] = 'xgb_confidence_score'
                if 'predicted_return_pct' in xgb_csv.columns:
                    xgb_ren['predicted_return_pct'] = 'xgb_predicted_return_pct'
                xgb_csv = xgb_csv.rename(columns=xgb_ren)
                df_to_display = df_to_display.merge(xgb_csv, on='ticker', how='left')
            # Merge XGB feature inputs to expose all engineered columns in UI
            if xgb_feat_csv is not None and 'ticker' in xgb_feat_csv.columns:
                xgb_feat_csv['ticker'] = xgb_feat_csv['ticker'].astype(str).str.upper()
                # If multiple dates exist, keep the latest per ticker to avoid row explosion
                if 'date' in xgb_feat_csv.columns:
                    try:
                        xgb_feat_csv['date'] = _pd.to_datetime(xgb_feat_csv['date'], errors='coerce')
                        xgb_feat_csv = xgb_feat_csv.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker'], keep='last')
                    except Exception:
                        pass
                extra_cols = [c for c in xgb_feat_csv.columns if c != 'ticker' and c not in df_to_display.columns]
                if extra_cols:
                    df_to_display = df_to_display.merge(xgb_feat_csv[['ticker'] + extra_cols], on='ticker', how='left')
                    try:
                        st.caption(f"Merged XGB feature columns added: {len(extra_cols)}")
                    except Exception:
                        pass
                # Backfill overlapping columns (e.g., rsi_14d) where current values are NaN
                overlap_cols = [c for c in xgb_feat_csv.columns if c != 'ticker' and c in df_to_display.columns]
                if overlap_cols:
                    feat_subset = xgb_feat_csv[['ticker'] + overlap_cols]
                    df_to_display = df_to_display.merge(feat_subset, on='ticker', how='left', suffixes=("", "_feat"))
                    filled = 0
                    for col in overlap_cols:
                        feat_col = f"{col}_feat"
                        if feat_col in df_to_display.columns:
                            try:
                                before_na = df_to_display[col].isna().sum()
                                df_to_display[col] = df_to_display[col].combine_first(df_to_display[feat_col])
                                after_na = df_to_display[col].isna().sum()
                                filled += max(0, before_na - after_na)
                            except Exception:
                                pass
                            try:
                                df_to_display.drop(columns=[feat_col], inplace=True)
                            except Exception:
                                pass
                    try:
                        if filled > 0:
                            st.caption(f"Backfilled {filled} missing values from XGB features (RSI/Momentum/Volume/etc.)")
                    except Exception:
                        pass
        if lstm_csv is not None and 'ticker' in lstm_csv.columns:
            lstm_csv['ticker'] = lstm_csv['ticker'].astype(str).str.upper()
            df_to_display = df_to_display.merge(lstm_csv, on='ticker', how='left')
        if ens_csv is not None and 'ticker' in ens_csv.columns:
            # Normalize ensemble
            ens_csv['ticker'] = ens_csv['ticker'].astype(str).str.upper().str.strip()
            ens_csv['ensemble_score'] = pd.to_numeric(ens_csv['ensemble_score'], errors='coerce')
            ens_csv = ens_csv.dropna(subset=['ensemble_score'])
            # Use ensemble file AS the source of truth (already top-25)
            ens_top = ens_csv[['ticker','ensemble_score']].copy().sort_values('ensemble_score', ascending=False).reset_index(drop=True)
            ens_top['ensemble_rank'] = ens_top.index + 1

            # Choose a feature-rich source to join details: prefer xgb_features_latest, fallback to featured_stocks_top, then weekly
            feat_all = None
            try:
                import pandas as _pd
                _x = _read_if_exists('data/top/xgb_features_latest.csv')
                _y = _read_if_exists('data/featured_stocks_top.csv')
                _z = _read_if_exists('data/top/xgb_weekly_output.csv')
                feat_all = _x if (_x is not None and 'ticker' in _x.columns) else (_y if (_y is not None and 'ticker' in _y.columns) else _z)
            except Exception:
                feat_all = None
            if feat_all is None:
                feat_all = df_to_display.copy()

            # Normalize features and collapse to latest per ticker if needed
            try:
                feat_all['ticker'] = feat_all['ticker'].astype(str).str.upper().str.strip()
                if 'date' in feat_all.columns:
                    try:
                        feat_all['date'] = _pd.to_datetime(feat_all['date'], errors='coerce')
                        feat_all = feat_all.sort_values(['ticker','date']).drop_duplicates(subset=['ticker'], keep='last')
                    except Exception:
                        pass
            except Exception:
                pass

            # INNER JOIN exact ensemble set with features and strictly preserve ensemble order
            try:
                df_to_display = ens_top.merge(
                    feat_all.drop(columns=['ensemble_score'], errors='ignore'),
                    on='ticker', how='inner'
                ).sort_values('ensemble_rank')
                # Lock order by setting index
                try:
                    df_to_display = df_to_display.set_index('ensemble_rank', drop=False)
                except Exception:
                    pass
            except Exception:
                # Fallback: left join
                df_to_display = ens_top.merge(
                    feat_all.drop(columns=['ensemble_score'], errors='ignore'),
                    on='ticker', how='left'
                ).sort_values('ensemble_rank')
                try:
                    df_to_display = df_to_display.set_index('ensemble_rank', drop=False)
                except Exception:
                    pass

            # Debug: show orders UI is using vs ensemble
            try:
                st.caption("Ensemble source: data/top/ensemble_scores_output.csv")
                st.text("Ensemble top 10: " + ", ".join(ens_top['ticker'].head(10).tolist()))
                st.text("Displayed top 10: " + ", ".join(df_to_display['ticker'].head(10).tolist()))
            except Exception:
                pass
    except Exception:
        pass
    
    # Enforce final ordering strictly by ensemble_rank if present; otherwise fall back
    try:
        if 'ensemble_rank' in df_to_display.columns:
            df_to_display = df_to_display.sort_values('ensemble_rank')
            try:
                df_to_display = df_to_display.set_index('ensemble_rank', drop=False)
            except Exception:
                pass
        elif 'ensemble_score' in df_to_display.columns:
            df_to_display['__ens__'] = pd.to_numeric(df_to_display['ensemble_score'], errors='coerce')
            df_to_display = df_to_display.sort_values('__ens__', ascending=False, na_position='last').drop(columns=['__ens__'])
        elif 'composite_score' in st.session_state.top_stocks_df.columns:
            order = st.session_state.top_stocks_df.sort_values('composite_score', ascending=False)['ticker'].astype(str).str.upper().tolist()
            df_to_display['__order__'] = pd.Categorical(df_to_display['ticker'].astype(str).str.upper(), categories=order, ordered=True)
            df_to_display = df_to_display.sort_values('__order__').drop(columns=['__order__'])
    except Exception:
        pass
    
    # Format numeric columns (only if present and numeric)
    if 'close' in df_to_display.columns:
        # Keep close numeric for downstream computations; format only at display-time where needed
        df_to_display['close'] = pd.to_numeric(df_to_display['close'].astype(str).str.replace(r'[^0-9.+-]','', regex=True), errors='coerce')
    for col in ['predicted_change','predicted_return_pct','xgb_predicted_return_pct','lstm_predicted_return_pct','ensemble_score','confidence_score','xgb_confidence_score','risk_score','composite_score']:
        if col in df_to_display.columns:
            if col in ['predicted_change']:
                df_to_display[col] = df_to_display[col].apply(lambda x: f"${float(x):+.2f}" if pd.notna(pd.to_numeric(str(x).replace('$',''), errors='coerce')) else str(x))
            elif col in ['ensemble_score']:
                df_to_display[col] = df_to_display[col].apply(lambda x: f"{float(x):+.3f}" if pd.notna(pd.to_numeric(str(x), errors='coerce')) else str(x))
            elif col.endswith('_predicted_return_pct') or col == 'predicted_return_pct':
                df_to_display[col] = df_to_display[col].apply(lambda x: f"{float(str(x).replace('%','')):+.2f}%" if pd.notna(pd.to_numeric(str(x).replace('%',''), errors='coerce')) else str(x))
            elif col.endswith('confidence_score'):
                df_to_display[col] = df_to_display[col].apply(lambda x: f"{float(x):.1f}" if pd.notna(pd.to_numeric(str(x), errors='coerce')) else str(x))
            elif col == 'risk_score' or col == 'composite_score':
                df_to_display[col] = df_to_display[col].apply(lambda x: f"{float(x):.1f}" if pd.notna(pd.to_numeric(str(x), errors='coerce')) else str(x))

    # Highlight rows also present in Momentum page (green background)
    try:
        momentum_path = os.getenv('FEATURED_STOCKS_MOMENTUM_CSV', 'data/momentum/featured_stocks_momentum.csv')
        mom_df = pd.read_csv(momentum_path)
        # Build winners like the Momentum page
        for col in ['momentum_30d','momentum_60d','trend_slope_15d','ma_20','macd','macd_signal','close']:
            if col in mom_df.columns:
                mom_df[col] = pd.to_numeric(mom_df[col], errors='coerce')
        if 'momentum_winner' in mom_df.columns and mom_df['momentum_winner'].notna().any():
            mask = mom_df['momentum_winner'] == True
        else:
            mask = (
                ((mom_df.get('momentum_60d', 0) > 0.5) | (mom_df.get('momentum_30d', 0) > 0.2)) &
                (mom_df.get('trend_slope_15d', 0) > 0) &
                (mom_df.get('close', 0) > mom_df.get('ma_20', 0)) &
                (mom_df.get('macd', 0) > mom_df.get('macd_signal', 0))
            )
        winners = mom_df[mask].copy()
        # Score and take top 20
        if not winners.empty:
            def _mw_score(row):
                m60 = row.get('momentum_60d', 0) if pd.notna(row.get('momentum_60d', 0)) else 0
                m30 = row.get('momentum_30d', 0) if pd.notna(row.get('momentum_30d', 0)) else 0
                slope30 = row.get('trend_slope_30d', 0) if 'trend_slope_30d' in winners.columns else 0
                slope30 = slope30 if pd.notna(slope30) else 0
                return 0.6*m60 + 0.3*m30 + 0.1*slope30
            winners['mw_score'] = winners.apply(_mw_score, axis=1)
            winners = winners.sort_values('mw_score', ascending=False).head(20)
        momentum_tickers = set(winners['ticker'].astype(str).str.upper()) if not winners.empty else set()
    except Exception:
        momentum_tickers = set()

    def _highlight_momentum(row):
        t = str(row.get('ticker', '')).upper()
        color = '#e6ffed' if t in momentum_tickers else ''
        return [f'background-color: {color}'] * len(row)

    try:
        styled = df_to_display.style.apply(_highlight_momentum, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        # Fallback without styling
        st.dataframe(df_to_display, use_container_width=True, hide_index=True)
    
    # --- Export Options ---
    st.header("💾 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        # Export the currently displayed (ensemble-ordered) table
        try:
            export_df = df_to_display.copy()
        except Exception:
            export_df = st.session_state.top_stocks_df.copy()
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"top_stocks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Portfolio summary (robust to formatted strings like "+0.64%")
        def _num(series):
            import pandas as _pd
            if series is None:
                return _pd.Series([], dtype=float)
            return _pd.to_numeric(_pd.Series(series).astype(str).str.replace(r'[^0-9.+-]+','', regex=True), errors='coerce')

        avg_return_val = _num(export_df.get('predicted_return_pct')).mean()
        avg_conf_val = _num(export_df.get('confidence_score')).mean()
        avg_risk_val = _num(export_df.get('risk_score')).mean()

        summary = f"""
Portfolio Summary - Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Total Stocks: {len(export_df)}
Average Predicted Return: {0.0 if pd.isna(avg_return_val) else avg_return_val:.2f}%
Average Confidence Score: {0.0 if pd.isna(avg_conf_val) else avg_conf_val:.1f}
Average Risk Score: {0.0 if pd.isna(avg_risk_val) else avg_risk_val:.1f}

Top 5 Picks:
"""
        for i, (_, stock) in enumerate(export_df.head().iterrows(), 1):
            pr = str(stock.get('predicted_return_pct',''))
            cs = str(stock.get('confidence_score',''))
            # Strip symbols and coerce
            try:
                import re
                pr_val = float(re.sub(r"[^0-9.+-]", "", pr)) if pr != '' else float('nan')
            except Exception:
                pr_val = float('nan')
            try:
                import re
                cs_val = float(re.sub(r"[^0-9.+-]", "", cs)) if cs != '' else float('nan')
            except Exception:
                cs_val = float('nan')
            pr_txt = f"{pr_val:.2f}%" if pr_val == pr_val else str(pr)
            cs_txt = f"{cs_val:.1f}" if cs_val == cs_val else str(cs)
            summary += f"{i}. {stock.get('ticker','')}: {pr_txt} return, {cs_txt} confidence\n"
        
        st.download_button(
            label="📄 Download Summary",
            data=summary,
            file_name=f"portfolio_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # --- Chatbot Interface ---
    st.header("🤖 AI Stock Analyst Chat")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about the stock recommendations, technical analysis, or portfolio insights..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if st.session_state.agent is not None:
                    try:
                        response = st.session_state.agent.invoke({"input": prompt})
                        response_text = response.get('output', "Sorry, I encountered an error.")
                        
                        # Handle empty responses
                        if not response_text or response_text.strip() == "":
                            response_text = "I'm sorry, I couldn't generate a response. Please try asking a different question or rephrase your query."
                        
                        # Add debug info for troubleshooting
                        print(f"Chatbot response: {response_text[:200]}...")
                        
                    except Exception as e:
                        response_text = f"Sorry, I encountered an error: {str(e)}"
                        print(f"Chatbot error: {e}")
                else:
                    response_text = "The analysis must be run first to initialize the chatbot."
                st.markdown(response_text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("🎯 Click the 'Find Top Stocks' button above to run the analysis and see stock recommendations.")
    
    # Show some helpful tips
    st.header("💡 How to Use This Screener")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Filter Settings**")
        st.write("• **Confidence Score**: Higher values = more confident predictions")
        st.write("• **Risk Score**: Lower values = less risky stocks")
        st.write("• **Diversification**: Limits exposure per sector/industry")
        st.write("• **Number of Stocks**: Choose how many recommendations to see")
    
    with col2:
        st.write("**📈 Analysis Features**")
        st.write("• **Technical Indicators**: RSI, MACD, Bollinger Bands, Volume")
        st.write("• **Fundamental Analysis**: Earnings surprises, growth rates")
        st.write("• **Risk Metrics**: Volatility, drawdown, Value at Risk")
        st.write("• **AI Chatbot**: Ask questions about recommendations")

## (Momentum Winners detailed section removed; available on its own page)

# --- Symbol Analysis Section (Always Available) ---
st.header("🔍 Individual Symbol Analysis")
st.write("Enter any stock symbol to analyze its current metrics and understand why it was or wasn't selected as a top pick.")

# Symbol input
col1, col2 = st.columns([2, 1])
with col1:
    symbol_input = st.text_input("Enter Stock Symbol (e.g., GOOGL, AAPL, TSLA):", placeholder="GOOGL", key="symbol_analysis_input")
with col2:
    analyze_button = st.button("🔍 Analyze Symbol", type="primary")

if analyze_button and symbol_input:
    symbol_input = symbol_input.upper().strip()
    
    # Load the featured data to get all available symbols
    try:
        # Try env path, then standardized top location, then legacy default
        env_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
        try_paths = [env_path, 'data/top/featured_stocks_top.csv', 'data/featured_stocks_top.csv']
        picked = None
        for _p in try_paths:
            if os.path.exists(_p):
                picked = _p
                break
        if picked is None:
            raise FileNotFoundError(f"No featured stocks CSV found in {try_paths}")
        featured_data = pd.read_csv(picked)
        symbol_data = featured_data[featured_data['ticker'] == symbol_input]
        
        if symbol_data.empty:
            st.error(f"❌ Symbol '{symbol_input}' not found in the dataset. Available symbols: {', '.join(featured_data['ticker'].unique())}")
        else:
            # Get the symbol data
            stock_data = symbol_data.iloc[0]
            
            # Run the stock selector to get predictions
            try:
                top_stocks = get_top_stocks(n=50, min_confidence=0, max_risk=100, diversify=False)
                symbol_in_top = top_stocks[top_stocks['ticker'] == symbol_input]
                
                st.success(f"✅ Analysis complete for {symbol_input}")
                
                # Display current metrics
                st.subheader(f"📊 Current Metrics for {symbol_input}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${stock_data['close']:.2f}")
                with col2:
                    st.metric("RSI (14d)", f"{stock_data['rsi_14d']:.1f}")
                with col3:
                    st.metric("Volatility (30d)", f"{stock_data['volatility_30d']:.3f}")
                with col4:
                    st.metric("Volume Ratio", f"{stock_data['volume_ratio']:.2f}")
                
                # Display momentum metrics
                st.subheader("📈 Momentum Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("5-Day Momentum", f"{stock_data['momentum_5d']:.3f}")
                with col2:
                    st.metric("10-Day Momentum", f"{stock_data['momentum_10d']:.3f}")
                with col3:
                    st.metric("20-Day Momentum", f"{stock_data['momentum_20d']:.3f}")
                # Extended momentum
                col1, col2, col3 = st.columns(3)
                if 'momentum_30d' in stock_data.index:
                    with col1:
                        st.metric("30-Day Momentum", f"{stock_data['momentum_30d']:.3f}")
                if 'momentum_60d' in stock_data.index:
                    with col2:
                        st.metric("60-Day Momentum", f"{stock_data['momentum_60d']:.3f}")
                if 'up_day_ratio_20d' in stock_data.index:
                    with col3:
                        st.metric("Up-Day Ratio (20d)", f"{stock_data['up_day_ratio_20d']*100:.1f}%")
                
                # Display prediction results
                st.subheader("🎯 Model Prediction Results")
                
                if not symbol_in_top.empty:
                    prediction_data = symbol_in_top.iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Predicted Return", f"{prediction_data['predicted_return_pct']:.2f}%", 
                                delta=f"{prediction_data['predicted_return_pct']:.2f}%")
                    with col2:
                        st.metric("Confidence Score", f"{prediction_data['confidence_score']:.1f}/100")
                    with col3:
                        st.metric("Risk Score", f"{prediction_data['risk_score']:.1f}/100")
                    with col4:
                        st.metric("Composite Score", f"{prediction_data['composite_score']:.1f}")
                    
                    # Show ranking
                    rank = top_stocks[top_stocks['ticker'] == symbol_input].index[0] + 1
                    st.info(f"🏆 **Ranking**: {symbol_input} is ranked #{rank} out of {len(top_stocks)} stocks analyzed")
                    
                else:
                    st.warning(f"⚠️ {symbol_input} was not selected in the top picks analysis")
                    
                    # Show why it wasn't selected
                    st.subheader("🔍 Why Wasn't It Selected?")
                    
                    # Calculate what the prediction would be
                    try:
                        # Get model predictions for this symbol (Top Stocks model)
                        model = joblib.load('models/stock_predictor_top.joblib')
                        # Use the model's expected training columns
                        training_cols = model.get_booster().feature_names
                        # Build a single-row feature frame matching the model schema
                        row_dict = {}
                        for col in training_cols:
                            val = stock_data.get(col, 0)
                            try:
                                val = float(val)
                            except Exception:
                                val = 0.0
                            row_dict[col] = val
                        import pandas as _pd_alias  # local alias to avoid shadowing
                        X_symbol = _pd_alias.DataFrame([row_dict])

                        # Make prediction (percentage over ~5 trading days)
                        predicted_return_pct = float(model.predict(X_symbol)[0])

                        st.metric("Model Prediction", f"{predicted_return_pct:.2f}%",
                                  delta=f"{predicted_return_pct:.2f}%")
                        
                        # Calculate confidence and risk scores
                        confidence_score = calculate_confidence_score(stock_data)
                        risk_score = calculate_risk_score(stock_data)
                        composite_score = (predicted_return_pct * 0.4) + (confidence_score * 0.4) + ((100 - risk_score) * 0.2)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Calculated Confidence", f"{confidence_score:.1f}/100")
                        with col2:
                            st.metric("Calculated Risk", f"{risk_score:.1f}/100")
                        with col3:
                            st.metric("Calculated Composite", f"{composite_score:.1f}")
                        
                    except Exception as e:
                        st.error(f"Could not calculate model prediction: {str(e)}")
                
                # Technical Analysis Summary
                st.subheader("📊 Technical Analysis Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Key Technical Signals:**")
                    signals = []
                    if stock_data.get('broke_resistance', False):
                        signals.append("✅ Broke Resistance")
                    else:
                        signals.append("❌ No Resistance Break")
                    
                    if stock_data.get('post_earnings_dip_rally', False):
                        signals.append("✅ Post-Earnings Dip Rally")
                    else:
                        signals.append("❌ No Post-Earnings Rally")
                    
                    if stock_data.get('strong_momentum', False):
                        signals.append("✅ Strong Momentum")
                    else:
                        signals.append("❌ Weak Momentum")
                    
                    if stock_data.get('breakout_confirmed', False):
                        signals.append("✅ Breakout Confirmed")
                    else:
                        signals.append("❌ No Breakout Confirmation")
                    
                    for signal in signals:
                        st.write(f"• {signal}")
                
                with col2:
                    st.write("**Risk Assessment:**")
                    volatility = stock_data['volatility_30d']
                    if volatility > 0.5:
                        st.write("• ⚠️ High Volatility")
                    elif volatility > 0.3:
                        st.write("• 📊 Moderate Volatility")
                    else:
                        st.write("• ✅ Low Volatility")
                    
                    rsi = stock_data['rsi_14d']
                    if rsi > 70:
                        st.write("• ⚠️ Overbought (RSI > 70)")
                    elif rsi < 30:
                        st.write("• 🟢 Oversold (RSI < 30)")
                    else:
                        st.write("• 📊 Neutral RSI")
                
                # Detailed explanation
                st.subheader("💡 Analysis Explanation")
                
                explanation = f"""
                **Analysis for {symbol_input} (${stock_data['close']:.2f})**
                
                **Current Technical Position:**
                - **RSI**: {stock_data['rsi_14d']:.1f} ({'Overbought' if stock_data['rsi_14d'] > 70 else 'Oversold' if stock_data['rsi_14d'] < 30 else 'Neutral'})
                - **Momentum**: 5d={stock_data['momentum_5d']:.3f}, 10d={stock_data['momentum_10d']:.3f}, 20d={stock_data['momentum_20d']:.3f}
                - **Volatility**: {stock_data['volatility_30d']:.3f} ({'High' if stock_data['volatility_30d'] > 0.5 else 'Moderate' if stock_data['volatility_30d'] > 0.3 else 'Low'})
                - **Volume**: {stock_data['volume_ratio']:.2f}x average ({'Above' if stock_data['volume_ratio'] > 1.2 else 'Below' if stock_data['volume_ratio'] < 0.8 else 'Normal'} average)
                
                **Key Signals:**
                - **Resistance Break**: {'✅ Yes' if stock_data.get('broke_resistance', False) else '❌ No'}
                - **Strong Momentum**: {'✅ Yes' if stock_data.get('strong_momentum', False) else '❌ No'}
                - **Breakout Confirmed**: {'✅ Yes' if stock_data.get('breakout_confirmed', False) else '❌ No'}
                
                **Why This Matters:**
                The model looks for stocks with strong technical signals, positive momentum, and confirmed breakouts. 
                Stocks with low momentum, no resistance breaks, or poor technical signals typically score lower 
                in the confidence and composite scoring system.
                """
                
                st.markdown(explanation)
                
            except Exception as e:
                st.error(f"Error analyzing symbol: {str(e)}")
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

elif analyze_button and not symbol_input:
    st.warning("Please enter a stock symbol to analyze.")