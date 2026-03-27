"""Streamlit UI for crypto_bot.

Start the bot first (in another terminal):
    python engine.py

Then run UI:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st


LOG_DIR = os.getenv("CRYPTO_BOT_LOG_DIR", "logs")
STATE_PATH = os.path.join(LOG_DIR, "state.json")
CONTROL_PATH = os.path.join(LOG_DIR, "control.json")
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")
PERF_CSV = os.path.join(LOG_DIR, "performance.csv")


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_control(command: str):
    os.makedirs(os.path.dirname(CONTROL_PATH) or ".", exist_ok=True)
    payload = {"command": command, "timestamp": datetime.utcnow().isoformat() + "Z"}
    with open(CONTROL_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


st.set_page_config(page_title="crypto_bot", layout="wide")
st.title("crypto_bot — dashboard")

with st.sidebar:
    st.header("Controls")
    c1, c2, c3 = st.columns(3)
    if c1.button("Pause", use_container_width=True):
        write_control("pause")
    if c2.button("Resume", use_container_width=True):
        write_control("resume")
    if c3.button("Stop", use_container_width=True):
        write_control("stop")

    st.divider()
    auto = st.checkbox("Auto-refresh", value=True)
    refresh_s = st.slider("Refresh (seconds)", 2, 30, 5)


state = load_json(STATE_PATH)

if not state:
    st.warning("No state found at logs/state.json yet. Start the bot: `python engine.py` and refresh.")
    st.code("python engine.py", language="bash")
else:
    # Header
    top = st.columns(4)
    top[0].metric("Mode", state.get("mode"))
    top[1].metric("Timeframe", state.get("timeframe"))
    top[2].metric("Dry-run", str(state.get("dry_run")))
    top[3].metric("Paused", str(state.get("paused")))

    p = state.get("portfolio", {})
    m = st.columns(6)
    m[0].metric("Equity", f"${p.get('equity', 0):,.2f}")
    m[1].metric("Cash", f"${p.get('cash', 0):,.2f}")
    m[2].metric("Drawdown", f"{p.get('drawdown_pct', 0):.2f}%")
    m[3].metric("Win rate", f"{p.get('win_rate', 0) * 100:.0f}%")
    m[4].metric("Profit factor", f"{p.get('profit_factor', 0):.2f}")
    m[5].metric("Circuit breaker", p.get("circuit_breaker", ""))

    if state.get("last_error"):
        st.error(f"Last error: {state['last_error']}")

    st.subheader("Open trades")
    open_trades = state.get("open_trades") or []
    if open_trades:
        st.dataframe(pd.DataFrame(open_trades), use_container_width=True)
    else:
        st.info("No open trades")

    tabs = st.tabs(["Closed trades", "Performance", "Raw state"])
    with tabs[0]:
        if os.path.exists(TRADES_CSV):
            df = pd.read_csv(TRADES_CSV)
            st.dataframe(df.tail(200), use_container_width=True)
        else:
            st.info("No logs/trades.csv yet")

    with tabs[1]:
        if os.path.exists(PERF_CSV):
            dfp = pd.read_csv(PERF_CSV)
            if "timestamp" in dfp.columns:
                dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], errors="coerce")
                dfp = dfp.dropna(subset=["timestamp"]).sort_values("timestamp")
                if "equity" in dfp.columns:
                    st.line_chart(dfp.set_index("timestamp")["equity"])
            st.dataframe(dfp.tail(200), use_container_width=True)
        else:
            st.info("No logs/performance.csv yet")

    with tabs[2]:
        st.json(state)


if auto:
    time.sleep(refresh_s)
    st.rerun()
