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
import plotly.graph_objects as go
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

    tabs = st.tabs(["Charts", "Closed trades", "Performance", "Raw state"])

    with tabs[0]:
        has_perf = os.path.exists(PERF_CSV)
        has_trades = os.path.exists(TRADES_CSV)

        if has_perf:
            dfp = pd.read_csv(PERF_CSV)
            if "timestamp" in dfp.columns:
                dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], errors="coerce")
                dfp = dfp.dropna(subset=["timestamp"]).sort_values("timestamp")

            col_left, col_right = st.columns(2)

            # Equity curve
            if "equity" in dfp.columns and not dfp.empty:
                with col_left:
                    st.subheader("Equity Curve")
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=dfp["timestamp"], y=dfp["equity"],
                        mode="lines", name="Equity",
                        line=dict(color="#00d4aa", width=2),
                        fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
                    ))
                    fig_eq.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=280,
                        xaxis_title=None, yaxis_title="USDT",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ccc"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

            # Drawdown chart
            if "equity" in dfp.columns and "peak_equity" in dfp.columns and not dfp.empty:
                with col_right:
                    st.subheader("Drawdown %")
                    dd = ((dfp["peak_equity"] - dfp["equity"]) / dfp["peak_equity"] * 100) * -1
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=dfp["timestamp"], y=dd,
                        mode="lines", name="Drawdown",
                        line=dict(color="#ff4b4b", width=2),
                        fill="tozeroy", fillcolor="rgba(255,75,75,0.15)",
                    ))
                    fig_dd.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=280,
                        xaxis_title=None, yaxis_title="%",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ccc"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("No logs/performance.csv yet — equity charts will appear once the bot runs.")

        if has_trades:
            dft = pd.read_csv(TRADES_CSV)

            col_left2, col_right2 = st.columns(2)

            # P&L per trade bar chart
            if "net_pnl" in dft.columns and not dft.empty:
                with col_left2:
                    st.subheader("P&L per Trade")
                    colors = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in dft["net_pnl"]]
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Bar(
                        x=list(range(len(dft))), y=dft["net_pnl"],
                        marker_color=colors, name="Net P&L",
                    ))
                    fig_pnl.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=280,
                        xaxis_title="Trade #", yaxis_title="USDT",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ccc"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

            # Win / Loss pie
            if "net_pnl" in dft.columns and not dft.empty:
                with col_right2:
                    st.subheader("Win / Loss Split")
                    wins = (dft["net_pnl"] >= 0).sum()
                    losses = (dft["net_pnl"] < 0).sum()
                    fig_pie = go.Figure(go.Pie(
                        labels=["Wins", "Losses"],
                        values=[wins, losses],
                        marker=dict(colors=["#00d4aa", "#ff4b4b"]),
                        hole=0.4,
                    ))
                    fig_pie.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ccc"),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No logs/trades.csv yet — trade charts will appear once trades close.")

    with tabs[1]:
        if os.path.exists(TRADES_CSV):
            df = pd.read_csv(TRADES_CSV)
            st.dataframe(df.tail(200), use_container_width=True)
        else:
            st.info("No logs/trades.csv yet")

    with tabs[2]:
        if os.path.exists(PERF_CSV):
            dfp = pd.read_csv(PERF_CSV)
            st.dataframe(dfp.tail(200), use_container_width=True)
        else:
            st.info("No logs/performance.csv yet")

    with tabs[3]:
        st.json(state)


if auto:
    time.sleep(refresh_s)
    st.rerun()
