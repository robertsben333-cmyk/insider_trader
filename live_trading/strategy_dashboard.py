from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from dotenv import load_dotenv

from live_trading.dashboard_service import StrategyDashboardSnapshot, build_dashboard_service
from live_trading.strategy_settings import ACTIVE_STRATEGY, DASHBOARD_CONFIG

try:
    import streamlit as st
    from streamlit.components.v1 import html as st_html
except ImportError:  # pragma: no cover - runtime dependency
    st = None
    st_html = None


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value:,.2f}%"


def _portfolio_frame(snapshot: StrategyDashboardSnapshot) -> pd.DataFrame:
    rows = []
    for row in snapshot.portfolio_rows:
        rows.append(
            {
                "symbol": row.symbol,
                "quantity": row.quantity,
                "avg_cost": round(row.avg_cost, 4),
                "ibkr_avg_cost": round(row.broker_avg_cost, 4),
                "mark_price": round(row.mark_price, 4),
                "market_value": round(row.market_value, 2),
                "unrealized_pnl": round(row.unrealized_pnl, 2),
                "last_updated_at": row.last_updated_at,
            }
        )
    return pd.DataFrame(rows)


def _trade_frame(snapshot: StrategyDashboardSnapshot) -> pd.DataFrame:
    rows = [asdict(row) for row in snapshot.trade_rows]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for col in ["price", "gross_notional", "commission", "net_cash_flow"]:
        frame[col] = frame[col].round(2)
    return frame


def _order_frame(snapshot: StrategyDashboardSnapshot) -> pd.DataFrame:
    rows = [asdict(row) for row in snapshot.order_rows]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["limit_price"] = frame["limit_price"].round(2)
    return frame


def render_dashboard(snapshot: StrategyDashboardSnapshot, st_module) -> None:
    metrics = snapshot.headline_metrics
    st_module.set_page_config(page_title="IBKR Strategy Dashboard", layout="wide")
    st_module.title("IBKR Strategy Dashboard")
    st_module.caption(
        f"Strategy `{snapshot.strategy_id}` | account `{snapshot.account_id}` | "
        f"baseline `{snapshot.baseline_mode}` | tracking since `{snapshot.baseline_started_at or 'not started'}`"
    )
    st_module.caption(f"Last refresh: `{snapshot.generated_at}`")

    if snapshot.stale_data or snapshot.connection_status != "connected":
        st_module.warning(snapshot.warning_message or "Rendering cached broker-backed snapshot.")
    else:
        st_module.success("Connected to IBKR and rendered a fresh broker-backed snapshot.")

    cards = st_module.columns(6)
    cards[0].metric("Total equity", _format_currency(metrics.total_equity), _format_currency(metrics.total_return_value))
    cards[1].metric("Total return", _format_pct(metrics.total_return_pct))
    cards[2].metric("Realized P&L", _format_currency(metrics.realized_pnl))
    cards[3].metric("Unrealized P&L", _format_currency(metrics.unrealized_pnl))
    cards[4].metric("Cash proxy", _format_currency(metrics.cash_balance))
    cards[5].metric("Open positions", str(metrics.open_positions))

    st_module.subheader("Current portfolio")
    portfolio = _portfolio_frame(snapshot)
    if portfolio.empty:
        st_module.info("No open strategy positions are being tracked.")
    else:
        st_module.dataframe(portfolio, use_container_width=True, hide_index=True)

    st_module.subheader("Recent broker executions")
    trades = _trade_frame(snapshot)
    if trades.empty:
        st_module.info("No broker executions have been captured yet.")
    else:
        st_module.dataframe(trades, use_container_width=True, hide_index=True)

    st_module.subheader("Open orders")
    orders = _order_frame(snapshot)
    if orders.empty:
        st_module.info("No open broker orders.")
    else:
        st_module.dataframe(orders, use_container_width=True, hide_index=True)


def _schedule_auto_refresh(interval_seconds: int) -> None:
    if st_html is None:
        return
    safe_interval = max(5, int(interval_seconds)) * 1000
    st_html(
        f"""
        <script>
        window.setTimeout(function() {{
          window.parent.location.reload();
        }}, {safe_interval});
        </script>
        """,
        height=0,
    )


def main() -> None:
    if st is None:  # pragma: no cover - runtime dependency
        raise RuntimeError("streamlit is not installed. Install requirements.txt before running the dashboard.")

    load_dotenv()
    service = build_dashboard_service()
    if "dashboard_sync_nonce" not in st.session_state:
        st.session_state["dashboard_sync_nonce"] = 0

    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
    refresh_seconds = st.sidebar.number_input(
        "Refresh interval (sec)",
        min_value=5,
        value=int(DASHBOARD_CONFIG.auto_refresh_seconds),
        step=5,
    )
    st.sidebar.caption(f"Default bind target: {DASHBOARD_CONFIG.streamlit_host}:{DASHBOARD_CONFIG.streamlit_port}")
    st.sidebar.caption(f"Active strategy: {ACTIVE_STRATEGY.strategy_id}")
    st.sidebar.caption("Startup and refresh both trigger a broker sync before rendering.")
    if st.sidebar.button("Sync now"):
        st.session_state["dashboard_sync_nonce"] += 1
        st.rerun()

    with st.spinner("Syncing broker state..."):
        snapshot = service.sync_broker_state()
    render_dashboard(snapshot, st)

    if auto_refresh:
        _schedule_auto_refresh(int(refresh_seconds))


if __name__ == "__main__":
    main()
