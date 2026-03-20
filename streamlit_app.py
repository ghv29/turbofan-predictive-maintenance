from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.workbench import charts, fleet, loader, advisor
from src.workbench.advisor import ChatMessage, build_initial_messages, call_grok_api
from src.workbench.fleet import compute_fleet_latest_snapshot

from src.workbench.predictor import rul_trajectory_for_engine, shap_for_single_row
from src.workbench.utils import health_color

st.set_page_config(
    page_title="Turbofan Fleet Health",
    layout="wide",
    page_icon="🛠️",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    st.session_state.setdefault("selected_dataset", "FD001")
    st.session_state.setdefault("selected_engine_id", None)
    st.session_state.setdefault("selected_cycle", None)
    st.session_state.setdefault("chat_history", [])  # list[ChatMessage-like dict]


def _sidebar(fleet_summary: fleet.FleetSummary, latest_snapshot: pd.DataFrame) -> str:
    with st.sidebar:
        st.markdown("### Turbofan RUL Workbench")
        st.markdown("**Author:** GHV")
        st.markdown("[GitHub: ghv29/turbofan-predictive-maintenance](https://github.com/ghv29/turbofan-predictive-maintenance)")

        st.caption("Random Forest (GridSearchCV) — RMSE 15.51 | MAE 10.89 | R² 0.858")

        st.markdown("---")
        page = st.radio(
            "Section",
            options=["Fleet Dashboard", "Engine Deep Dive", "AI Advisor"],
            index=0,
        )

        st.markdown("---")
        st.markdown("#### Fleet snapshot")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total engines", fleet_summary.total_engines)
            st.metric("Critical", fleet_summary.critical_count)
        with c2:
            st.metric("Warning", fleet_summary.warning_count)
            st.metric("Healthy", fleet_summary.healthy_count)

        st.markdown("---")
        st.markdown("#### Select engine")
        datasets = sorted(latest_snapshot["dataset"].unique())
        dataset = st.selectbox("Dataset", datasets, index=0, key="sidebar_dataset")
        subset = latest_snapshot[latest_snapshot["dataset"] == dataset]
        engine_ids = sorted(subset["engine_id"].unique().tolist())
        engine_id = st.selectbox("Engine ID", engine_ids, key="sidebar_engine")

        # Keep in session state for cross-page navigation
        st.session_state["selected_dataset"] = dataset
        st.session_state["selected_engine_id"] = int(engine_id)

        if st.button("Go to Deep Dive"):
            st.session_state["active_page"] = "Engine Deep Dive"

        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state["chat_history"] = []

        return page


def _fleet_dashboard(latest_snapshot: pd.DataFrame, summary: fleet.FleetSummary) -> None:
    st.markdown("### Fleet Health Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total engines", summary.total_engines)
    with c2:
        st.metric("Critical", summary.critical_count)
    with c3:
        st.metric("Warning", summary.warning_count)
    with c4:
        st.metric("Healthy", summary.healthy_count)

    st.markdown("#### Fleet health grid")
    fig_grid = charts.fleet_health_grid(latest_snapshot)
    st.plotly_chart(fig_grid, use_container_width=True)

    st.markdown("#### Dataset comparison")
    ds_stats = fleet.dataset_level_stats(latest_snapshot)
    fig_bar = charts.dataset_comparison_bar(ds_stats)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### RUL distribution")
    ruls = fleet.rul_distribution(latest_snapshot)
    fig_hist = charts.rul_distribution_hist(ruls)
    st.plotly_chart(fig_hist, use_container_width=True)


def _engine_deep_dive(latest_snapshot: pd.DataFrame) -> None:
    st.markdown("### Engine Deep Dive")

    ds = st.session_state.get("selected_dataset")
    engine_id = st.session_state.get("selected_engine_id")

    if ds is None or engine_id is None:
        st.info("Select an engine in the sidebar to begin.")
        return

    latest_row = latest_snapshot[
        (latest_snapshot["dataset"] == ds) & (latest_snapshot["engine_id"] == engine_id)
    ]
    if latest_row.empty:
        st.warning("Selected engine not found in latest snapshot.")
        return

    latest_row = latest_row.iloc[0]
    current_cycle = int(latest_row["cycle"])
    predicted_rul = float(latest_row["predicted_RUL"])
    status = str(latest_row["health_status"])

    st.session_state["selected_cycle"] = current_cycle

    # Header
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Engine ID", engine_id)
    with c2:
        st.metric("Dataset", ds)
    with c3:
        st.metric("Current cycle", current_cycle)
    with c4:
        st.metric("Predicted RUL (cycles)", f"{predicted_rul:.1f}")

    badge_color = health_color(status)
    st.markdown(
        f"<div style='padding:0.5rem 1rem; border-radius:0.75rem; "
        f"background-color:{badge_color}; color:black; font-weight:bold; "
        f"width: fit-content;'>"
        f"{status}</div>",
        unsafe_allow_html=True,
    )

    # Degradation timeline
    st.markdown("#### Degradation timeline (sensors + predicted RUL)")
    try:
        traj = rul_trajectory_for_engine(ds, engine_id)
        # Extract top 3 SHAP features for the current row and then build their history
        shap_vals = shap_for_single_row(ds, engine_id, current_cycle)
        top3 = [name for name, _ in sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]]

        engineered = loader.load_engineered_dataset(ds)
        history = engineered[engineered["engine_id"] == engine_id].copy()
        df_top = history[["cycle"] + top3]

        fig_timeline = charts.degradation_timeline(df_top, traj, current_cycle)
        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not compute degradation timeline for this engine: {exc}")

    # SHAP explanation
    st.markdown("#### Why was this RUL predicted?")
    try:
        shap_vals = shap_for_single_row(ds, engine_id, current_cycle)
        fig_shap = charts.shap_top_features_bar(shap_vals, top_n=10)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "Sensor labels are anonymized per NASA CMAPSS dataset convention. "
            "s4_rollmean is the dominant degradation signal globally (59.6% feature importance)."
        )
    except Exception as exc:
        st.error(f"Could not compute SHAP explanation for this engine: {exc}")




def _ai_advisor(fleet_summary: fleet.FleetSummary, latest_snapshot: pd.DataFrame) -> None:
    st.markdown("### AI Advisor")
    st.caption("Ask questions in natural language about the fleet or a specific engine.")

    selected_engine: Optional[Dict[str, Any]] = None
    ds = st.session_state.get("selected_dataset")
    engine_id = st.session_state.get("selected_engine_id")
    if ds is not None and engine_id is not None:
        row = latest_snapshot[
            (latest_snapshot["dataset"] == ds) & (latest_snapshot["engine_id"] == engine_id)
        ]
        if not row.empty:
            r = row.iloc[0]
            selected_engine = {
                "dataset": ds,
                "engine_id": int(engine_id),
                "cycle": int(r["cycle"]),
                "predicted_RUL": float(r["predicted_RUL"]),
                "health_status": str(r["health_status"]),
            }

    if not st.session_state["chat_history"]:
        # Show starter suggestions
        st.markdown("Try one of these questions:")
        st.markdown(
            "- Which engines need maintenance this week?\n"
            "- Why is Engine X flagged as critical?\n"
            "- What's the risk of delaying maintenance by 7 cycles?\n"
            "- Compare FD001 and FD003 fleet health"
        )

    user_input = st.chat_input("Ask the AI advisor...")

    # Initialize messages if needed
    if not st.session_state["chat_history"]:
        initial = build_initial_messages(fleet_summary, latest_snapshot, selected_engine)
        st.session_state["chat_history"] = [m.__dict__ for m in initial]

    # Render history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    tool_ctx: Dict[str, Any] = {}
                    try:
                        tool_ctx = advisor.build_tool_context_for_question(
                            user_input=user_input,
                            fleet_summary=fleet_summary,
                            latest_snapshot=latest_snapshot,
                            selected_engine=selected_engine,
                        )
                    except Exception:
                        tool_ctx = {}

                    tools_used = tool_ctx.get("tools_used", [])
                    tool_context_text = tool_ctx.get("context", "")

                    messages = [ChatMessage(**m) for m in st.session_state["chat_history"]]
                    if tool_context_text:
                        # Append as the last system message so it overrides any stale
                        # selected-engine context that may be present earlier in history.
                        messages.append(ChatMessage(role="system", content=tool_context_text))

                    answer = call_grok_api(messages)
                except Exception as exc:
                    answer = f"AI advisor is currently unavailable: {exc}"
                    tools_used = []

                if tools_used:
                    st.caption(f"Tools used: {', '.join(tools_used)}")
                st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})


def main() -> None:
    _init_state()

    try:
        latest_snapshot, summary = compute_fleet_latest_snapshot()
    except Exception as exc:
        st.error(f"Failed to compute fleet snapshot: {exc}")
        return

    active_page = st.session_state.get("active_page", "Fleet Dashboard")
    page = _sidebar(summary, latest_snapshot)
    if page != active_page:
        st.session_state["active_page"] = page
        active_page = page

    if active_page == "Fleet Dashboard":
        _fleet_dashboard(latest_snapshot, summary)
    elif active_page == "Engine Deep Dive":
        _engine_deep_dive(latest_snapshot)
    else:
        _ai_advisor(summary, latest_snapshot)


if __name__ == "__main__":
    main()

