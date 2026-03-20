from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os
import re

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from .fleet import FleetSummary, dataset_level_stats, top_critical_engines
from .predictor import rul_trajectory_for_engine, shap_for_single_row
from .utils import degradation_slope, rul_to_health_status


load_dotenv()


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


def _build_system_prompt(
    fleet_summary: FleetSummary,
    latest_snapshot: pd.DataFrame,
    selected_engine: Optional[Dict[str, Any]] = None,
) -> str:
    # Top 5 critical engines
    top5 = top_critical_engines(latest_snapshot, top_n=5)
    lines: List[str] = []
    for _, row in top5.iterrows():
        lines.append(
            f"- Engine {int(row['engine_id'])} | Dataset {row['dataset']} | "
            f"RUL {row['predicted_RUL']:.1f} cycles | Top SHAP feature: s4_rollmean"
        )
    top5_block = "\n".join(lines) if lines else "No engines available."

    engine_block = "No engine is currently selected."
    if selected_engine is not None:
        ds = selected_engine["dataset"]
        eid = int(selected_engine["engine_id"])
        cycle = int(selected_engine["cycle"])
        pred_rul = float(selected_engine["predicted_RUL"])
        status = str(selected_engine["health_status"])

        # Degradation trend
        try:
            traj = rul_trajectory_for_engine(ds, eid)
            summary = degradation_slope(
                traj["cycle"].to_numpy(), traj["predicted_RUL"].to_numpy()
            )
            trend_label = summary.label
        except Exception:
            trend_label = "unknown"

        # SHAP top 3 drivers
        try:
            shap_vals = shap_for_single_row(ds, eid, cycle)
            top3 = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            drivers = ", ".join(f"{name} ({val:+.3f})" for name, val in top3)
        except Exception:
            drivers = "Not available"

        engine_block = (
            f"Engine: {eid} | Dataset: {ds} | Predicted RUL: {pred_rul:.1f}\n"
            f"Health status: {status}\n"
            f"Top degradation drivers: {drivers}\n"
            f"Degradation trend: {trend_label}"
        )

    prompt = f"""
You are an expert predictive maintenance engineer assistant for a
turbofan engine fleet monitoring system. You help maintenance
managers make scheduling and risk decisions.

FLEET STATUS (real-time from model predictions):
- Total engines: {fleet_summary.total_engines}
- Critical (RUL < 30): {fleet_summary.critical_count} engines
- Warning (RUL 30–70): {fleet_summary.warning_count} engines
- Healthy (RUL > 70): {fleet_summary.healthy_count} engines

TOP 5 CRITICAL ENGINES:
{top5_block}

MODEL INFORMATION:
Algorithm: Random Forest (GridSearchCV optimized)
RMSE: ±15.51 cycles | MAE: 10.89 | R²: 0.858
RUL cap: 125 cycles. Always acknowledge model uncertainty in
recommendations. The model's RMSE means actual RUL could vary
by ±15 cycles from the prediction.

CURRENTLY SELECTED ENGINE (if user is in Deep Dive view):
{engine_block}

SENSOR CONTEXT:
Sensor labels are anonymized per NASA CMAPSS convention.
Top global signal: s4_rollmean (59.6% importance). This likely
corresponds to a high-pressure compressor or turbine temperature
sensor based on CMAPSS research literature.

Answer questions concisely in engineering terms. Reference actual
predicted values. Prioritize actionable recommendations.
When asked about costs or scheduling, reason through trade-offs
using the RMSE uncertainty as a risk factor.
""".strip()
    return prompt


def call_grok_api(
    messages: List[ChatMessage],
    model: str = "grok-3-mini",
) -> str:
    """
    Call the xAI Grok chat completion API using an OpenAI-compatible interface.
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is not set. Add it to your .env file to enable the AI advisor."
        )

    url = "https://api.x.ai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [m.__dict__ for m in messages],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def build_initial_messages(
    fleet_summary: FleetSummary,
    latest_snapshot: pd.DataFrame,
    selected_engine: Optional[Dict[str, Any]] = None,
) -> List[ChatMessage]:
    """
    Build the initial system message for a new conversation.
    """
    system_prompt = _build_system_prompt(fleet_summary, latest_snapshot, selected_engine)
    return [ChatMessage(role="system", content=system_prompt)]


def _extract_delay_cycles(user_input: str) -> Optional[int]:
    """
    Extract an integer delay like "by 7 cycles" from user text.
    """
    m = re.search(r"\bby\s+(\d+)\s*(?:cycles|cycle)\b", user_input, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s*(?:cycles|cycle)\b", user_input, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_datasets_from_text(user_input: str) -> List[str]:
    """
    Extract FD00x dataset ids from the user question.
    """
    found = re.findall(r"\bFD0?\d{3}\b", user_input.upper())
    # Normalize "FD1xx" forms into "FD00x"
    normalized: List[str] = []
    for ds in found:
        # Ensure exactly FD00x format
        digits = re.sub(r"[^0-9]", "", ds)
        if len(digits) == 3:
            normalized.append(f"FD{digits}")
    # Keep unique while preserving order
    seen = set()
    out: List[str] = []
    for ds in normalized:
        if ds not in seen:
            out.append(ds)
            seen.add(ds)
    return out


def build_tool_context_for_question(
    user_input: str,
    fleet_summary: FleetSummary,
    latest_snapshot: pd.DataFrame,
    selected_engine: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Minimal deterministic tool layer for the AI Advisor.

    We don't use model-side tool calling. Instead, we detect common intents
    (critical engines, dataset comparison, delay impact) and inject computed
    results into the prompt to keep answers grounded.
    """
    tools_used: List[str] = []
    blocks: List[str] = []

    ui = user_input.lower()

    # Always include the currently selected engine context so the advisor
    # cannot drift when the user changes engine selection mid-conversation.
    if selected_engine is not None:
        ds = selected_engine.get("dataset")
        eid = selected_engine.get("engine_id")
        cycle = selected_engine.get("cycle")
        pred_rul = selected_engine.get("predicted_RUL")
        health_status = selected_engine.get("health_status")

        blocks.append(
            "TOOL RESULT: CURRENTLY SELECTED ENGINE CONTEXT\n"
            f"- Engine {int(eid)} | Dataset {ds}\n"
            f"- Current cycle: {int(cycle)}\n"
            f"- Predicted RUL: {float(pred_rul):.1f} cycles\n"
            f"- Health status: {health_status}"
        )
        tools_used.append("get_selected_engine_context")

        # For “drivers/why” questions, include top 3 local SHAP drivers.
        needs_driver_context = any(k in ui for k in ["degradation driver", "degradation drivers", "driver", "drivers"])
        needs_topk = any(k in ui for k in ["top 3", "top three", "top3", "top 10", "top10", "top"])
        needs_why = any(k in ui for k in ["why", "behind", "explain"])
        if needs_driver_context and (needs_topk or needs_why):
            try:
                shap_vals = shap_for_single_row(str(ds), int(eid), int(cycle))
                top3 = sorted(shap_vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                drivers_txt = ", ".join(f"{name} ({val:+.3f})" for name, val in top3)
                blocks.append(
                    "TOOL RESULT: TOP DRIVERS (LOCAL SHAP)\n"
                    f"- Top 3 drivers: {drivers_txt}"
                )
                tools_used.append("get_top_3_shap_drivers")
            except Exception:
                blocks.append(
                    "TOOL RESULT: TOP DRIVERS (LOCAL SHAP)\n"
                    "- Top 3 drivers: Not available (SHAP computation failed/unavailable)."
                )

    # 1) Maintenance / critical request
    if selected_engine is not None or any(k in ui for k in ["maintenance", "critical", "urgent", "this week"]):
        if any(k in ui for k in ["maintenance", "critical", "urgent", "this week"]):
            tools_used.append("get_top_critical_engines")
            top5 = top_critical_engines(latest_snapshot, top_n=5)
            lines = []
            for _, row in top5.iterrows():
                lines.append(
                    f"- Engine {int(row['engine_id'])} | {row['dataset']} | "
                    f"predicted_RUL={row['predicted_RUL']:.1f} | {row['health_status']}"
                )
            blocks.append("TOOL RESULT: TOP 5 CRITICAL ENGINES\n" + "\n".join(lines))

    # 2) Delay / risk request (requires an engine selection context)
    delay_cycles = _extract_delay_cycles(user_input)
    delay_intent = any(k in ui for k in ["delay", "delaying", "delayed", "risk", "postpone", "delaying maintenance"])
    if delay_intent and delay_cycles is not None and selected_engine is not None:
        ds = selected_engine["dataset"]
        eid = int(selected_engine["engine_id"])
        current_cycle = int(selected_engine["cycle"])
        target_cycle = current_cycle + int(delay_cycles)

        tools_used.append("get_delay_impact")
        traj = rul_trajectory_for_engine(ds, eid)

        current_rows = traj[traj["cycle"] == current_cycle]
        if current_rows.empty:
            # Fallback to nearest cycle if exact doesn't exist (should be rare)
            idx = (traj["cycle"] - current_cycle).abs().idxmin()
            current_rul = float(traj.loc[idx, "predicted_RUL"])
        else:
            current_rul = float(current_rows["predicted_RUL"].iloc[0])

        target_rows = traj[traj["cycle"] == target_cycle]
        if not target_rows.empty:
            future_rul = float(target_rows["predicted_RUL"].iloc[0])
            proxy_used = False
        else:
            # Simple extrapolation using fitted RUL-vs-cycle slope
            summary = degradation_slope(
                traj["cycle"].to_numpy(),
                traj["predicted_RUL"].to_numpy(),
            )
            future_rul = current_rul + float(summary.slope) * float(delay_cycles)
            future_rul = float(np.clip(future_rul, 0, 125))
            proxy_used = True

        delta = future_rul - current_rul
        new_status = rul_to_health_status(future_rul)

        blocks.append(
            "TOOL RESULT: DELAY IMPACT\n"
            f"- Engine {eid} | {ds}\n"
            f"- Current cycle: {current_cycle} | predicted_RUL_now={current_rul:.1f} | {rul_to_health_status(current_rul)}\n"
            f"- Target cycle: {target_cycle} | predicted_RUL_after={future_rul:.1f} | {new_status}\n"
            f"- Delta: {delta:+.1f} cycles\n"
            f"- Future RUL source: {'proxy (extrapolation)' if proxy_used else 'exact from trajectory'}"
        )

    # 3) Dataset comparison request
    if any(k in ui for k in ["compare", "versus", "between"]) or "fd" in ui:
        dataset_ids = _extract_datasets_from_text(user_input)
        if not dataset_ids:
            # fallback: compare all datasets we have in engineered files
            dataset_ids = sorted(latest_snapshot["dataset"].unique().tolist())

        # Keep it small: only show those datasets we found
        stats = latest_snapshot.copy()
        stats = stats[stats["dataset"].isin(dataset_ids)]

        tools_used.append("get_dataset_level_stats")
        avg = dataset_level_stats(stats)
        lines = []
        for _, row in avg.iterrows():
            lines.append(f"- {row['dataset']}: avg_predicted_RUL={row['avg_predicted_RUL']:.1f}")

        blocks.append("TOOL RESULT: DATASET COMPARISON\n" + "\n".join(lines))

    context = "\n\n".join(blocks).strip()
    return {"tools_used": tools_used, "context": context}

