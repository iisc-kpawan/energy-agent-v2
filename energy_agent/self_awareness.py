"""Deterministic, secret-safe platform self-awareness tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import tool


def _category(name: str) -> str:
    if name in {"get_platform_capabilities", "describe_agent_team", "explain_artifact_locations", "list_registered_tools"}:
        return "platform_awareness"
    if name in {"calibrate_occupancy", "get_calibration_job", "calculate_calibration_metrics"}:
        return "calibration"
    if name in {"run_energyplus_optimization", "run_energyplus_sensitivity", "get_energyplus_study_job"}:
        return "real_engineering_studies"
    if name.startswith("run_surrogate_"):
        return "surrogate_demo"
    if name in {"run_energyplus_simulation"}:
        return "simulation"
    if name in {"calculate_energy_performance", "create_interactive_plot", "visualize_loop_diagram"}:
        return "results_and_visualization"
    if name.startswith(("inspect_", "get_", "list_", "load_", "discover_", "validate_", "check_")):
        return "inspection_and_diagnostics"
    if name.startswith(("modify_", "add_", "change_", "copy_")):
        return "model_modification"
    return "administration"


def self_awareness_tools(
    registry: dict[str, list[Any]], agent_snapshot: list[dict], root: Path,
    data_dir: Path, output_root: Path,
) -> list:
    """Create tools backed by a registry populated after all tools are built."""

    @tool("list_registered_tools")
    def list_tools(category: str = "all", include_descriptions: bool = True) -> str:
        """Return the authoritative live unique tool registry and exact count. Use this for any question about available tools; never infer or invent an inventory."""
        unique = {item.name: item for item in registry.get("tools", [])}
        rows = []
        for name in sorted(unique):
            item_category = _category(name)
            if category != "all" and category != item_category:
                continue
            row = {"name": name, "category": item_category}
            if include_descriptions:
                row["description"] = str(getattr(unique[name], "description", ""))
            rows.append(row)
        return json.dumps({"status": "verified", "source": "live_application_registry",
                           "filter": category, "unique_tool_count": len(rows), "tools": rows})

    @tool("describe_agent_team")
    def describe_agents() -> str:
        """Return the authoritative specialist-agent roster and each role's permitted tool names/count."""
        rows = []
        for agent in agent_snapshot:
            names = sorted(agent["tools"])
            rows.append({"id": agent["id"], "name": agent["name"], "purpose": agent["purpose"],
                         "permitted_tool_count": len(names), "permitted_tools": names})
        return json.dumps({"agent_count": len(rows), "agents": rows,
                           "note": "Role permissions are intentional; no specialist receives every tool."})

    @tool("get_platform_capabilities")
    def platform_capabilities() -> str:
        """Describe application version, modes, real versus demo workflows, and honest current limitations."""
        return json.dumps({
            "product": "Energy Agent V2", "version": "2.0.0", "architecture": "dual-mode-agentic-v2",
            "modes": {
                "multi-agent": "Routes work to dedicated specialists and a QA reviewer.",
                "direct": "Original compatibility mode allowing the model to choose MCP tools directly.",
            },
            "real_workflows": ["EnergyPlus simulation", "EPI/EUI and results extraction",
                               "occupancy calibration against fixed measured CSV",
                               "lighting optimization", "lighting or occupancy sensitivity"],
            "demo_only": ["surrogate optimization", "surrogate calibration", "surrogate sensitivity"],
            "limitations": ["single-parameter occupancy calibration", "lighting-only optimization",
                            "one-at-a-time single-parameter sensitivity", "hourly Electricity:Facility time series",
                            "sequential EnergyPlus evaluations", "no Bayesian optimization or PSO"],
            "security": "This tool never returns credentials, API keys, MCP tokens, or passwords.",
        })

    @tool("explain_artifact_locations")
    def artifact_locations() -> str:
        """Explain where uploads, conversations, simulation outputs, calibration jobs, and study results are stored and downloaded."""
        return json.dumps({
            "project_root": str(root.resolve()), "application_data": str(data_dir.resolve()),
            "energyplus_outputs": str(output_root.resolve()),
            "calibration_jobs": str((data_dir / "calibration_jobs").resolve()),
            "engineering_studies": str((data_dir / "engineering_studies").resolve()),
            "download_access": "Completed chat runs and job file APIs provide authenticated download URLs.",
            "persistence": "Conversation database, imported artifacts, job summaries, and production outputs persist across service restarts.",
            "secrets_excluded": True,
        })

    return [list_tools, describe_agents, platform_capabilities, artifact_locations]
