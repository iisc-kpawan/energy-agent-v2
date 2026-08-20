from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import PurePosixPath
from dataclasses import dataclass
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
AGENT_TIMEOUT_SECONDS = 600

READ_TOOLS = {
    "copy_file", "load_idf_model", "get_model_summary", "check_simulation_settings",
    "inspect_schedules", "inspect_people", "inspect_lights", "inspect_electric_equipment",
    "list_zones", "get_surfaces", "get_materials", "validate_idf", "get_output_variables",
    "get_output_meters", "list_sample_files", "list_available_files", "get_server_configuration", "get_server_status",
    "discover_hvac_loops", "get_loop_topology", "get_server_logs", "get_error_logs",
}
WRITE_TOOLS = {
    "modify_people", "modify_lights", "modify_electric_equipment", "modify_simulation_control",
    "modify_run_period", "change_infiltration_by_mult", "add_window_film_outside",
    "add_coating_outside", "add_output_variables", "add_output_meters",
}
SIM_TOOLS = {"run_energyplus_simulation"}
RESULT_TOOLS = {"calculate_energy_performance", "create_interactive_plot", "visualize_loop_diagram", "get_error_logs"}
ADMIN_TOOLS = {"clear_logs"}
OPTIMIZATION_TOOLS = {"run_optimization_demo"}
CALIBRATION_TOOLS = {"run_calibration_demo", "calculate_calibration_metrics"}
SENSITIVITY_TOOLS = {"run_sensitivity_demo"}

# Small intent-specific tool groups keep large JSON schemas out of unrelated prompts.
TOOL_INTENTS = (
    (("available", "sample", "file", "weather"), {"list_sample_files"}),
    (("full catalog", "installation examples", "all examples"), {"list_available_files"}),
    (("copy", "duplicate", "derived model"), {"copy_file"}),
    (("zone", "surface", "material", "envelope"), {"load_idf_model", "get_model_summary", "list_zones", "get_surfaces", "get_materials"}),
    (("schedule", "people", "light", "equipment", "load"), {"load_idf_model", "inspect_schedules", "inspect_people", "inspect_lights", "inspect_electric_equipment"}),
    (("hvac", "loop", "topology"), {"load_idf_model", "discover_hvac_loops", "get_loop_topology", "visualize_loop_diagram"}),
    (("validate", "setting", "control", "run period"), {"validate_idf", "check_simulation_settings", "get_error_logs"}),
    (("simulate", "simulation", "energyplus", "annual", "design day"), {"check_simulation_settings", "run_energyplus_simulation", "get_error_logs"}),
    (("eui", "epi", "energy performance", "site energy", "source energy", "floor area"), {"calculate_energy_performance"}),
    (("plot", "chart", "visual"), {"create_interactive_plot"}),
    (("meter",), {"get_output_meters"}),
    (("output variable", "time series"), {"get_output_variables"}),
    (("result", "output", "saving", "unmet"), {"calculate_energy_performance", "get_error_logs"}),
    (("modify people", "change people", "occupancy", "occupant"), {"load_idf_model", "inspect_people", "modify_people"}),
    (("modify light", "change light", "lighting power"), {"load_idf_model", "inspect_lights", "modify_lights"}),
    (("modify equipment", "change equipment", "plug load"), {"load_idf_model", "inspect_electric_equipment", "modify_electric_equipment"}),
    (("modify simulation control", "change simulation control"), {"modify_simulation_control", "check_simulation_settings"}),
    (("modify run period", "change run period", "run period"), {"modify_run_period", "check_simulation_settings"}),
    (("infiltration", "air leakage"), {"change_infiltration_by_mult"}),
    (("window film",), {"add_window_film_outside"}),
    (("coating",), {"add_coating_outside"}),
    (("add output variable",), {"add_output_variables", "get_output_variables"}),
    (("add output meter",), {"add_output_meters", "get_output_meters"}),
    (("clear logs", "clear server logs"), {"clear_logs"}),
    (("optimize", "optimise", "optimization", "pareto", "nsga", "objective", "hypervolume"), {"run_optimization_demo"}),
    (("calibrate", "calibration", "nmbe", "cvrmse", "measured data", "utility bill"), {"run_calibration_demo", "calculate_calibration_metrics"}),
    (("sensitivity", "morris", "sobol", "parameter screening"), {"run_sensitivity_demo"}),
)


@dataclass(frozen=True)
class Specialist:
    key: str
    name: str
    purpose: str
    tools: set[str]


SPECIALISTS = {
    "planner": Specialist("planner", "Planning Agent", "Turn complex goals into measurable steps and dependencies.", set()),
    "model_analyst": Specialist("model_analyst", "Model Analyst", "Inspect IDF structure, loads, schedules, envelope and HVAC without modifying files.", READ_TOOLS),
    "retrofit_engineer": Specialist("retrofit_engineer", "Retrofit Engineer", "Create safe derived model changes; never overwrite an original model.", READ_TOOLS | WRITE_TOOLS),
    "simulation_engineer": Specialist("simulation_engineer", "Simulation Engineer", "Configure and run EnergyPlus simulations and diagnose run failures.", READ_TOOLS | SIM_TOOLS | ADMIN_TOOLS),
    "results_analyst": Specialist("results_analyst", "Results Analyst", "Interpret simulation artifacts, compare alternatives and generate visual results.", READ_TOOLS | RESULT_TOOLS),
    "qa_reviewer": Specialist("qa_reviewer", "QA Reviewer", "Independently validate models, results and claims; return pass, warning or fail.", READ_TOOLS | RESULT_TOOLS),
    "sensitivity_analyst": Specialist("sensitivity_analyst", "Sensitivity Analyst", "Define bounded parameter spaces, screen influential variables, and prevent wasteful searches.", READ_TOOLS | SENSITIVITY_TOOLS),
    "optimization_expert": Specialist("optimization_expert", "Optimization Expert", "Run budgeted, reproducible single- or multi-objective searches and report Pareto evidence.", READ_TOOLS | OPTIMIZATION_TOOLS),
    "calibration_expert": Specialist("calibration_expert", "Calibration Expert", "Align measured data, calculate calibration metrics, optimize bounded parameters, and enforce acceptance criteria.", READ_TOOLS | CALIBRATION_TOOLS),
    "error_analyst": Specialist("error_analyst", "Error Analyst", "Diagnose EnergyPlus warnings, severe errors, failed runs, and invalid evidence without modifying models.", READ_TOOLS),
}


class MultiAgentOrchestrator:
    def __init__(self, tools: list, memory, store, api_key: str, default_model: str, max_parallel: int = 4):
        self.tools = tools
        self.memory = memory
        self.store = store
        self.api_key = api_key
        self.default_model = default_model
        self.max_parallel = max_parallel
        self._agents: dict[tuple[str, str], Any] = {}
        self._write_lock = asyncio.Lock()

    def _model(self, name: str):
        normalized = name.strip().removeprefix("models/")
        if normalized.startswith("gemini"):
            return ChatGoogleGenerativeAI(model=normalized, google_api_key=self.api_key, temperature=0.1)
        return ChatOllama(model=normalized, temperature=0.1)

    def _agent(self, specialist: Specialist, model: str):
        return self._agent_for_tools(specialist, model, specialist.tools)

    def _agent_for_tools(self, specialist: Specialist, model: str, selected: set[str]):
        tool_key = ",".join(sorted(selected))
        key = (specialist.key, model, tool_key)
        if key not in self._agents:
            allowed = [t for t in self.tools if t.name in selected]
            instructions = f"""You are the {specialist.name} in a controlled EnergyPlus engineering team.
Purpose: {specialist.purpose}
Use only the tools assigned to you. Base claims on tool output. Do not invent paths, model facts, or simulation results.
Use list_sample_files for ordinary sample discovery. Use list_available_files only when the user explicitly asks for the full EnergyPlus installation examples or weather-data catalog.
Honor the exact model and weather file named by the user. Never substitute, pivot to, or test a different model unless the requested file is conclusively absent.
If you are the Simulation Engineer and run_energyplus_simulation is available, you must execute it for a simulation request and report its exact output directory and runtime.
If you are the Results Analyst, call calculate_energy_performance only with the completed simulation output directory supplied by a prior report. Never pass an IDF file as output_directory and never run a simulation yourself.
When a model path exists in DURABLE PROJECT CONTEXT, use it without asking again.
For modifications, always create a new output file and never overwrite the source. Return concise findings plus evidence.
You are a specialist, not the user-facing assistant."""
            self._agents[key] = create_react_agent(self._model(model), allowed, prompt=instructions)
        return self._agents[key]

    @staticmethod
    def _select_tools(specialist: Specialist, task: str) -> set[str]:
        # Durable memory and prior reports may mention unrelated operations. Only the
        # active request should influence schemas sent to the provider.
        if specialist.key == "qa_reviewer" and "PRIOR SPECIALIST REPORTS:" in task:
            return set()
        q = task.rsplit("CURRENT USER REQUEST:", 1)[-1]
        q = q.split("PRIOR SPECIALIST REPORTS:", 1)[0].lower()
        selected: set[str] = set()
        for terms, names in TOOL_INTENTS:
            if any(term in q for term in terms):
                selected |= names
        selected &= specialist.tools
        # Role contracts take precedence over the generic eight-schema budget.
        # Otherwise alphabetical truncation can silently remove the one action a
        # specialist exists to perform in broad end-to-end requests.
        if specialist.key == "simulation_engineer" and any(x in q for x in ("simulate", "simulation", "annual", "energyplus")):
            return {"list_sample_files", "check_simulation_settings", "validate_idf", "run_energyplus_simulation", "get_error_logs"} & specialist.tools
        if specialist.key == "results_analyst":
            return ({"calculate_energy_performance", "get_error_logs"} if any(
                x in q for x in ("eui", "epi", "energy performance", "site energy", "source energy", "floor area", "result")
            ) else selected) & specialist.tools
        if specialist.key == "optimization_expert":
            return OPTIMIZATION_TOOLS & specialist.tools
        if specialist.key == "sensitivity_analyst":
            return SENSITIVITY_TOOLS & specialist.tools
        if specialist.key == "calibration_expert":
            return CALIBRATION_TOOLS & specialist.tools
        if selected:
            return set(sorted(selected)[:8])
        defaults = {
            "model_analyst": {"load_idf_model", "get_model_summary", "list_sample_files"},
            "retrofit_engineer": {"copy_file", "load_idf_model", "get_model_summary"},
            "simulation_engineer": {"check_simulation_settings", "run_energyplus_simulation", "get_error_logs"},
            "results_analyst": {"get_output_variables", "get_output_meters", "create_interactive_plot"},
            "qa_reviewer": {"validate_idf", "get_error_logs", "get_server_status"},
        }
        return defaults.get(specialist.key, set()) & specialist.tools

    @staticmethod
    def route(message: str) -> list[str]:
        q = message.lower()
        routes: list[str] = []
        complex_goal = any(x in q for x in ("best way", "optimiz", "strategy", "reduce annual", "compare options", "improve energy"))
        modify = any(x in q for x in ("modify", "change", "increase", "decrease", "reduce", "add film", "add coating", "retrofit"))
        existing_outputs = any(x in q for x in ("completed simulation", "/outputs/", "/simulation_output/"))
        maintenance = any(x in q for x in ("clear logs", "clear server logs"))
        simulate = bool(re.search(r"\b(simulate|run|execute|start|perform|conduct|handle|complete)\b", q)) and any(
            x in q for x in ("simulat", "energyplus", "annual", "design day")
        ) and not existing_outputs
        results = any(x in q for x in ("result", "plot", "chart", "compare", "saving", "eui", "epi", "energy performance", "unmet hour"))
        optimization = any(x in q for x in ("optimize", "optimise", "optimization", "pareto", "nsga", "hypervolume"))
        calibration = any(x in q for x in ("calibrate", "calibration", "nmbe", "cvrmse", "utility bill", "measured data"))
        sensitivity = any(x in q for x in ("sensitivity", "morris", "sobol", "parameter screening"))
        error_analysis = any(x in q for x in ("error", "warning", "failed run", "failure", "diagnose"))
        inspect = any(x in q for x in ("inspect", "zone", "surface", "material", "schedule", "hvac", "load", "validate", "model", "idf"))
        inspect = inspect or ("summar" in q and any(x in q for x in ("model", "idf", "building")))
        inspect = inspect or ("list" in q and any(x in q for x in ("sample", "available", "model", "idf", "file")))
        if complex_goal: routes.append("planner")
        if inspect or complex_goal or not (modify or simulate or results): routes.append("model_analyst")
        if modify: routes.append("retrofit_engineer")
        if sensitivity or optimization or calibration: routes.append("sensitivity_analyst")
        if optimization: routes.append("optimization_expert")
        if calibration: routes.append("calibration_expert")
        if error_analysis: routes.append("error_analyst")
        if simulate: routes.append("simulation_engineer")
        if maintenance: routes.append("simulation_engineer")
        if results: routes.append("results_analyst")
        if modify or simulate or results or optimization or calibration or sensitivity: routes.append("qa_reviewer")
        return list(dict.fromkeys(routes))

    @staticmethod
    def _text(result: dict) -> str:
        for message in reversed(result.get("messages", [])):
            if getattr(message, "type", "") == "ai" and message.content:
                if isinstance(message.content, str): return message.content
                return "\n".join(x.get("text", "") for x in message.content if isinstance(x, dict))
        return "No specialist report was produced."

    @staticmethod
    def _tool_calls(result: dict) -> list[dict]:
        found = []
        for message in result.get("messages", []):
            for call in getattr(message, "tool_calls", []) or []:
                found.append({"name": call.get("name", "?"), "args": call.get("args", {})})
        return found

    @staticmethod
    def _usage(result: dict, fallback_text: str = "") -> dict:
        """Aggregate provider usage metadata from every model response in an agent loop."""
        prompt = completion = total = calls = 0
        for message in result.get("messages", []):
            if getattr(message, "type", "") != "ai": continue
            calls += 1
            usage = getattr(message, "usage_metadata", None) or {}
            response_usage = (getattr(message, "response_metadata", None) or {}).get("usage_metadata", {})
            usage = usage or response_usage
            p = usage.get("input_tokens", usage.get("prompt_token_count", 0)) or 0
            c = usage.get("output_tokens", usage.get("candidates_token_count", 0)) or 0
            t = usage.get("total_tokens", usage.get("total_token_count", 0)) or (p + c)
            prompt += int(p); completion += int(c); total += int(t)
        estimated = calls > 0 and total == 0
        if estimated:
            # Provider-neutral fallback: an explicit approximation, never presented as billing truth.
            prompt = max(1, len(fallback_text) // 4)
            total = prompt
        return {"api_calls": calls, "prompt_tokens": prompt, "completion_tokens": completion,
                "total_tokens": total, "estimated": estimated}

    async def _run_one(self, key: str, model: str, task: str, run_id: str, conversation_id: str) -> dict:
        specialist = SPECIALISTS[key]
        started_at = time.monotonic()
        self.store.update_run(run_id, event={"agent": key, "status": "started"})
        try:
            selected_tools = self._select_tools(specialist, task)
            # EPI extraction is deterministic and must not be left to an LLM's
            # discretion. Once the simulation specialist has supplied an output
            # directory, invoke the parser directly and preserve its exact JSON.
            if specialist.key == "results_analyst" and "calculate_energy_performance" in selected_tools:
                paths = re.findall(r"(/workspace/[^\s`\"']*/outputs/[^\s`\"']+)", task)
                if paths:
                    output_directory = paths[-1].rstrip(".,;:)]}")
                    candidate = PurePosixPath(output_directory)
                    if candidate.suffix:
                        output_directory = str(candidate.parent)
                    calculator = next(t for t in self.tools if t.name == "calculate_energy_performance")
                    raw = await calculator.ainvoke({"output_directory": output_directory})
                    usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "estimated": False}
                    tool_calls = [{"name": "calculate_energy_performance",
                                   "args": {"output_directory": output_directory}}]
                    self.store.add_usage(run_id, conversation_id, key, model, tool_calls=1, **usage)
                    duration = round(time.monotonic() - started_at, 2)
                    report = {"agent": key, "name": specialist.name,
                              "report": json.dumps(raw, ensure_ascii=False, default=str),
                              "tools": tool_calls, "usage": usage, "elapsed_seconds": duration}
                    self.store.update_run(run_id, event={"agent": key, "status": "completed",
                                                         "tools": ["calculate_energy_performance"], "elapsed_seconds": duration})
                    return report
            agent = self._agent_for_tools(specialist, model, selected_tools)
            lock = self._write_lock if key == "retrofit_engineer" else _NullAsyncContext()
            async with lock:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": [{"role": "user", "content": task}]}),
                    timeout=AGENT_TIMEOUT_SECONDS,
                )
            tool_calls = self._tool_calls(result)
            usage = self._usage(result, task)
            self.store.add_usage(run_id, conversation_id, key, model, tool_calls=len(tool_calls), **usage)
            duration = round(time.monotonic() - started_at, 2)
            report = {"agent": key, "name": specialist.name, "report": self._text(result), "tools": tool_calls, "usage": usage, "elapsed_seconds": duration}
            self.store.update_run(run_id, event={"agent": key, "status": "completed", "tools": [x["name"] for x in report["tools"]], "elapsed_seconds": duration})
            return report
        except Exception as exc:
            logger.exception("Specialist %s failed", key)
            self.store.update_run(run_id, event={"agent": key, "status": "failed", "error": str(exc)})
            return {"agent": key, "name": specialist.name, "report": f"Specialist failed: {exc}", "tools": [], "error": str(exc)}

    async def run(self, conversation_id: str, message: str, model: str | None = None, mode: str = "multi-agent", run_id: str | None = None) -> dict:
        workflow_started_at = time.monotonic()
        model = model or self.default_model
        self.store.add_message(conversation_id, "user", message)
        context = self.memory.build(conversation_id)
        run_id = run_id or self.store.create_run(conversation_id)
        if mode == "direct":
            direct = Specialist("direct_agent", "Direct EnergyPlus Agent",
                                "Provide the original conversational workflow with access to every connected tool.",
                                {tool.name for tool in self.tools})
            self.store.update_run(run_id, route=["direct_agent"], event={"agent": "direct_agent", "status": "started"})
            task = f"{self.memory.prompt(context)}\n\nCURRENT USER REQUEST:\n{message}"
            result = await asyncio.wait_for(
                self._agent_for_tools(direct, model, direct.tools).ainvoke({"messages": [{"role": "user", "content": task}]}),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
            answer, calls, usage = self._text(result), self._tool_calls(result), self._usage(result, task)
            self.store.add_usage(run_id, conversation_id, "direct_agent", model, tool_calls=len(calls), **usage)
            metadata = {"run_id": run_id, "agents": ["direct_agent"], "tools": calls, "mode": "direct"}
            elapsed = round(time.monotonic() - workflow_started_at, 2)
            answer = f"{answer}\n\n**Total workflow time:** {elapsed:.2f} seconds"
            self.store.add_message(conversation_id, "assistant", answer, metadata)
            self.store.update_run(run_id, status="completed", event={"agent": "direct_agent", "status": "completed", "tools": [x["name"] for x in calls], "elapsed_seconds": elapsed})
            self.memory.compact(conversation_id)
            return {"response": answer, "run_id": run_id, "agents": ["direct_agent"], "tools_used": calls,
                    "mode": "direct", "elapsed_seconds": elapsed, "usage": self.store.usage_summary(conversation_id)}
        routes = self.route(message)
        self.store.update_run(run_id, route=routes, event={"agent": "supervisor", "status": "planned", "routes": routes})
        base = self.memory.prompt(context)
        task = f"{base}\n\nCURRENT USER REQUEST:\n{message}"

        # Read-only agents may run concurrently. Consequential workflows run in dependency order.
        read_routes = [r for r in routes if r in {"planner", "model_analyst"}]
        reports = await asyncio.gather(*(self._run_one(r, model, task, run_id, conversation_id) for r in read_routes))
        completed = {r["agent"] for r in reports}
        for key in routes:
            if key in completed: continue
            prior = "\n\n".join(f"{r['name']}:\n{r['report']}" for r in reports)
            reports.append(await self._run_one(key, model, task + "\n\nPRIOR SPECIALIST REPORTS:\n" + prior, run_id, conversation_id))

        synthesis_prompt = f"""You are the Supervisor of an EnergyPlus engineering team. Answer the user directly.
Synthesize the specialist reports below. Resolve contradictions in favor of tool-backed evidence and QA findings.
Clearly distinguish completed actions, proposed actions, failures, assumptions and next steps. Do not claim a simulation ran unless evidence says so.
If specialist evidence contains an `artifacts` object or `/api/v2/analysis/files/` URLs, preserve those exact URLs as clickable Markdown download links in the final answer.
For engineering calculations or simulations, format the answer as a professional Markdown report with:
1. `## Execution Summary` and a table with Step, Status, Time Taken, and Evidence columns. Use each specialist's exact `elapsed_seconds`; write `N/A` when timing evidence is unavailable.
2. `## Results` containing the key numeric metrics, units, model, weather file, runtime, and output paths actually supported by evidence.
3. `## Issues` when anything failed or remains uncertain.
4. `## Next Steps` with concise actionable items.
Use normal Markdown headings and tables, not decorative rows of equals signs. Never invent an EPI, EUI, floor area, benchmark, or success status.
{base}
USER REQUEST: {message}
SPECIALIST REPORTS:
{json.dumps(reports, ensure_ascii=False)}"""
        supervisor = create_react_agent(self._model(model), [], prompt="You are the user-facing EnergyPlus Supervisor.")
        result = await asyncio.wait_for(
            supervisor.ainvoke({"messages": [{"role": "user", "content": synthesis_prompt}]}),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        answer = self._text(result)
        supervisor_usage = self._usage(result, synthesis_prompt)
        self.store.add_usage(run_id, conversation_id, "supervisor", model, tool_calls=0, **supervisor_usage)
        tools = [tool for report in reports for tool in report.get("tools", [])]
        metadata = {"run_id": run_id, "agents": routes, "tools": tools}
        elapsed = round(time.monotonic() - workflow_started_at, 2)
        answer = f"{answer}\n\n**Total workflow time:** {elapsed:.2f} seconds"
        self.store.add_message(conversation_id, "assistant", answer, metadata)
        self.store.update_run(run_id, status="completed", event={"agent": "supervisor", "status": "completed", "elapsed_seconds": elapsed})
        self.memory.compact(conversation_id)
        return {"response": answer, "run_id": run_id, "agents": routes, "tools_used": tools,
                "elapsed_seconds": elapsed, "usage": self.store.usage_summary(conversation_id)}


class _NullAsyncContext:
    async def __aenter__(self): return self
    async def __aexit__(self, *_): return False
