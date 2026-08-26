"""Async adapter around the existing EnergyPlus-MCP tools.

Engineering services call this adapter instead of knowing MCP response prefixes
or tool schemas. Every candidate receives isolated model and output paths.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from energy_agent.analytics.timeseries import TimeSeries, load_energyplus_meter_csv


def _idf_objects(text: str) -> list[tuple[list[str], list[tuple[int, int]]]]:
    """Return IDF fields and their source spans without rewriting the model.

    Comments are ignored and delimiters terminate fields. Source spans contain
    only the field value, allowing precise edits that preserve formatting and
    extensible objects that older Eppy releases cannot round-trip.
    """
    objects: list[tuple[list[str], list[tuple[int, int]]]] = []
    fields: list[str] = []
    spans: list[tuple[int, int]] = []
    token_start: int | None = None
    in_comment = False
    for index, char in enumerate(text):
        if in_comment:
            if char in "\r\n":
                in_comment = False
            continue
        if char == "!":
            in_comment = True
            continue
        if char not in " \t\r\n," and token_start is None:
            token_start = index
        if char in ",;":
            start = token_start if token_start is not None else index
            end = index
            while end > start and text[end - 1].isspace():
                end -= 1
            fields.append(text[start:end].strip())
            spans.append((start, end))
            token_start = None
            if char == ";":
                if fields and fields[0]:
                    objects.append((fields, spans))
                fields, spans = [], []
    return objects


def _json_payload(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        text = "\n".join(str(block.get("text", "")) for block in raw if isinstance(block, dict) and block.get("type") == "text")
    elif hasattr(raw, "content"):
        return _json_payload(raw.content)
    else:
        text = str(raw)
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"EnergyPlus tool returned no structured JSON: {text[:500]}")
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse EnergyPlus tool response: {text[:500]}") from exc


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SimulationEvidence:
    model_path: str
    output_directory: str
    meter_csv: str
    series: TimeSeries
    simulation: dict


class EnergyPlusRunner:
    """Run model modifications and simulations through existing MCP tools."""

    REQUIRED_TOOLS = {"inspect_people", "modify_people", "add_output_meters", "run_energyplus_simulation"}

    def __init__(self, tools: list[Any], allowed_roots: list[Path]):
        self.tools = {tool.name: tool for tool in tools}
        missing = self.REQUIRED_TOOLS - self.tools.keys()
        if missing:
            raise RuntimeError(f"EnergyPlus MCP is missing required tools: {sorted(missing)}")
        self.allowed_roots = [root.resolve() for root in allowed_roots]

    def resolve_file(self, raw_path: str, suffix: str) -> Path:
        supplied = Path(raw_path)
        candidates = [supplied] if supplied.is_absolute() else [root / supplied for root in self.allowed_roots]
        if not supplied.is_absolute():
            candidates += [root / "sample_files" / supplied.name for root in self.allowed_roots]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.suffix.lower() == suffix and any(resolved == root or root in resolved.parents for root in self.allowed_roots):
                return resolved
        raise FileNotFoundError(f"Allowed {suffix} file not found: {raw_path}")

    async def _invoke(self, name: str, arguments: dict) -> dict:
        return _json_payload(await self.tools[name].ainvoke(arguments))

    async def inspect_people(self, idf_path: Path) -> dict:
        result = await self._invoke("inspect_people", {"idf_path": str(idf_path)})
        if not result.get("success") or not result.get("people_objects"):
            raise ValueError(f"Model has no calibratable People objects: {result.get('error', result)}")
        return result

    async def configure_hourly_electricity(self, source_idf: Path, target_idf: Path) -> Path:
        target_idf.parent.mkdir(parents=True, exist_ok=True)
        text = source_idf.read_text(encoding="utf-8", errors="replace")
        objects = _idf_objects(text)
        already_hourly = any(
            fields[0].lower() in {"output:meter", "output:meter:meterfileonly"}
            and len(fields) >= 3
            and fields[1].lower() == "electricity:facility"
            and fields[2].lower() == "hourly"
            for fields, _ in objects
        )
        if not already_hourly:
            text = text.rstrip() + (
                "\n\nOutput:Meter:MeterFileOnly,\n"
                "    Electricity:Facility,     !- Key Name\n"
                "    Hourly;                    !- Reporting Frequency\n"
            )
        target_idf.write_text(text, encoding="utf-8")
        return target_idf

    @staticmethod
    def occupancy_modifications(inspection: dict, multiplier: float) -> tuple[list[dict], list[dict]]:
        if multiplier <= 0:
            raise ValueError("Occupancy multiplier must be positive")
        modifications, trace = [], []
        mapping = {
            "people": ("People", "number_of_people", "Number_of_People", lambda value: value * multiplier),
            "people/area": ("People/Area", "people_per_area", "People_per_Floor_Area", lambda value: value * multiplier),
            "area/person": ("Area/Person", "area_per_person", "Floor_Area_per_Person", lambda value: value / multiplier),
        }
        for people in inspection["people_objects"]:
            raw_method = str(people.get("calculation_method", "")).strip()
            method_key = raw_method.lower()
            if method_key not in mapping:
                raise ValueError(f"Unsupported People calculation method {raw_method!r} for {people.get('name')}")
            method, source_key, field, transform = mapping[method_key]
            try:
                baseline = float(people[source_key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"People object {people.get('name')} has no valid {source_key}") from exc
            candidate = transform(baseline)
            modifications.append({"target": f"name:{people['name']}", "field_updates": {field: candidate}})
            trace.append({"object": people["name"], "method": method, "field": field,
                          "baseline": baseline, "candidate": candidate})
        return modifications, trace

    async def apply_occupancy_multiplier(self, source_idf: Path, target_idf: Path, inspection: dict, multiplier: float) -> list[dict]:
        target_idf.parent.mkdir(parents=True, exist_ok=True)
        modifications, trace = self.occupancy_modifications(inspection, multiplier)
        text = source_idf.read_text(encoding="utf-8", errors="replace")
        replacements: list[tuple[int, int, str]] = []
        by_name = {item["object"]: item for item in trace}
        field_index = {"Number_of_People": 5, "People_per_Floor_Area": 6, "Floor_Area_per_Person": 7}
        for fields, spans in _idf_objects(text):
            if fields[0].lower() != "people" or len(fields) < 8 or fields[1] not in by_name:
                continue
            item = by_name.pop(fields[1])
            index = field_index[item["field"]]
            replacements.append((*spans[index], f"{item['candidate']:.12g}"))
        if by_name:
            raise RuntimeError(f"Could not safely locate People objects in IDF: {sorted(by_name)}")
        for start, end, value in sorted(replacements, reverse=True):
            text = text[:start] + value + text[end:]
        target_idf.write_text(text, encoding="utf-8")
        return trace

    async def apply_lighting_multiplier(self, source_idf: Path, target_idf: Path, multiplier: float) -> list[dict]:
        """Scale every Lights design input while preserving its calculation method."""
        if multiplier <= 0:
            raise ValueError("Lighting multiplier must be positive")
        text = source_idf.read_text(encoding="utf-8", errors="replace")
        replacements: list[tuple[int, int, str]] = []
        trace: list[dict] = []
        methods = {
            "lightinglevel": (5, "Lighting_Level", lambda value: value * multiplier),
            "watts/area": (6, "Watts_per_Floor_Area", lambda value: value * multiplier),
            "watts/person": (7, "Watts_per_Person", lambda value: value * multiplier),
        }
        for fields, spans in _idf_objects(text):
            if fields[0].lower() != "lights" or len(fields) < 8:
                continue
            method = fields[4].strip().lower()
            if method not in methods:
                raise ValueError(f"Unsupported Lights calculation method {fields[4]!r} for {fields[1]}")
            index, field, transform = methods[method]
            try:
                baseline = float(fields[index])
            except ValueError as exc:
                raise ValueError(f"Lights object {fields[1]} has no valid {field}") from exc
            candidate = transform(baseline)
            replacements.append((*spans[index], f"{candidate:.12g}"))
            trace.append({"object": fields[1], "method": fields[4], "field": field,
                          "baseline": baseline, "candidate": candidate})
        if not trace:
            raise ValueError("Model has no optimizable Lights objects")
        for start, end, value in sorted(replacements, reverse=True):
            text = text[:start] + value + text[end:]
        target_idf.parent.mkdir(parents=True, exist_ok=True)
        target_idf.write_text(text, encoding="utf-8")
        return trace

    async def simulate_electricity(self, model_path: Path, weather_path: Path, output_directory: Path, default_year: int) -> SimulationEvidence:
        output_directory.mkdir(parents=True, exist_ok=True)
        result = await self._invoke("run_energyplus_simulation", {
            "idf_path": str(model_path), "weather_file": str(weather_path),
            "output_directory": str(output_directory), "annual": True,
            "design_day": False, "readvars": True, "expandobjects": True,
        })
        if not result.get("success"):
            raise RuntimeError(result.get("error_details") or result.get("error") or "EnergyPlus simulation failed")
        csv_files = sorted(output_directory.glob("*Meter.csv")) + sorted(output_directory.glob("*.csv"))
        last_error = None
        for path in dict.fromkeys(csv_files):
            try:
                series = load_energyplus_meter_csv(path, default_year=default_year)
                return SimulationEvidence(str(model_path), str(output_directory), str(path), series, result)
            except ValueError as exc:
                last_error = exc
        raise FileNotFoundError(f"No CSV containing hourly Electricity:Facility was generated in {output_directory}: {last_error}")
