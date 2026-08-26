"""Measured and EnergyPlus time-series loading with explicit alignment rules."""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class TimeSeries:
    timestamps: tuple[datetime, ...]
    values: tuple[float, ...]
    unit: str
    value_name: str
    source: str

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.values) or not self.timestamps:
            raise ValueError("Time series must have equal non-empty timestamp and value arrays")


def _parse_timestamp(raw: str, default_year: int | None = None) -> datetime:
    value = raw.strip()
    formats = (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    if default_year is not None:
        compact = re.sub(r"\s+", " ", value)
        end_of_day = re.match(r"^(\d{1,2}/\d{1,2}) 24:(\d{2}):(\d{2})$", compact)
        if end_of_day:
            base = datetime.strptime(f"{default_year}/{end_of_day.group(1)} 00:{end_of_day.group(2)}:{end_of_day.group(3)}", "%Y/%m/%d %H:%M:%S")
            return base + timedelta(days=1)
        for fmt in ("%m/%d %H:%M:%S", "%m/%d %H:%M"):
            try:
                parsed = datetime.strptime(compact, fmt)
                # EnergyPlus may report 24:00 separately; ordinary outputs are
                # handled here while unsupported forms fail explicitly.
                return parsed.replace(year=default_year)
            except ValueError:
                pass
    raise ValueError(f"Invalid timestamp: {raw!r}")


def load_measured_csv(
    path: str | Path, *, value_column: str = "electricity", timestamp_column: str = "timestamp",
    unit: str = "kWh", missing: str = "reject",
) -> TimeSeries:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if missing not in {"reject", "drop"}:
        raise ValueError("missing must be 'reject' or 'drop'")
    rows: list[tuple[datetime, float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        absent = [name for name in (timestamp_column, value_column) if name not in headers]
        if absent:
            raise ValueError(f"Measured CSV missing required columns: {absent}")
        for line_number, row in enumerate(reader, start=2):
            raw_timestamp, raw_value = (row.get(timestamp_column) or "").strip(), (row.get(value_column) or "").strip()
            if not raw_timestamp or not raw_value:
                if missing == "drop":
                    continue
                raise ValueError(f"Missing measured value at CSV line {line_number}")
            try:
                timestamp = _parse_timestamp(raw_timestamp)
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"Invalid measured data at CSV line {line_number}: {exc}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite measured value at CSV line {line_number}")
            rows.append((timestamp, value))
    if not rows:
        raise ValueError("Measured CSV contains no usable rows")
    rows.sort(key=lambda item: item[0])
    if len({timestamp for timestamp, _ in rows}) != len(rows):
        raise ValueError("Measured CSV contains duplicate timestamps")
    return TimeSeries(tuple(x[0] for x in rows), tuple(x[1] for x in rows), unit, value_column, str(source))


def load_energyplus_meter_csv(
    path: str | Path, *, meter_name: str = "Electricity:Facility", default_year: int,
) -> TimeSeries:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        timestamp_column = next((h for h in headers if h.strip().lower() in {"date/time", "datetime", "timestamp"}), None)
        meter_candidates = [h for h in headers if meter_name.lower() in h.lower()]
        # Models often request the same meter at monthly, run-period, and
        # hourly frequencies. Calibration must use the hourly column; choosing
        # the first matching header would select a sparse monthly column.
        meter_column = next((h for h in meter_candidates if "(hourly)" in h.lower()), None)
        meter_column = meter_column or (meter_candidates[0] if meter_candidates else None)
        if not timestamp_column or not meter_column:
            raise ValueError(f"EnergyPlus CSV lacks Date/Time or meter {meter_name!r}; columns={headers}")
        unit_match = re.search(r"\[([^]]+)\]", meter_column)
        raw_unit = unit_match.group(1) if unit_match else "unknown"
        rows: list[tuple[datetime, float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = _parse_timestamp(row[timestamp_column], default_year)
                value = float(row[meter_column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid EnergyPlus meter row at line {line_number}: {exc}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite EnergyPlus value at CSV line {line_number}")
            if raw_unit.lower() in {"j", "joule", "joules"}:
                value /= 3_600_000.0
            elif raw_unit.lower() not in {"kwh", "kw-h"}:
                raise ValueError(f"Unsupported EnergyPlus electricity unit: {raw_unit}")
            rows.append((timestamp, value))
    return TimeSeries(tuple(x[0] for x in rows), tuple(x[1] for x in rows), "kWh", meter_name, str(source))


def align_series(measured: TimeSeries, simulated: TimeSeries, *, policy: str = "month_day_time") -> tuple[list[float], list[float], dict]:
    if measured.unit.lower() != simulated.unit.lower():
        raise ValueError(f"Unit mismatch: measured={measured.unit}, simulated={simulated.unit}")
    if policy == "exact":
        key = lambda value: value
    elif policy == "month_day_time":
        key = lambda value: (value.month, value.day, value.hour, value.minute, value.second)
    else:
        raise ValueError("alignment policy must be 'exact' or 'month_day_time'")
    simulated_map = {key(t): value for t, value in zip(simulated.timestamps, simulated.values)}
    measured_values, simulated_values, missing = [], [], []
    for timestamp, value in zip(measured.timestamps, measured.values):
        match = key(timestamp)
        if match not in simulated_map:
            missing.append(timestamp.isoformat())
            continue
        measured_values.append(value)
        simulated_values.append(simulated_map[match])
    if missing:
        raise ValueError(f"Simulation is missing {len(missing)} measured timestamps; first={missing[:3]}")
    if len(measured_values) != len(simulated.values):
        extra = len(simulated.values) - len(measured_values)
    else:
        extra = 0
    return measured_values, simulated_values, {
        "policy": policy, "aligned_count": len(measured_values), "missing_simulated": 0,
        "extra_simulated": max(0, extra), "unit": measured.unit,
    }
