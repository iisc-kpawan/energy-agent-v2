import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from energy_agent.analytics.timeseries import TimeSeries
from energy_agent.calibration.occupancy import OccupancyCalibrationRequest, calibrate_occupancy
from energy_agent.simulation.runner import EnergyPlusRunner, SimulationEvidence, _json_payload


def test_mcp_content_blocks_are_normalized_to_json():
    assert _json_payload([{"type": "text", "text": "Result:\n{\"success\": true}"}]) == {"success": True}


def test_occupancy_multiplier_handles_all_people_calculation_methods():
    inspection = {"people_objects": [
        {"name": "Direct", "calculation_method": "people", "number_of_people": 10},
        {"name": "Density", "calculation_method": "People/Area", "people_per_area": 0.1},
        {"name": "Inverse", "calculation_method": "Area/Person", "area_per_person": 20},
    ]}
    modifications, trace = EnergyPlusRunner.occupancy_modifications(inspection, 0.5)
    values = [next(iter(item["field_updates"].values())) for item in modifications]
    assert values == pytest.approx([5, 0.05, 40])
    assert [item["method"] for item in trace] == ["People", "People/Area", "Area/Person"]


class FakeRunner:
    def __init__(self, root: Path):
        self.root = root
        self.current_multiplier = 1.0

    def resolve_file(self, raw_path, suffix):
        path = Path(raw_path)
        assert path.suffix == suffix
        return path

    async def inspect_people(self, _idf):
        return {"success": True, "people_objects": [
            {"name": "People", "calculation_method": "People", "number_of_people": 10}
        ]}

    async def configure_hourly_electricity(self, source, target):
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text(source.read_text())
        return target

    async def apply_occupancy_multiplier(self, _source, target, _inspection, multiplier):
        self.current_multiplier = multiplier
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text("candidate")
        return [{"object": "People", "baseline": 10, "candidate": 10 * multiplier}]

    async def simulate_electricity(self, model, _weather, output, default_year):
        output.mkdir(parents=True, exist_ok=True)
        timestamps = tuple(datetime(default_year, 1, 1) + timedelta(hours=i) for i in range(4))
        # Hidden reference multiplier is 0.75; exact measured values are 75.
        values = tuple(100 * self.current_multiplier for _ in timestamps)
        meter = output / "resultMeter.csv"; meter.write_text("test")
        return SimulationEvidence(str(model), str(output), str(meter), TimeSeries(timestamps, values, "kWh", "Electricity:Facility", str(meter)), {"success": True})


def test_real_calibration_loop_improves_baseline_and_records_trace(tmp_path):
    idf = tmp_path / "model.idf"; idf.write_text("Version,26.1;")
    epw = tmp_path / "weather.epw"; epw.write_text("weather")
    measured = tmp_path / "measured.csv"
    measured.write_text("timestamp,electricity\n" + "\n".join(
        f"2026-01-01 {hour:02d}:00,75" for hour in range(4)
    ))
    request = OccupancyCalibrationRequest(
        str(idf), str(epw), str(measured), lower_bound=0.5, upper_bound=1.5,
        maximum_evaluations=9, stopping_tolerance=0.0001,
        output_directory=str(tmp_path / "runs"),
    )
    result = asyncio.run(calibrate_occupancy(request, FakeRunner(tmp_path)))
    assert result["mode"] == "energyplus-backed-real-calibration"
    assert result["calibrated"]["objective"] < result["baseline"]["objective"]
    assert result["calibrated"]["occupancy_multiplier"] == pytest.approx(0.75)
    assert result["energyplus_evaluations"] >= 4
    assert (tmp_path / "runs" / "calibration_result.json").is_file()
    assert (tmp_path / "runs" / "calibration_iterations.csv").is_file()
