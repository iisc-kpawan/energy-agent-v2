from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from energy_agent.engineering_studies import EnergyPlusStudyRequest
from energy_agent.simulation.runner import EnergyPlusRunner, _idf_objects


def test_idf_parser_ignores_comments_and_preserves_empty_fields():
    objects = _idf_objects("! heading\nPeople, P1, Zone, Sch, People, 10, , , 0.3;")
    assert objects[0][0] == ["People", "P1", "Zone", "Sch", "People", "10", "", "", "0.3"]


def test_safe_lighting_multiplier_changes_only_active_design_field(tmp_path: Path):
    source = tmp_path / "source.idf"
    target = tmp_path / "target.idf"
    source.write_text(
        "Version,26.1;\nLights, L1, Zone, AlwaysOn, LightingLevel, 100, , , 0.0;\n"
        "Lights, L2, Zone, AlwaysOn, Watts/Area, , 8.5, , 0.0;\n",
        encoding="utf-8",
    )
    runner = object.__new__(EnergyPlusRunner)
    trace = asyncio.run(runner.apply_lighting_multiplier(source, target, 0.8))
    fields = [item[0] for item in _idf_objects(target.read_text(encoding="utf-8"))]
    assert fields[1][5] == "80"
    assert fields[2][6] == "6.8"
    assert len(trace) == 2
    assert source.read_text(encoding="utf-8").find("100") > 0


def test_real_study_bounds_must_include_baseline():
    request = EnergyPlusStudyRequest("model.idf", "weather.epw", lower_bound=0.5, upper_bound=0.9)
    with pytest.raises(ValueError, match="baseline"):
        request.validate("optimization")


def test_optimization_scope_is_explicit():
    request = EnergyPlusStudyRequest("model.idf", "weather.epw", parameter="occupancy_multiplier")
    with pytest.raises(ValueError, match="lighting_multiplier only"):
        request.validate("optimization")
