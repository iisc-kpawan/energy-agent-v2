import csv
import math
from datetime import datetime

import pytest

from energy_agent.analytics.metrics import (
    calculate_calibration_metrics, calculate_cvrmse, calculate_mae,
    calculate_mape, calculate_nmbe, calculate_rmse,
)
from energy_agent.analytics.timeseries import TimeSeries, align_series, load_measured_csv


def test_metrics_have_expected_values_and_units():
    measured, simulated = [10.0, 20.0, 30.0], [12.0, 18.0, 33.0]
    assert calculate_rmse(measured, simulated) == pytest.approx(math.sqrt(17 / 3))
    assert calculate_mae(measured, simulated) == pytest.approx(7 / 3)
    assert calculate_nmbe(measured, simulated, 1) == pytest.approx(7.5)
    assert calculate_cvrmse(measured, simulated, 1) == pytest.approx(math.sqrt(17 / 2) / 20 * 100)
    result = calculate_calibration_metrics(measured, simulated, unit="kWh", fitted_parameters=1)
    assert result["value_unit"] == "kWh"
    assert result["sample_count"] == 3


def test_mape_reports_zero_exclusions():
    value, excluded = calculate_mape([0, 10], [5, 12])
    assert value == pytest.approx(20)
    assert excluded == 1
    assert calculate_mape([0, 0], [1, 2]) == (None, 2)


@pytest.mark.parametrize("measured,simulated", [([1], [1]), ([1, 2], [1]), ([1, math.nan], [1, 2])])
def test_metrics_reject_invalid_arrays(measured, simulated):
    with pytest.raises(ValueError):
        calculate_rmse(measured, simulated)


def test_measured_csv_is_validated_and_sorted(tmp_path):
    path = tmp_path / "measured.csv"
    path.write_text("timestamp,electricity\n2026-01-01 01:00,39.8\n2026-01-01 00:00,42.1\n", encoding="utf-8")
    series = load_measured_csv(path, unit="kWh")
    assert series.timestamps[0] == datetime(2026, 1, 1, 0, 0)
    assert series.values == (42.1, 39.8)


def test_measured_csv_rejects_missing_columns_invalid_timestamp_and_nan(tmp_path):
    missing = tmp_path / "missing.csv"; missing.write_text("time,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_measured_csv(missing)
    invalid = tmp_path / "invalid.csv"; invalid.write_text("timestamp,electricity\nwrong,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid measured data"):
        load_measured_csv(invalid)
    nan = tmp_path / "nan.csv"; nan.write_text("timestamp,electricity\n2026-01-01 00:00,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        load_measured_csv(nan)


def test_alignment_can_ignore_calendar_year_but_not_missing_intervals():
    measured = TimeSeries((datetime(2026, 1, 1, 0),), (10.0,), "kWh", "electricity", "meter")
    simulated = TimeSeries((datetime(2002, 1, 1, 0),), (9.0,), "kWh", "Electricity:Facility", "sim")
    m, s, metadata = align_series(measured, simulated)
    assert (m, s) == ([10.0], [9.0])
    assert metadata["aligned_count"] == 1
    missing = TimeSeries((datetime(2002, 1, 1, 1),), (9.0,), "kWh", "Electricity:Facility", "sim")
    with pytest.raises(ValueError, match="missing 1"):
        align_series(measured, missing)
