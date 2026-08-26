"""LangChain tools that start and inspect deterministic calibration jobs."""
from __future__ import annotations

import json
from langchain_core.tools import tool

from .jobs import CalibrationJobManager
from .occupancy import OccupancyCalibrationRequest


def calibration_tools(manager: CalibrationJobManager) -> list:
    @tool("calibrate_occupancy")
    async def start_real_occupancy_calibration(
        idf_path: str, epw_path: str, measured_data_path: str,
        measured_value_column: str = "electricity", timestamp_column: str = "timestamp",
        measured_unit: str = "kWh", lower_bound: float = 0.5, upper_bound: float = 1.5,
        maximum_evaluations: int = 12, objective: str = "rmse",
    ) -> str:
        """Start REAL EnergyPlus-backed occupancy calibration against fixed measured CSV data. Returns a persistent job ID immediately; it never uses a surrogate equation."""
        request = OccupancyCalibrationRequest(
            idf_path=idf_path, epw_path=epw_path, measured_data_path=measured_data_path,
            measured_value_column=measured_value_column, timestamp_column=timestamp_column,
            measured_unit=measured_unit, lower_bound=lower_bound, upper_bound=upper_bound,
            maximum_evaluations=maximum_evaluations, objective=objective,
        )
        return json.dumps(await manager.start(request))

    @tool("get_calibration_job")
    def get_real_calibration_job(job_id: str) -> str:
        """Get progress or final numerical evidence for a persistent real EnergyPlus calibration job."""
        return json.dumps(manager.status(job_id))

    return [start_real_occupancy_calibration, get_real_calibration_job]
