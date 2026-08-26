"""Persisted background execution for long EnergyPlus calibration jobs."""
from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .occupancy import OccupancyCalibrationRequest, calibrate_occupancy
from energy_agent.simulation.runner import EnergyPlusRunner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CalibrationJobManager:
    def __init__(self, root: Path, runner: EnergyPlusRunner):
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.tasks: dict[str, asyncio.Task] = {}
        for status_file in self.root.glob("*/status.json"):
            try:
                state = json.loads(status_file.read_text(encoding="utf-8"))
                if state.get("state") in {"queued", "running"}:
                    state.update({"state": "interrupted", "error": "Application restarted before job completion", "updated_at": _now()})
                    status_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
            except Exception:
                continue

    def _directory(self, job_id: str) -> Path:
        if not job_id.startswith("cal_") or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in job_id):
            raise KeyError("Invalid calibration job ID")
        return self.root / job_id

    def _write(self, job_id: str, state: dict) -> None:
        directory = self._directory(job_id); directory.mkdir(parents=True, exist_ok=True)
        state["job_id"] = job_id; state["updated_at"] = _now()
        (directory / "status.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def status(self, job_id: str) -> dict:
        path = self._directory(job_id) / "status.json"
        if not path.is_file(): raise KeyError(f"Calibration job not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    async def start(self, request: OccupancyCalibrationRequest) -> dict:
        request.validate()
        job_id = f"cal_{secrets.token_hex(8)}"
        directory = self._directory(job_id)
        request = replace(request, output_directory=str(directory))
        state = {"state": "queued", "created_at": _now(), "request": asdict(request), "progress": {"stage": "queued"}}
        self._write(job_id, state)
        self.tasks[job_id] = asyncio.create_task(self._execute(job_id, request))
        return self.status(job_id)

    async def _execute(self, job_id: str, request: OccupancyCalibrationRequest) -> None:
        state = self.status(job_id); state["state"] = "running"; self._write(job_id, state)
        async def progress(payload: dict) -> None:
            current = self.status(job_id); current["progress"] = payload
            if payload.get("stage") == "completed": current["state"] = "completed"
            self._write(job_id, current)
        try:
            result = await calibrate_occupancy(request, self.runner, progress)
            state = self.status(job_id); state.update({"state": "completed", "result": result, "completed_at": _now()})
            self._write(job_id, state)
        except Exception as exc:
            state = self.status(job_id); state.update({"state": "failed", "error": str(exc), "completed_at": _now()})
            self._write(job_id, state)

    def files(self, job_id: str) -> list[dict]:
        directory = self._directory(job_id)
        if not directory.is_dir(): raise KeyError(f"Calibration job not found: {job_id}")
        return [{"name": str(path.relative_to(directory)).replace("\\", "/"), "size": path.stat().st_size}
                for path in sorted(directory.rglob("*")) if path.is_file()]

    def file(self, job_id: str, relative: str) -> Path:
        directory = self._directory(job_id).resolve(); target = (directory / relative).resolve()
        if directory not in target.parents or not target.is_file(): raise KeyError("Calibration artifact not found")
        return target
