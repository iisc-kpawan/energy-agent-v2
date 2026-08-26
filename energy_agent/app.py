from __future__ import annotations

import asyncio
import base64
import copy
import logging
import os
import socket
import subprocess
import shutil
import html
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import Settings
from .memory import ContextBuilder
from .orchestrator import MultiAgentOrchestrator, SPECIALISTS
from .analysis import analysis_tools, calibration_metrics, guideline14_assessment, demo_optimization, demo_calibration, demo_sensitivity, export_result
from .calibration.jobs import CalibrationJobManager
from .calibration.occupancy import OccupancyCalibrationRequest
from .calibration.tools import calibration_tools
from .engineering_studies import EngineeringStudyManager, EnergyPlusStudyRequest, engineering_study_tools
from .simulation.runner import EnergyPlusRunner
from .storage import Store

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
settings = Settings.load(ROOT)
settings.ensure_directories()
store = Store(settings.database, settings.artifacts_dir)
memory = ContextBuilder(store, settings.memory_turn_limit, settings.summary_char_limit)
logger = logging.getLogger("energy_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

mcp_client = None
orchestrator: MultiAgentOrchestrator | None = None
init_status, init_error = "pending", ""
chat_jobs: dict[str, dict] = {}
calibration_job_manager: CalibrationJobManager | None = None
engineering_study_manager: EngineeringStudyManager | None = None


def _patch_array_items(node):
    if not isinstance(node, dict): return
    if node.get("type") == "array" and "items" not in node: node["items"] = {"type": "string"}
    for value in node.values():
        if isinstance(value, dict): _patch_array_items(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict): _patch_array_items(item)


def _fix_tool_schemas(tools):
    for tool in tools:
        schema_cls = getattr(tool, "args_schema", None)
        if schema_cls is None: continue
        try:
            original = schema_cls.model_json_schema.__func__ if hasattr(schema_cls.model_json_schema, "__func__") else schema_cls.model_json_schema
            def make(fn):
                @classmethod
                def patched(cls, *args, **kwargs):
                    try: schema = copy.deepcopy(fn(cls, *args, **kwargs))
                    except TypeError: schema = copy.deepcopy(fn(*args, **kwargs))
                    _patch_array_items(schema); return schema
                return patched
            schema_cls.model_json_schema = make(original)
        except Exception: pass
    return tools


def docker_status() -> tuple[bool, str]:
    try:
        docker = docker_command()
        result = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=30)
        if result.returncode: return False, result.stderr.strip() or "Docker daemon is unavailable"
        return True, ""
    except FileNotFoundError: return False, "Docker is not installed or not on PATH"
    except subprocess.TimeoutExpired: return False, "Docker command timed out"


def docker_command() -> str:
    """Find Docker for both system-wide and recommended per-user Windows installs."""
    discovered = shutil.which("docker")
    if discovered:
        return discovered
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe"
        if candidate.is_file():
            return str(candidate)
    return "docker"


async def initialize_mcp():
    global mcp_client, orchestrator, calibration_job_manager, engineering_study_manager, init_status, init_error
    init_status = "connecting"
    try:
        mcp_url = os.getenv("ENERGYPLUS_MCP_URL", "").strip()
        if mcp_url:
            token = os.getenv("ENERGYPLUS_MCP_TOKEN", "").strip()
            config = {"url": mcp_url, "transport": "streamable_http"}
            if token:
                config["headers"] = {"Authorization": f"Bearer {token}"}
            mcp_client = MultiServerMCPClient({"energyplus": config})
        else:
            ok, error = docker_status()
            if not ok:
                init_status, init_error = "error", error; return
            args = ["run", "--rm", "-i", "--user", "root", "-e", "UV_PROJECT_ENVIRONMENT=/opt/eplus-venv",
                    "-e", "UV_LINK_MODE=copy", "-v", f"{ROOT / 'EnergyPlus-MCP'}:/workspace",
                    "-v", "energyplus-mcp-deps:/root/.cache/uv", "-v", "energyplus-mcp-venv:/opt/eplus-venv",
                    "-w", "/workspace/energyplus-mcp-server",
                    "energyplus-mcp-dev", "uv", "run", "--no-dev", "python", "-m", "energyplus_mcp_server.server"]
            mcp_client = MultiServerMCPClient({"energyplus": {"command": docker_command(), "args": args, "transport": "stdio"}})
        # Compose may start the lightweight API before the EnergyPlus image has
        # finished its slower initialization. Retry the dependency instead of
        # permanently failing the application lifespan on the first race.
        last_error = None
        for attempt in range(15):
            try:
                mcp_tools = await mcp_client.get_tools()
                break
            except Exception as exc:
                last_error = exc
                init_status = f"connecting ({attempt + 1}/15)"
                await asyncio.sleep(2)
        else:
            raise RuntimeError(f"EnergyPlus MCP did not become ready: {last_error}")
        allowed_roots = [ROOT, settings.artifacts_dir, ROOT / "EnergyPlus-MCP" / "energyplus-mcp-server"]
        output_root = os.getenv("ENERGYPLUS_OUTPUT_ROOT", "").strip()
        if output_root: allowed_roots.append(Path(output_root))
        runner = EnergyPlusRunner(mcp_tools, allowed_roots)
        calibration_job_manager = CalibrationJobManager(settings.data_dir / "calibration_jobs", runner)
        engineering_study_manager = EngineeringStudyManager(settings.data_dir / "engineering_studies", runner)
        tools = _fix_tool_schemas(mcp_tools + analysis_tools() + calibration_tools(calibration_job_manager)
                                  + engineering_study_tools(engineering_study_manager))
        orchestrator = MultiAgentOrchestrator(tools, memory, store, settings.google_api_key,
                                               settings.default_model, settings.max_parallel_agents)
        init_status = "ready"
        logger.info("Multi-agent system ready with %d MCP tools", len(tools))
    except Exception as exc:
        logger.exception("MCP initialization failed")
        init_status, init_error = "error", str(exc)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(initialize_mcp())
    yield
    if not task.done(): task.cancel()


app = FastAPI(title="EnergyPlus Agentic Engineering Platform", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def require_production_login(request: Request, call_next):
    """Optional HTTP Basic gate; enabled whenever APP_PASSWORD is configured."""
    expected_password = os.getenv("APP_PASSWORD", "")
    if not expected_password or request.url.path == "/api/health":
        return await call_next(request)
    expected_user = os.getenv("APP_USERNAME", "energy-admin")
    supplied = request.headers.get("authorization", "")
    valid = False
    if supplied.startswith("Basic "):
        try:
            decoded = base64.b64decode(supplied[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
            valid = secrets.compare_digest(username, expected_user) and secrets.compare_digest(password, expected_password)
        except (ValueError, UnicodeDecodeError):
            pass
    if not valid:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Energy Agent V2"'})
    return await call_next(request)

app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5000").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
static_dir = ROOT / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(): return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent_ready": init_status == "ready", "init_status": init_status,
            "init_error": init_error, "model": settings.default_model, "api_key_set": bool(settings.google_api_key),
            "architecture": "dual-mode-agentic-v2", "version": "2.0.0",
            "specialists": [s.name for s in SPECIALISTS.values()]}


@app.get("/api/v2/capabilities")
async def v2_capabilities():
    return {"version": "2.0.0", "modes": ["direct", "multi-agent"],
            "analysis": ["real EnergyPlus occupancy calibration", "real EnergyPlus lighting optimization",
                         "real EnergyPlus lighting/occupancy sensitivity", "deterministic calibration metrics",
                         "surrogate optimization demo", "surrogate calibration demo", "surrogate sensitivity demo"],
            "guardrails": {"max_demo_evaluations": 500, "seeded_reproducibility": True,
                           "human_approval_for_real_high-budget_runs": True},
            "sources": ["https://besos.readthedocs.io/en/stable/example-notebooks/How-to-Guides/BuildingOptimization.html",
                        "https://doi.org/10.1080/19401493.2026.2653969"]}


def _analysis_export(result: dict, run_name: str) -> dict:
    paths = export_result(result, settings.data_dir / "analysis", run_name)
    return {kind: f"/api/v2/analysis/files/{Path(path).name}" for kind, path in paths.items() if Path(path).is_file()}


@app.get("/api/v2/analysis/files/{filename}")
async def analysis_file(filename: str):
    if Path(filename).name != filename: raise HTTPException(400, "Invalid filename")
    target = (settings.data_dir / "analysis" / filename).resolve()
    root = (settings.data_dir / "analysis").resolve()
    if root not in target.parents or not target.is_file(): raise HTTPException(404, "Analysis artifact not found")
    return FileResponse(target, filename=target.name)


@app.post("/api/v2/optimization/demo")
async def optimization_demo(request: Request):
    body = await request.json(); evaluations = int(body.get("evaluations", 30)); seed = int(body.get("seed", 42))
    result = demo_optimization(evaluations, seed)
    run_name = f"optimization-demo-{os.urandom(5).hex()}"
    result["artifacts"] = _analysis_export(result, run_name)
    return result


@app.post("/api/v2/calibration/metrics")
async def calibration_metric_api(request: Request):
    body = await request.json(); metrics = calibration_metrics(body.get("measured", []), body.get("simulated", []))
    return {"metrics": metrics, "assessment": guideline14_assessment(metrics, str(body.get("interval", "monthly")))}


@app.post("/api/v2/calibration/demo")
async def calibration_demo(request: Request):
    body = await request.json(); evaluations = int(body.get("evaluations", 40)); seed = int(body.get("seed", 42))
    result = demo_calibration(evaluations, seed)
    run_name = f"calibration-demo-{os.urandom(5).hex()}"
    result["artifacts"] = _analysis_export(result, run_name)
    return result


@app.post("/api/v2/sensitivity/demo")
async def sensitivity_demo(request: Request):
    body = await request.json(); trajectories = int(body.get("trajectories", 8)); seed = int(body.get("seed", 42))
    result = demo_sensitivity(trajectories, seed)
    run_name = f"sensitivity-demo-{os.urandom(5).hex()}"
    result["artifacts"] = _analysis_export(result, run_name)
    return result


@app.get("/api/models")
async def models():
    choices = [{"id": settings.default_model, "name": settings.default_model, "provider": "google"}]
    for name in ("gemini-3.1-flash-lite", "gemini-3-flash-preview"):
        if name not in {x["id"] for x in choices}: choices.append({"id": name, "name": name, "provider": "google"})
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=1.5)
            for item in response.json().get("models", []):
                choices.append({"id": item["name"], "name": item["name"], "provider": "ollama"})
    except Exception: pass
    return {"models": choices}


@app.get("/api/tools")
async def tools():
    if not orchestrator: return {"tools": [], "error": init_error or "MCP is connecting"}
    return {"tools": [{"name": t.name, "description": t.description} for t in orchestrator.tools]}


@app.get("/api/agents")
async def agents():
    return {"agents": [{"id": s.key, "name": s.name, "purpose": s.purpose, "tool_count": len(s.tools)} for s in SPECIALISTS.values()]}


@app.get("/api/projects")
async def projects(): return {"projects": store.list_projects()}


@app.post("/api/projects")
async def create_project(request: Request):
    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name: raise HTTPException(400, "Project name is required")
    project = store.create_project(name)
    conversation = store.create_conversation(project["id"])
    return {"project": project, "conversation": conversation}


@app.post("/api/projects/{project_id}/conversations")
async def create_conversation(project_id: str, request: Request):
    body = await request.json()
    return store.create_conversation(project_id, str(body.get("title", "New conversation"))[:100])


@app.get("/api/projects/{project_id}/conversations")
async def conversations(project_id: str): return {"conversations": store.list_conversations(project_id)}


@app.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(conversation_id: str):
    store.get_conversation(conversation_id)
    return {"messages": store.messages(conversation_id)}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    deleted = store.delete_conversation(conversation_id)
    return {"deleted": deleted, "conversation_id": conversation_id}


@app.post("/api/projects/{project_id}/upload")
async def upload(project_id: str, file: UploadFile = File(...), label: str = Form("Uploaded model")):
    store.get_project(project_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".idf", ".epw", ".csv"}: raise HTTPException(400, "Only .idf, .epw, and measured .csv files are accepted")
    incoming = settings.data_dir / "incoming"
    incoming.mkdir(exist_ok=True)
    temp = incoming / f"upload-{os.urandom(8).hex()}{suffix}"
    total = 0
    with temp.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > 100 * 1024 * 1024:
                temp.unlink(missing_ok=True); raise HTTPException(413, "File exceeds 100 MB")
            output.write(chunk)
    try:
        artifact_type = "measured_csv" if suffix == ".csv" else suffix[1:]
        artifact = store.import_artifact(project_id, temp, artifact_type, {"original_name": file.filename, "label": label})
        version = store.create_model_version(project_id, artifact["id"], label) if suffix == ".idf" else None
        return {"artifact": artifact, "model_version": version}
    finally: temp.unlink(missing_ok=True)


@app.get("/api/projects/{project_id}/artifacts")
async def project_artifacts(project_id: str):
    return {"artifacts": store.list_artifacts(project_id)}


@app.post("/api/v2/calibration/occupancy")
async def start_occupancy_calibration(request: Request):
    if not calibration_job_manager: raise HTTPException(503, init_error or "Calibration service is connecting")
    body = await request.json()
    try:
        specification = OccupancyCalibrationRequest(**body)
        return await calibration_job_manager.start(specification)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v2/calibration/jobs/{job_id}")
async def calibration_job(job_id: str):
    if not calibration_job_manager: raise HTTPException(503, "Calibration service is unavailable")
    try: return calibration_job_manager.status(job_id)
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


@app.get("/api/v2/calibration/jobs/{job_id}/files")
async def calibration_job_files(job_id: str):
    if not calibration_job_manager: raise HTTPException(503, "Calibration service is unavailable")
    try:
        files = calibration_job_manager.files(job_id)
        for item in files:
            item["url"] = f"/api/v2/calibration/jobs/{job_id}/files/{item['name']}"
        return {"job_id": job_id, "files": files}
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


@app.get("/api/v2/calibration/jobs/{job_id}/files/{file_path:path}")
async def calibration_job_file(job_id: str, file_path: str):
    if not calibration_job_manager: raise HTTPException(503, "Calibration service is unavailable")
    try:
        target = calibration_job_manager.file(job_id, file_path)
        return FileResponse(target, filename=target.name)
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


@app.post("/api/v2/studies/{kind}")
async def start_energyplus_study(kind: str, request: Request):
    if kind not in {"optimization", "sensitivity"}: raise HTTPException(404, "Unknown study type")
    if not engineering_study_manager: raise HTTPException(503, init_error or "Engineering study service is connecting")
    try:
        specification = EnergyPlusStudyRequest(**await request.json())
        return await engineering_study_manager.start(kind, specification)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/v2/studies/jobs/{job_id}")
async def energyplus_study_job(job_id: str):
    if not engineering_study_manager: raise HTTPException(503, "Engineering study service is unavailable")
    try: return engineering_study_manager.status(job_id)
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


@app.get("/api/v2/studies/jobs/{job_id}/files")
async def energyplus_study_files(job_id: str):
    if not engineering_study_manager: raise HTTPException(503, "Engineering study service is unavailable")
    try:
        files = engineering_study_manager.files(job_id)
        for item in files: item["url"] = f"/api/v2/studies/jobs/{job_id}/files/{item['name']}"
        return {"job_id": job_id, "files": files}
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


@app.get("/api/v2/studies/jobs/{job_id}/files/{file_path:path}")
async def energyplus_study_file(job_id: str, file_path: str):
    if not engineering_study_manager: raise HTTPException(503, "Engineering study service is unavailable")
    try:
        target = engineering_study_manager.file(job_id, file_path)
        return FileResponse(target, filename=target.name)
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = str(body.get("message", "")).strip()
    conversation_id = str(body.get("conversation_id", "")).strip()
    model = str(body.get("model", settings.default_model)).strip()
    mode = str(body.get("mode", "multi-agent")).strip()
    if not message or not conversation_id: raise HTTPException(400, "message and conversation_id are required")
    if model.startswith("gemini") and not settings.google_api_key:
        raise HTTPException(503, "GOOGLE_API_KEY is not configured")
    if not orchestrator: raise HTTPException(503, init_error or "Multi-agent system is still connecting")
    if mode not in {"multi-agent", "direct"}: raise HTTPException(400, "mode must be multi-agent or direct")
    try: return await orchestrator.run(conversation_id, message, model, mode)
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(502, f"Agent run failed: {exc}") from exc


async def _execute_chat_job(run_id: str, conversation_id: str, message: str, model: str, mode: str) -> None:
    try:
        result = await orchestrator.run(conversation_id, message, model, mode, run_id=run_id)
        chat_jobs[run_id] = {"state": "completed", "result": result}
    except Exception as exc:
        logger.exception("Background chat request failed")
        store.update_run(run_id, status="failed", event={"agent": "system", "status": "failed", "error": str(exc)}, error=str(exc))
        chat_jobs[run_id] = {"state": "failed", "error": f"Agent run failed: {exc}"}


@app.post("/api/chat/start")
async def start_chat(request: Request):
    body = await request.json()
    message = str(body.get("message", "")).strip()
    conversation_id = str(body.get("conversation_id", "")).strip()
    model = str(body.get("model", settings.default_model)).strip()
    mode = str(body.get("mode", "multi-agent")).strip()
    if not message or not conversation_id: raise HTTPException(400, "message and conversation_id are required")
    if model.startswith("gemini") and not settings.google_api_key: raise HTTPException(503, "GOOGLE_API_KEY is not configured")
    if not orchestrator: raise HTTPException(503, init_error or "Multi-agent system is still connecting")
    if mode not in {"multi-agent", "direct"}: raise HTTPException(400, "mode must be multi-agent or direct")
    run_id = store.create_run(conversation_id)
    chat_jobs[run_id] = {"state": "running"}
    asyncio.create_task(_execute_chat_job(run_id, conversation_id, message, model, mode))
    return {"run_id": run_id, "state": "running"}


@app.get("/api/chat/jobs/{run_id}")
async def chat_job(run_id: str):
    job = chat_jobs.get(run_id)
    if not job:
        try:
            run = store.get_run(run_id)
            return {"run_id": run_id, "state": run["status"], "run": run}
        except KeyError as exc: raise HTTPException(404, str(exc)) from exc
    return {"run_id": run_id, **job, "run": store.get_run(run_id)}


@app.get("/api/runs/{run_id}")
async def run_status(run_id: str):
    try: return store.get_run(run_id)
    except KeyError as exc: raise HTTPException(404, str(exc)) from exc


def _run_output_directory(run_id: str) -> Path:
    run = store.get_run(run_id)
    for message in reversed(store.messages(run["conversation_id"])):
        metadata = message.get("metadata", {})
        if metadata.get("run_id") != run_id: continue
        for call in reversed(metadata.get("tools", [])):
            raw = (call.get("args") or {}).get("output_directory")
            if raw and "/outputs/" in raw:
                candidate = Path(raw)
                resolved = candidate.resolve()
                root = Path(os.getenv(
                    "ENERGYPLUS_OUTPUT_ROOT",
                    "/workspace/energyplus-mcp-server/outputs",
                )).resolve()
                if resolved.is_dir() and (resolved == root or root in resolved.parents):
                    return resolved
    raise HTTPException(404, "No completed simulation output is attached to this run")


@app.get("/api/runs/{run_id}/artifacts")
async def run_artifacts(run_id: str):
    directory = _run_output_directory(run_id)
    run = store.get_run(run_id)
    files = [{"name": p.name, "size": p.stat().st_size,
              "url": f"/api/runs/{run_id}/files/{p.name}"}
             for p in sorted(directory.iterdir()) if p.is_file()]
    return {"run_id": run_id, "status": run["status"], "agents": run["route"],
            "events": run["events"], "output_directory": str(directory), "files": files,
            "geometry_url": f"/api/runs/{run_id}/geometry.svg" if any(p.suffix.lower() == ".dxf" for p in directory.iterdir()) else None}


@app.get("/api/runs/{run_id}/files/{filename}")
async def download_run_file(run_id: str, filename: str):
    if Path(filename).name != filename: raise HTTPException(400, "Invalid filename")
    target = _run_output_directory(run_id) / filename
    if not target.is_file(): raise HTTPException(404, "Output file not found")
    return FileResponse(target, filename=target.name)


@app.get("/api/runs/{run_id}/geometry.svg")
async def run_geometry(run_id: str):
    directory = _run_output_directory(run_id)
    dxf = next(iter(sorted(directory.glob("*.dxf"))), None)
    if not dxf: raise HTTPException(404, "Geometry output not found")
    lines = dxf.read_text(encoding="utf-8", errors="replace").splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    faces, current = [], None
    for code, value in pairs:
        if code == "0":
            if current: faces.append(current)
            current = {} if value.upper() == "3DFACE" else None
        elif current is not None and code in {"10","20","30","11","21","31","12","22","32","13","23","33"}:
            try: current[code] = float(value)
            except ValueError: pass
    if current: faces.append(current)
    polygons = []
    for face in faces:
        pts = []
        for n in range(4):
            x, y, z = face.get(str(10+n)), face.get(str(20+n)), face.get(str(30+n))
            if None not in (x, y, z): pts.append((x - .65*y, z + .28*(x+y)))
        if len(pts) >= 3: polygons.append(pts)
    if not polygons: raise HTTPException(422, "No 3D faces found in geometry output")
    xs=[x for p in polygons for x,_ in p]; ys=[y for p in polygons for _,y in p]
    span=max(max(xs)-min(xs), max(ys)-min(ys), 1); scale=430/span
    colors=("#2563eb","#0ea5e9","#14b8a6","#8b5cf6","#64748b")
    shapes=[]
    for i,p in enumerate(polygons):
        points=" ".join(f"{35+(x-min(xs))*scale:.1f},{465-(35+(y-min(ys))*scale):.1f}" for x,y in p)
        shapes.append(f'<polygon points="{points}" fill="{colors[i%len(colors)]}" fill-opacity=".35" stroke="#93c5fd" stroke-width="1.5"/>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" role="img" aria-label="EnergyPlus building geometry"><rect width="500" height="500" rx="16" fill="#0b1220"/><g>{''.join(shapes)}</g><text x="20" y="482" fill="#94a3b8" font-family="sans-serif" font-size="13">{html.escape(dxf.stem)} · EnergyPlus DXF projection</text></svg>'''
    return Response(svg, media_type="image/svg+xml", headers={"Cache-Control":"no-store"})


@app.get("/api/usage")
async def usage(conversation_id: str | None = None):
    return store.usage_summary(conversation_id)


def main():
    import uvicorn
    port = int(os.getenv("PORT", "5000"))
    if "PORT" not in os.environ:
        for candidate in range(5000, 5010):
            with socket.socket() as sock:
                try: sock.bind(("0.0.0.0", candidate)); port = candidate; break
                except OSError: continue
    print(f"EnergyPlus Multi-Agent System: http://localhost:{port}")
    uvicorn.run("energy_agent.app:app", host="0.0.0.0", port=port, log_level="info")
