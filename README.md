# Energy Agent V2

Energy Agent V2 is a persistent multi-agent EnergyPlus engineering application.
It coordinates dedicated planning, model inspection, simulation, results,
optimization, sensitivity, calibration, error analysis, and QA agents through an
EnergyPlus MCP service.

## Engineering workflow modes

### 1. Real EnergyPlus-backed workflows

Real calibration keeps a user-supplied measured CSV fixed while a deterministic
Python optimizer changes an occupancy multiplier in isolated working IDFs. Each
candidate is run through EnergyPlus, hourly `Electricity:Facility` output is
aligned with the measured timestamps, and Python calculates RMSE, MAE, MAPE,
NMBE, and CV(RMSE). Measured/reference data is **not** generated or altered by
EnergyPlus in production.

Real optimization currently searches a bounded lighting multiplier to minimize
simulated annual facility electricity. Real sensitivity currently performs
one-at-a-time normalized sensitivity for either a lighting or occupancy
multiplier. Both run as persistent background jobs with bounded evaluations,
isolated output folders, traces, provenance hashes, failure records, and JSON
results.

```text
User -> Orchestrator -> Specialist -> deterministic study job
     -> safe working IDF -> EnergyPlus-MCP -> actual EnergyPlus
     -> hourly meter CSV -> numerical analytics -> result/explanation
```

### 2. Surrogate/demo workflows

Tools and endpoints named `run_surrogate_*_demo` are fast educational examples.
They use algebraic surrogate behavior and **do not run EnergyPlus**. The planner
selects them only when a user explicitly asks for a demo or surrogate.

### 3. Unit and integration tests

Run fast application tests with `python -m pytest -q tests`. The heavyweight
integration test is opt-in because it performs repeated real simulations:

```bash
RUN_ENERGYPLUS_INTEGRATION=1 python -m pytest -q tests/integration
```

The integration fixture creates an EnergyPlus-generated reference profile only
to verify that calibration can recover a hidden multiplier. That synthetic test
profile is never represented as real measured building data.

Current real-workflow limitations: occupancy calibration is single-parameter;
optimization changes only lighting; sensitivity screens one parameter at a
time; the supported time-series meter is hourly `Electricity:Facility`; runs are
sequential; Bayesian optimization, PSO, and multi-parameter calibration are not
implemented. Existing `calculate_energy_performance` remains the deterministic
EnergyPlus-MCP pathway for annual energy, gross floor area, and EPI/EUI.

## Local development

1. Copy `.env.example` to `.env` and set a fresh Gemini API key.
2. Build the EnergyPlus development image with `build-energyplus.ps1`.
3. Start the stack with `docker compose up -d --build`.
4. Open `http://localhost:5000`.

## Production

Production images are published by GitHub Actions to GitHub Container Registry.
See [DEPLOYMENT.md](DEPLOYMENT.md) for the server, secrets, HTTPS, update, and
backup procedure.

## Temporary GitHub Codespaces test

Create repository-scoped Codespaces secrets named `GOOGLE_API_KEY` and
`APP_PASSWORD`, then choose **Code → Codespaces → Create codespace**. The
containers start automatically and port 5000 appears in the **PORTS** tab. It is
private by default and requires the username `energy-admin` plus the password in
the `APP_PASSWORD` secret.

Secrets and generated runtime results are intentionally excluded from Git.
