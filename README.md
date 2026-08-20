# Energy Agent V2

Energy Agent V2 is a persistent multi-agent EnergyPlus engineering application.
It coordinates dedicated planning, model inspection, simulation, results,
optimization, sensitivity, calibration, error analysis, and QA agents through an
EnergyPlus MCP service.

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
