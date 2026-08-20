$ErrorActionPreference = "Stop"
docker build -t energyplus-mcp-dev -f EnergyPlus-MCP/.devcontainer/Dockerfile EnergyPlus-MCP/.devcontainer
