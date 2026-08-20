# Production deployment

The GitHub workflow publishes two private images:

- `ghcr.io/iisc-kpawan/energy-agent-v2-app`
- `ghcr.io/iisc-kpawan/energy-agent-v2-energyplus`

## Server requirements

Use a 64-bit Ubuntu Docker host with at least 4 CPU cores, 8 GB RAM, 30 GB disk,
and inbound TCP 80/443. EnergyPlus workloads benefit from more CPU and memory.

## First deployment

1. Install Docker Engine and the Compose plugin on the server.
2. Copy `compose.production.yml`, `deploy/Caddyfile`, and `.env.production.example`.
3. Rename `.env.production.example` to `.env.production` and replace every placeholder.
4. Authenticate the server to private GHCR images with a read-packages token.
5. Run:

   ```sh
   docker compose --env-file .env.production -f compose.production.yml pull
   docker compose --env-file .env.production -f compose.production.yml up -d
   ```

6. For a domain, point its DNS A/AAAA records at the server, set `APP_DOMAIN` to
   the domain and `PUBLIC_ORIGIN` to its `https://` URL, then recreate Caddy.

## Updating

Push to `main`, wait for the GitHub workflow, then run the pull/up commands again.

## Backups

Back up the `app_data` and `simulations` Docker volumes. They contain chat history,
uploaded models, usage data, and generated simulation results.

Never commit `.env`, `.env.production`, API keys, access tokens, passwords, the
runtime database, or generated simulation outputs.
