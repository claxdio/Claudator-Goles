# Live Betfair Odds Pipeline — Design

**Status:** Approved

## Motivation

The market-odds/rest-days/red-card plan confirmed `own_market_wp` — the no-vig, market-implied win probability derived from bookmaker odds — is by far the single most valuable feature in the model (4x the next feature). The closing-line experiment (`docs/superpowers/specs/2026-07-11-closing-line-odds-experiment-design.md`) confirmed pre-match-average-style odds timing is already good enough; there's no need to chase the freshest possible pre-kickoff price. What's still missing for a *live* system is a source of **live** odds at all — today the model only has historical, pre-match odds from football-data.co.uk, ingested in batch after the fact. This plan builds the first piece of Phase 2's live pipeline: a service that continuously fetches real odds from Betfair Exchange and stores timestamped snapshots, so a future inference step can feed the model an `own_market_wp` computed from the odds actually available at any given live moment, not a stale pre-match number.

**Scope of this plan:** fetch and persist live odds snapshots only. Feeding those snapshots into live inference, and sending Telegram alerts, are separate follow-up sub-projects (per the phase-2 breakdown already agreed) — deliberately not built here, so this plan stays independently testable and deployable.

## Infrastructure context (verified against the real VPS)

- VPS: `85.239.245.73`, Ubuntu 22.04, managed entirely through **Dokploy** (self-hosted PaaS on Docker Swarm + Traefik). An existing project ("anjuma") runs there as Dokploy-managed containers (backend, frontend, Postgres) — nothing on the VPS runs outside Dokploy (no bare systemd services, no crontab).
- This project gets its **own Dokploy project** (`Claudator-Goles`, already created by the user, empty), with its own container(s) and network — fully isolated from `anjuma` at the Dokploy level, same mechanism already used for the existing project.
- A dedicated SSH keypair (`~/.ssh/id_ed25519_goles_vps` on the developer's machine) was generated and installed in `root`'s `authorized_keys` on the VPS specifically for this work, separate from the developer's personal key.
- The GitHub repo `https://github.com/claxdio/Claudator-Goles` (branch `master`) is now the deploy source Dokploy will build from.

## Betfair authentication (verified against official docs, not assumed)

- **Non-interactive ("bot") login is certificate-based**, independent of app-key type: `POST https://identitysso-cert.betfair.com/api/certlogin` with header `X-Application: <app key>`, form body `username=<betfair username>&password=<betfair password>` (URL-encoded), presenting a client TLS certificate. Response: `{"sessionToken": "...", "loginStatus": "SUCCESS"}`.
- Certificate: a self-signed 2048-bit RSA key/cert pair (`client-2048.key` kept secret on the VPS only, `client-2048.crt` uploaded by the user to their Betfair account's security settings page — this upload step requires the user's own Betfair login and cannot be done on their behalf).
- **Delayed application key**: auto-active on creation (no manual approval step), returns market prices with a **3-5 minute lag**, and cannot be used to place real-money bets. This is exactly what we want — the pipeline never places bets, it only reads prices.
- **Session lifetime is not documented** by Betfair. The client must not assume a fixed duration: any API call that fails due to an invalid/expired session triggers an automatic re-login, rather than the poller trying to pre-emptively guess when to refresh.
- Once logged in, ordinary Exchange API-NG calls (`listCompetitions`, `listMarketCatalogue`, `listMarketBook`) use both `X-Application` (app key) and `X-Authentication` (session token) headers.

**External prerequisites the user must complete themselves (blocking real end-to-end testing, not blocking development):**
1. Register a delayed Application Key via the Betfair Developer Program portal.
2. Generate the self-signed certificate (this plan's Task 1 does this via OpenSSL, run on the VPS) and upload the resulting `.crt` to their Betfair account's security settings.

Everything else in this plan is buildable and testable (via mocked HTTP) before those two steps complete; the poller simply won't be able to log in for real until they do.

## Architecture

New subpackage `src/goles/betfair/`, kept separate from the existing training pipeline (`src/goles/dataset.py`, `train_gbt.py`, etc.) since this is a long-running production service, not a batch script:

- `auth.py` — `BetfairSession`: performs cert-login, exposes `session_token`, and a `request(method, url, ...)` wrapper that retries once via re-login on an invalid-session error.
- `client.py` — thin wrappers over the 3 Exchange API-NG calls needed: `list_competitions`, `list_market_catalogue`, `list_market_book`, scoped to `eventTypeId=1` (Soccer). Competitions are discovered dynamically by name match against "English Premier League" / "Bundesliga" at startup and cached, rather than hardcoding competition IDs from memory (unverified IDs would be a silent correctness risk).
- `odds_store.py` — SQLite schema + writer for `live_odds.db` (a **new, separate database file from `data/goles.db`** — this one holds live production state on the VPS, not historical training data): one `odds_snapshots` table (`fetched_at`, `betfair_event_id`, `home_team`, `away_team`, `market_type` [`MATCH_ODDS`/`OVER_UNDER_25`], `home_wp`/`draw_wp`/`away_wp`/`over_wp` computed via the existing no-vig math, `raw_json` for auditability).
- `poller.py` — the persistent loop: on startup, logs in, discovers today's/upcoming fixtures in the 2 tracked leagues, then repeats forever: for each tracked fixture, fetch `MATCH_ODDS` + `OVER_UNDER_25` market books, compute no-vig probabilities, write a snapshot row, sleep `POLL_INTERVAL_SECONDS` (default 60s — compatible with the delayed key's 3-5 minute data lag; polling faster wouldn't get fresher data). Refreshes its fixture list periodically (new matches get added, finished matches drop off) rather than only once at startup.
- `Dockerfile` — a small Python image running `python -m goles.betfair.poller` as its entrypoint.

**Team-name matching:** Betfair identifies fixtures by event/runner name strings, which won't exactly match our Understat-sourced `teams.name` values (same class of problem `football_data.py`'s `TEAM_NAME_ALIASES` already solves for football-data.co.uk). A small alias table is added specifically for Betfair's naming, built the same way — verified empirically, not guessed — and any fixture that can't be matched is logged loudly and skipped, never silently fuzzy-matched.

**No-vig math:** reuses the existing `compute_no_vig_probabilities`/`compute_no_vig_two_way` from `goles.loaders.football_data` unchanged (best-available back price stands in for the bookmaker's decimal odds — a deliberate simplification for this MVP; refining to a liquidity-weighted mid-price is listed under future work, not built here).

## Deployment

- Dokploy "Application" (not a Dokploy Schedule) — a persistent, `restart: always` container, since it needs to hold a live Betfair session and poll on a short interval; a scheduled one-shot job would force a fresh login (and fresh fixture discovery) on every run, which is both slower and closer to Betfair's login rate limits.
- Config via environment variables (Dokploy secrets): `BETFAIR_APP_KEY`, `BETFAIR_USERNAME`, `BETFAIR_PASSWORD`; the certificate/key files mounted as Dokploy file mounts (never committed to git).
- `live_odds.db` lives on a Dokploy-managed volume so it survives redeploys/restarts.
- Testing: same TDD discipline as the rest of the project — every Betfair HTTP call is mocked in tests; no test requires real network access or real credentials. The only manual, real-network verification step is a one-time run against the actual Betfair delayed endpoint once the user has completed both external prerequisites above.

## Out of scope (deliberately, for this plan)

- No automated bet placement, ever — this service is read-only.
- No live inference (consuming these snapshots to produce a live goal-probability prediction) — separate follow-up plan.
- No Telegram bot — separate follow-up plan.
- No ClubElo wiring — already deprioritized in the market-odds plan's "Próximos pasos" now that market odds cover similar ground.
- No ability to place the certificate/credentials setup on the user's behalf beyond generating the cert files themselves (the Betfair-side account actions are inherently manual).

## Estado de despliegue

Deployed as a new Dokploy Application, `betfair-odds-poller`, inside the existing `Claudator-Goles` project's `production` environment (`http://85.239.245.73:3000`). Configuration: GitHub source (`claxdio/Claudator-Goles`, branch `master`), Dockerfile build (`Dockerfile.betfair`), a named volume (`betfair-odds-poller-data` → `/app/data`) for `live_odds.db` persistence, and two bind mounts wiring the Task 1 certificate/key (`/root/goles-betfair-certs/client-2048.{crt,key}` on the host → `/run/secrets/betfair/client-2048.{crt,key}` in the container). The Dokploy GitHub App needed its repository access extended to this repo first (it only had the `anjuma` project before) — a one-time GitHub setting change.

Real run confirmed the whole build/deploy/mount pipeline end-to-end: the Docker image built successfully from the git repo (~90s, consistent with the manual build test), the container started, and it crashed immediately with a clean `KeyError: 'BETFAIR_APP_KEY'` traceback in the Dokploy Logs tab — exactly the loud, readable failure mode required, since `BETFAIR_APP_KEY`/`BETFAIR_USERNAME`/`BETFAIR_PASSWORD` were deliberately left unset.

**Remaining before real odds flow:** the user has already obtained the delayed Application Key (visible in Betfair's Accounts API Demo Tool) and uploaded the Task 1 certificate to their Betfair account's security settings during this session. The only remaining step is entering the three real environment variable values into the Dokploy Application's Environment tab and redeploying — no further code or infrastructure changes are needed.
