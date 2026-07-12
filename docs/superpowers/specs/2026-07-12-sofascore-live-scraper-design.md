# Sofascore Live Match-State Scraper — Design

**Status:** Approved

## Motivation

The trained model predicts "probability of a goal in the next 15 minutes" **during** a live match, using 36 features — the large majority (minute, score, xG accumulated so far, shot counts, red cards, etc.) require live in-play match state. Today the project has zero live in-play data source; the only live signal available is market odds (from the just-completed Betfair pipeline). Before live inference can produce a real prediction, this gap has to close. This plan builds that missing piece: a service that continuously mirrors live shot and red-card events for the two tracked leagues (ENG-Premier League, GER-Bundesliga) into a local database, in a shape the existing (already-built, already-tested) `compute_ml_features`/`compute_state_at_minute` feature code can consume.

**Scope of this plan:** fetch and persist live shot/card events only. Translating raw Sofascore fields into the exact ML feature vector, running the model, and sending Telegram alerts are separate follow-up sub-projects — deliberately kept out of this plan so it stays independently testable and deployable, mirroring how the Betfair odds pipeline was split from live inference.

## Verified findings (not assumed)

- **Sofascore has no official third-party API** (their own FAQ says so) and its real API (`api.sofascore.com`) sits behind bot-fingerprint protection: a plain `requests.get` with just a `User-Agent` header gets HTTP 403 from both a home connection and the VPS.
- **`soccerdata` (already a project dependency) already solves this**: its `Sofascore` reader class successfully fetches data using `tls_requests.Client()`, a TLS-impersonation HTTP client that mimics a real browser's handshake. Confirmed directly: `soccerdata.Sofascore(...).read_schedule()` returns real data (380 rows for ENG-Premier League 2324) where a bare `requests` call is blocked. `soccerdata.Sofascore` itself only exposes `read_league_table`/`read_leagues`/`read_schedule`/`read_seasons` — no live/shot-level methods — so this plan uses the same underlying `tls_requests.Client()` directly (`import tls_requests` — already installed transitively via `soccerdata`, add `wrapper-tls-requests` to `pyproject.toml` explicitly since we import it directly now) against the real endpoints `soccerdata` doesn't wrap.
- **Endpoints confirmed working via `tls_requests.Client()`** (real responses fetched during design, not guessed):
  - `GET https://api.sofascore.com/api/v1/sport/football/events/live` → all live matches worldwide (44 at test time), each with `id`, `tournament.name`, `homeTeam.name`, `awayTeam.name`, `status.description`.
  - `GET https://api.sofascore.com/api/v1/event/{id}/shotmap` → `shotmap: list`, each shot has a **stable unique `id`** (e.g. `7684954`), `time` (minute, int), `xg` (float), `shotType` (`"goal"`/`"block"`/`"miss"`/etc.), `situation` (e.g. `"corner"`, `"assisted"`, `"regular"` — confirmed **different vocabulary from Understat's** `"SetPiece"`/`"FromCorner"`/`"DirectFreekick"`), `isHome` (bool), `playerCoordinates` (`x`/`y`/`z` — confirmed **different coordinate convention from Understat's** normalized 0–1 scale), `bodyPart`.
  - `GET https://api.sofascore.com/api/v1/event/{id}/incidents` → `incidents: list`, each with `time`, `incidentType` (e.g. `"goal"`, `"period"`; card incidents are expected to be `"card"` per community documentation of this API, but **no real red card was observed during this verification** — the exact `incidentClass` value for a red card must be confirmed against a real occurrence during implementation, not assumed), `isHome`. Incidents have no stable per-item id (unlike shots).
- **Important, previously-flagged gap confirmed real**: `src/goles/features.py` already contains a comment (`own_linebreak_shots`/`own_transition_shots`, derived from Understat's `lastAction` field) stating these have **no live equivalent in the Sofascore/FotMob feeds**. Verified: Sofascore's shotmap has no "action leading to the shot" concept at all. These two features cannot be computed live, ever, from this source — a future live-inference plan must default them (e.g. to `0.0`, the same "missing signal" convention already used for missing market odds) or retrain without them. Not this plan's problem to solve, but recorded here so it isn't lost.
- Because of the situation-vocabulary and coordinate-system mismatches above, translating a raw Sofascore shot into `compute_ml_features`'s exact input shape (`situation` values it checks, the box-detection `location_x` threshold) requires a **separate, empirically-verified mapping** — explicitly out of scope for this plan, which stores raw verified Sofascore fields rather than guessing a translation now.

## Architecture

New subpackage `src/goles/sofascore/`. **Deployment target: the developer's home Windows PC, not the VPS** — see "Actualización: bloqueo por IP de datacenter" below for why.

- `client.py` — thin wrappers over the 3 confirmed endpoints (`list_live_events`, `get_shotmap`, `get_incidents`), each taking a `tls_requests.Client` instance (duck-typed, so tests pass a stub — same pattern as `goles.betfair.client`).
- `team_aliases.py` — `SOFASCORE_TEAM_NAME_ALIASES: dict[str, str]` (starts empty — Sofascore's exact team-name strings for our tracked teams haven't been observed yet, same honest-starter-table precedent as `goles.betfair.team_aliases`) + `normalize_sofascore_team_name`.
- `store.py` — SQLite schema + writer for a **new, separate database** `data/live_match_state.db` (distinct from both `data/goles.db` and the VPS-side `live_odds.db`), gitignored like the rest of `data/`: two tables, `shots` and `cards`, storing the raw verified fields above (not yet translated into the ML feature shape — that translation is the next sub-project's job).
- `poller.py` — the persistent loop: each cycle, calls `list_live_events`, filters to events whose tournament name exactly matches one of `TRACKED_TOURNAMENTS = ["Premier League", "Bundesliga"]` (exact match, not substring — avoids false positives like "Scottish Premier League"), and for each matched live match fetches shotmap + incidents and upserts new rows. After each cycle, `scp`s the updated `live_match_state.db` to the VPS (`root@85.239.245.73:/root/goles-live-match-state/live_match_state.db`) using the same dedicated SSH key already set up on this machine (`~/.ssh/id_ed25519_goles_vps`) — no new server/API to maintain, and it lands wherever the future live-inference plan expects to read it from (that plan already runs on the VPS, alongside `live_odds.db`).
- Runs via **Windows Task Scheduler**: a task configured to start at logon/system startup, run the poller script in the project's existing venv, and restart automatically on failure — the standard no-extra-software way to keep a script alive on Windows.

## Data model

`shots` table: `shot_id INTEGER PK AUTOINCREMENT`, `sofascore_shot_id INTEGER NOT NULL UNIQUE` (Sofascore's own stable id — the natural idempotency key), `sofascore_event_id INTEGER NOT NULL`, `fetched_at TEXT NOT NULL`, `team TEXT NOT NULL` (`"home"`/`"away"`), `minute INTEGER NOT NULL`, `xg REAL NOT NULL`, `is_goal INTEGER NOT NULL` (`shotType == "goal"`), `shot_type TEXT NOT NULL` (raw), `situation TEXT` (raw, nullable), `location_x REAL` (raw, nullable), `location_y REAL` (raw, nullable), `body_part TEXT` (raw, nullable).

`cards` table: `card_id INTEGER PK AUTOINCREMENT`, `sofascore_event_id INTEGER NOT NULL`, `fetched_at TEXT NOT NULL`, `team TEXT NOT NULL`, `minute INTEGER NOT NULL`, `card_type TEXT NOT NULL` (raw `incidentClass` value), with a `UNIQUE(sofascore_event_id, team, minute)` constraint as the idempotency key (incidents have no stable per-item id, but two red cards for the same team in the same real match minute is not a realistic collision). **Only red-card incidents are persisted** (yellow cards excluded at ingestion) — same "red cards only, weaker/noisier signal for yellow" rationale already documented in the market-rest-features plan's Global Constraints. The exact `incidentClass` string(s) meaning "red card" must be confirmed against a real observed red-card incident during implementation (fail loud / log-and-skip if the shape doesn't match what's expected, never guess a filter that might silently miss real red cards).

## Verification / Testing

Same TDD discipline as the rest of the project: every Sofascore HTTP call mocked in tests (via a stub client with a `.get()` method, no real network access required). The real-network verification steps are manual (same precedent as the Betfair poller): a real run against the live endpoints confirming shots/cards accumulate in `data/live_match_state.db` for whatever matches are live in the tracked leagues at the time, confirming the `scp` sync actually lands the file on the VPS, and confirming the Windows Task Scheduler task restarts the poller after a simulated failure.

## Out of scope (deliberately, for this plan)

- Translating raw Sofascore fields into `compute_ml_features`'s exact input shape (situation-vocabulary mapping, box-coordinate threshold calibration) — next sub-project.
- Running the model / producing a live prediction — next sub-project.
- Telegram delivery — next sub-project.

## Actualización: bloqueo por IP de datacenter (no geográfico) — FotMob descartado

Once `client.py` was about to be built against the real endpoints, a from-the-VPS verification run (same discipline as every other endpoint check in this project) hit `403` on `https://api.sofascore.com/api/v1/sport/football/events/live` — both directly from the Missouri VPS and through the UK Oracle proxy already built for Betfair. This ruled out a geo-block (the same fix that worked for Betfair does nothing here) and pointed at an IP-reputation/ASN-based block against known datacenter/hosting-provider ranges (Contabo and Oracle Cloud both blocked; the same request from a home residential connection succeeds). Two false leads were investigated and ruled out concretely before reaching that conclusion: a UDP receive-buffer warning tied to the TLS-impersonation library's QUIC handshake (fixed via `sysctl -w net.core.rmem_max=...` on the VPS host, but the `403` persisted unchanged, proving it wasn't the cause), and a transient rate-limit from repeated testing (ruled out by retrying cleanly after a 15-minute gap — still `403`).

**FotMob was evaluated as an alternative and rejected.** Its unofficial API requires a signed `x-mas` request header that community wrapper libraries obtain by querying a **third-party, unidentified server at a bare IP address** (`http://46.101.91.154:6006/`) — an unverified, unmaintained-by-anyone-known dependency for producing an authentication token. This is a worse trust posture than Sofascore's IP-based block, not a better one, so FotMob is not used.

**Resolution:** run this scraper from the developer's home Windows PC (confirmed working — residential IPs aren't subject to this block), syncing its output to the VPS via `scp` after each poll cycle, rather than as a Dokploy service. This is a real operational tradeoff (the home PC must stay powered on and connected for the scraper to keep running) accepted explicitly by the user in preference to a paid residential-proxy service, which would have been the first time this project paid for infrastructure.

## Estado de despliegue

Deployed as a Windows Scheduled Task (`GolesSofascorePoller`, `schtasks`/`Register-ScheduledTask`, trigger `AtLogOn`, `RestartCount 999` / `RestartInterval 1 minute`) on the developer's home PC. Registration required an elevated (Administrator) PowerShell session — the normal user session got `Access Denied` from `Register-ScheduledTask`.

**Confirmed working:** the task starts the poller, which connects to Sofascore successfully from the home connection (TLS-impersonation library loads, no `403` — the datacenter-IP block described above does not apply here) and runs its normal cycle, printing `"0 partidos en vivo encontrados en las ligas trackeadas."` without error.

**Not yet verified (blocked on the football calendar, not a defect):** it's mid-July 2026 — both ENG-Premier League and GER-Bundesliga are in their off-season, so there are genuinely zero live official matches for the poller to find right now. This means the following are still unverified and must be checked once the season resumes (August):
- Real shots/cards actually accumulating in `data/live_match_state.db`.
- The `scp` sync landing the file on the VPS (`/root/goles-live-match-state/live_match_state.db`).
- The `RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}` assumption in `src/goles/sofascore/poller.py`, against a real observed red card.
- The Task Scheduler restart-on-failure behavior under a real crash.

None of this blocks the plan from being considered code-complete — the poller is built, tested, deployed, and confirmed reachable; what remains is real-world data verification that can only happen once matches are actually being played.
