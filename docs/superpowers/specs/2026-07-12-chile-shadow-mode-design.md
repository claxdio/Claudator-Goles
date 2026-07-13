# Chile Shadow Mode: Live Observation Without Betting Signal — Design

**Status:** Approved

## Motivation

The Chile trial's decision gate (see `2026-07-12-chile-own-xg-and-backfill-design.md` → Resultado) came out **negative**: BSS 0.0035 on held-out 2026 Chilean matches, below the ≳0.01 bar. Per that plan's rule, the original Phase D (live inference + Telegram *as betting signal*) is not built.

This is the honest alternative the user chose instead: **shadow mode**. Run the full live pipeline end-to-end — live Chilean matches, real-time feature computation, the already-trained Chilean model, Telegram delivery — but with every alert explicitly labeled **"modo prueba — sin edge confirmado, solo observación"**. Never presented as a betting signal.

What this buys:

1. **The user sees the whole pipeline working live this month**, on teams he knows, instead of waiting for August.
2. **It accumulates a live-serve validation set.** Every prediction is persisted with its Sofascore event id; after matches later land in `goles_chile.db` (re-running the resumable backfill), predictions join against real outcomes — measuring the *live serving path's* BSS, which the offline backtest can't do (it can't catch train/serve skew bugs).
3. **In parallel, the Betfair poller (Task 6 of the previous plan) keeps accumulating Chilean odds.** In a few weeks that enables a market-aware retrain — a legitimate second attempt at the gate before August.

## What already exists (all verified, nothing assumed)

- **Home-PC Sofascore live poller** (`src/goles/sofascore/poller.py`): polls `list_live_events` every 60 s, persists shots/red cards to `data/live_match_state.db`, scp-syncs to the VPS. Runs on the home PC (Sofascore blocks datacenter IPs — verified). Currently tracks EPL/Bundesliga by tournament *name*.
- **Live events carry `tournament.uniqueTournament.id`** (verified live 2026-07-12) — so Chilean filtering can use utids 11653/1240 (the hard constraint from the previous spec: never name-match, Paraguay collision).
- **Current match minute is computable** from `time.initial` (seconds at period start: 0 first half, 2700 second half) + `time.currentPeriodStartTimestamp` (verified live). Edge cases: some events have an empty `time` dict, and during halftime the timestamp is stale → in both cases skip inference that cycle.
- **Trained artifacts on this PC:** `data/model/xg_booster.txt` (own xG model), `data/model_chile/booster.txt` + `platt.json` (Chilean goal model, trained 2022–2024, calibrated on 2025), `data/goles_chile.db` (1,691 backfilled matches — the source for pre-match priors).
- **Translation layer** (`sofascore/translate.py`) handles Chilean shots including own goals; **Sofascore publishes `xg: null` for Chile**, and the live store's `xg` column is NOT NULL — so today a Chilean shot would be dropped by the per-shot exception handler. Shadow mode must compute our own xG at ingest.
- **Telegram**: no code exists in the repo yet. Bot token/chat id will come from env vars (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).

## Architecture

Everything runs inside the **existing home-PC poller process** (Chile can't be polled from the VPS). Three new pieces, two modified:

**A. `src/goles/telegram.py` (new).** `send_message(client, token, chat_id, text) -> bool` via the Bot API. Failures return False with a printed warning — Telegram being down must never crash the poller. If env vars are missing at startup, shadow mode still runs and persists predictions; alerts are printed to stdout only (warned once).

**B. `src/goles/sofascore/shadow.py` (new) — the shadow inference engine.**
- Discovery: filter live events by `uniqueTournament.id ∈ TRACKED_UTIDS` (imported from `backfill.py` — same dict).
- Minute estimation from the verified `time` fields; `None` (skip) when not in-progress, at halftime, or `time` is empty. Inference only runs when the estimated minute is within **[20, 80]** — the model's training cutoff range; outside it the model is out-of-distribution.
- Live feature assembly that **exactly mirrors `dataset.build_dataset`'s row assembly**: `compute_ml_features` + rest days + market features hard-zero + `trailing_prior_xg` + `poisson_prob` with **blend = 0.1** (the value training used — train/serve consistency is enforced by a test that runs the same synthetic match through both paths and asserts identical feature dicts).
- Pre-match priors come from `data/goles_chile.db` by **exact team-name lookup** (both the live feed and the backfill store Sofascore's own names — same source). Unknown team (promoted, renamed) → prior 0.0 + rest days 7.0, matching the matchday-1 behavior in training. Season = current calendar year as string.
- Prediction: `FEATURE_NAMES` ordering → booster → Platt scaling, using `data/model_chile` artifacts.
- Alerting: calibrated P(goal in next 15) ≥ **0.30** (near the top of the range the model actually produced in the backtest — rare, interesting moments) with a **15-minute per-(event, team) cooldown**, state persisted in the DB so restarts don't re-alert. Message format is fixed and always carries the banner:

```
🧪 MODO PRUEBA — sin edge confirmado, solo observación
⚽ Colo-Colo 1-0 Cobresal (CHI-Liga de Primera)
Min 63' — P(gol de Cobresal en próx. 15 min): 34%
```

**C. `src/goles/sofascore/store.py` (modified).** Two new tables in `live_match_state.db`: `shadow_predictions` (every prediction, every cycle: event id, timestamp, team, minute, probability, features JSON — the future live-serve validation set) and `shadow_alerts` (what was sent when — doubles as the cooldown state).

**D. `src/goles/sofascore/poller.py` (modified).**
- Discovery: EPL/Bundesliga by name (unchanged) **plus** Chilean events by utid.
- Chilean shots get `xg` computed via `translate_shot` + our xG booster at persist time (Sofascore's is null; own goals get 0.0). The store keeps Sofascore's *raw* field conventions otherwise (raw coordinates, raw attribution) — the translated view lives only in shadow's in-memory path, same as the backfill's split.
- After each poll cycle, `shadow_cycle(...)` runs per live Chilean event: estimate minute → fetch shotmap → translate (own xG, own-goal flip) → read red cards from the live store → assemble features for both teams → predict → persist both predictions → alert if threshold+cooldown allow. Shadow fetches its own shotmap (a couple of extra requests per minute at most) to stay decoupled from `poll_once`'s contract.
- Startup degradation: if `data/model_chile` or `data/goles_chile.db` is missing, shadow mode is disabled with a loud warning and the EPL poller keeps working untouched.

## What this is NOT

- **Not a betting signal.** The model's Chilean BSS is ~0. Every message says so. No stake sizing, no odds comparison, no "value" framing anywhere in the copy.
- **Not the original Phase D.** That assumed a positive gate and remains unbuilt.
- **Not a new model.** Same `data/model_chile` artifacts the backtest produced; shadow mode changes serving, not training.

## Operational notes

- The poller is a home-PC process (not Dokploy): deploy = `git pull` + restart the process.
- Priors staleness: `goles_chile.db` is a snapshot; re-run `python -m goles.sofascore.backfill` weekly (resumable — skips existing matches) to keep priors fresh *and* to land finished matches for joining against `shadow_predictions`.
- Evaluation (in ~2–4 weeks, separate step): join `shadow_predictions` to re-backfilled `goles_chile.db` outcomes via the Sofascore event id → live-serve BSS + calibration. This also cross-checks the offline backtest.

## Out of scope

- Any betting-signal framing, stake logic, or odds-aware alerting.
- The market-aware Chilean retrain (needs weeks of Betfair odds — separate future spec).
- EPL/Bundesliga shadow inference (no trained-for-live model decision yet; August).
- Automating the weekly backfill re-run (manual for now).
