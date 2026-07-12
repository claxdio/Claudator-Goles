# Chile Trial: Own xG Model + Historical Backfill + Retrain — Design

**Status:** Approved

## Motivation

The user wants a low-scale live trial of the goal-prediction model **this month**, using the Chilean leagues (his home region, whose teams he knows well), instead of waiting for the Premier League/Bundesliga season restart in August. The blocker: the model's features are mostly xG-derived, and **no free source publishes shot-level xG for Chilean football** — this was verified exhaustively, not assumed:

- **Sofascore**: full shotmaps for Liga de Primera (verified: 23–35 shots/match, 100% with coordinates, situation, bodyPart, goalMouthLocation; historical seasons listed back to 2019) — but `xg: null` on every Chilean shot.
- **FotMob**: no xG for Chile at all. Verified by inspecting a real finished Liga de Primera match's full embedded payload (`__NEXT_DATA__`): all 32 "expected_goals" mentions were UI translation strings; `content.shotmap.shots` empty; no expected-goals entry in match stats. (Third-party AI research claiming "guaranteed live xG for Chilean Primera División on FotMob" was simply wrong.) FotMob's old `/api` endpoints are also gone (404 + HTML), though `apigw.fotmob.com/searchapi` and page-embedded Next.js data remain scrapeable.
- **FootyStats**: hard Cloudflare 403 even to the TLS-impersonation client that gets through Sofascore's protection. Automating the user's manual "remove the premium blur via dev tools" trick would require full browser automation fighting active anti-bot — fragile, high-maintenance, and even then only exposes *team-level* xG totals, not the per-shot minute-stamped values the model's features require.
- **The Stats Don't Lie**: accessible, but only season-level team xG aggregates; Chile isn't even listed in its xG index.
- **FBref**: no xG for Chile (Opta doesn't collect advanced data for this league — confirmed by the user's own research).
- **football-data.co.uk**: does not cover Chile at all (`new/CHL.csv` is China). **No free historical market odds exist for Chile.**

## The insight that makes this feasible

We already own everything needed to **compute xG ourselves**:

- `data/goles.db` holds **106,538 historical shots** (Understat, EPL+Bundesliga 2018–2024) with `location_x`/`location_y` (0–1, toward attacking goal), `situation` (`OpenPlay`/`FromCorner`/`SetPiece`/`DirectFreekick`/`Penalty`), `shot_type` (**body part**: `RightFoot` 54k / `LeftFoot` 33k / `Head` 19k / `OtherBodyPart` 0.5k), and both the `is_goal` outcome and Understat's own `xg` as a reference label.
- Sofascore's Chilean shotmaps carry the matching inputs: `playerCoordinates` (x ≈ % of pitch length from the opponent's goal line, y ≈ % of width — different convention, must be mapped), `situation` (`regular`/`assisted`/`corner`/`set-piece`/`fast-break`/`free-kick`/`penalty` — different vocabulary, must be mapped), `bodyPart` (`right-foot`/`left-foot`/`head` — maps directly to Understat's `shot_type`), and `time` (minute).

Training a shot→P(goal) classifier on our labeled Understat shots and applying it to Sofascore's Chilean shots gives us minute-stamped shot-level xG for Chile — for **both** the historical backfill (training data) and the live feed (the home-PC scraper already works against Sofascore). One consistent source and one consistent xG scale across train and serve.

**Built-in validation set for the coordinate/vocabulary mapping:** Sofascore *does* publish real xG for top-tier competitions (verified on a FIFA World Cup match). Applying our own xG model to a top-tier Sofascore shotmap and comparing per-shot against Sofascore's published xG directly validates the whole translation layer — if the mapping is wrong, the comparison will show it immediately.

## Verified Sofascore identifiers and coverage (fetched, not guessed)

| Competition | uniqueTournament id | Shot data? |
|---|---|---|
| Liga de Primera (Chile 1st tier) | **11653** | Yes — full shotmaps, current + historical (season ids: 2026=88493, 2025=71131, 2024=57883, 2023=48017, 2022=40515, 2021=36048, 2020=26951, 2019=22328) |
| Liga de Ascenso (Chile 2nd tier) | **1240** | Yes — verified on 3 finished matches, 100% coordinates |
| Copa Chile | 1221 | **No — 404 on shotmap even for Universidad de Chile matches. Excluded from the model** (goal/card alerts only, if ever wanted) |

**Critical naming hazard, verified the hard way:** Paraguay's second tier is also named "Primera División B" on Sofascore (utid 22759 — initially mistaken for Chile's). All Chilean filtering **must use uniqueTournament ids, never tournament-name matching.**

Backfill endpoints (all confirmed working via `tls_requests` from the home PC): `/unique-tournament/{utid}/seasons`, `/unique-tournament/{utid}/season/{sid}/events/last/{page}` (30 events/page), `/event/{id}/shotmap`, `/event/{id}/incidents`.

## Architecture

Three offline stages (this spec), with the live stage (D) deliberately deferred:

**A. Own xG model** (`src/goles/xg_model.py` + `src/goles/train_xg.py`): LightGBM binary classifier P(goal|shot) trained on the 106,538 Understat shots. Features: distance to goal center and shot angle (computed from `location_x`/`location_y` on a 105×68 m pitch), situation (categorical), body part (categorical). The shot *outcome* (`is_goal`) is the label; Understat's `xg` column is used only as an evaluation reference (correlation/MAE of our predictions vs Understat's), never as an input. Artifacts saved under `data/model/` next to the existing goal-model artifacts. Penalties get special handling (fixed ≈0.76 xG, the empirical penalty conversion rate — a model trained mostly on open play should not extrapolate to them).

**B. Sofascore→Understat translation layer** (`src/goles/sofascore/translate.py`): converts a raw Sofascore shot dict into the Understat-convention shot dict the rest of the codebase already understands — coordinates mapped to 0–1-toward-goal (`location_x = 1 − x/100`, `location_y = y/100`, verified empirically via the top-tier xG comparison above), situation vocabulary mapped (`corner`→`FromCorner`, `set-piece`→`SetPiece`, `free-kick`→`DirectFreekick`, `penalty`→`Penalty`, `regular`/`assisted`/`fast-break`→`OpenPlay`; unknown values fail loud in logs, never guess silently), bodyPart mapped to `shot_type`. **Key interface decision: Chilean shots are stored already translated into Understat conventions**, so `compute_ml_features`, `features.py`'s box-threshold logic (BOX_X_THRESHOLD=0.84), `priors.py`, and `dataset.py` all work on Chilean data completely unchanged.

**C. Chilean historical backfill + retrain** (`src/goles/sofascore/backfill.py` + `src/goles/train_gbt_chile.py`): paginate every season 2022–2026 of utids 11653 and 1240 from the home PC (rate-limited, ~1 req/s; roughly 2,500 matches × 3 requests ≈ 2 hours; resumable — already-stored matches are skipped), writing into a **new, separate database `data/goles_chile.db`** that reuses the existing `goles.db` schema verbatim (`teams`/`matches`/`shots`/`cards`; `matches.understat_id` stores the Sofascore event id — documented, pragmatic column reuse that keeps every existing query working). Shots are stored with our computed xG and translated fields; red cards come from `/incidents` — and since backfilled seasons contain many real red cards, **the backfill empirically resolves the still-unverified `incidentClass` red-card vocabulary** (log all observed `card` incidentClass values; the poller's `RED_CARD_INCIDENT_CLASSES = {"red", "yellowRed"}` assumption gets confirmed or corrected with real data). Then retrain the goal model on Chilean data with the existing pipeline (`build_dataset`/`train_gbt` pointed at the Chile DB): train 2022–2024, validation 2025, test 2026-to-date (~120 finished matches). Market features are all 0.0 (no historical odds exist for Chile — the "missing market" convention the model already handles); `own_linebreak_shots`/`own_transition_shots` are 0.0 in both training and test (Sofascore has no lastAction — consistent, no train/serve skew).

**Decision gate:** stage C's backtest (BSS + calibration on held-out Chilean matches) answers "does the model work for Chile?" **before** any live infrastructure is built. Phase D (live inference + Telegram, reusing the already-running home-PC scraper extended to utids 11653/1240 and the already-validated Telegram bot token/chat id) is a separate follow-up spec, conditional on C's result. Expectation set honestly: without market odds (the EPL model's single strongest feature) the Chilean model should land *below* the EPL's BSS 0.0335 — the trial's question is whether it stays meaningfully above zero.

**Parallel, cheap, optional:** add Chilean Primera División to the Betfair VPS poller's tracked competitions so live odds start accumulating now for a future market-aware retrain (weeks of odds+outcomes needed before useful — not part of this plan's gate).

## Out of scope

- Live inference loop and Telegram delivery (Phase D — own spec after C's gate).
- Copa Chile in the model (no shot data exists; alerts-only at most, later).
- Scraping FootyStats/FotMob/TSDL (investigated and rejected above).
- Any paid service (unchanged project constraint).
