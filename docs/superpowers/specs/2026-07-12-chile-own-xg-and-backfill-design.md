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

## Task 1 note: xG-vs-Understat correlation ceiling (0.68, not the planned 0.80)

Real training run on all 106,538 historical shots: correlation with Understat's own xG on held-out validation = **0.6780** (MAE 0.0598; our mean xG 0.1081 vs Understat's 0.1093 vs actual goal rate 0.1033 — well calibrated in aggregate, just less correlated shot-by-shot).

Investigated as a potential bug (systematic-debugging process) before proceeding:
- Vocabulary in `situation`/`shot_type` is clean — no unexpected values, no null coordinates.
- Train-set correlation (0.6956) ≈ validation-set correlation (0.6780) — rules out overfitting.
- Feature geometry checked and sane (angle/distance formulas, y centered at 0.5, x distribution as expected).
- Tried a richer model (raw x/y + polynomial terms + more leaves/capacity as an alternative feature set): correlation got *worse* (0.6537), not better — ruling out "our feature engineering is just missing an obvious transform."

**Task 3 confirms this was the right call.** Real run against a finished top-tier match (Sofascore event 12813015, 33 shots, all comparable): correlation with Sofascore's own published xG = **0.9127** (MAE 0.0275, means 0.0887 ours vs 0.0751 Sofascore's), far above the ≥0.75 gate. Sofascore's own xG model is evidently closer in complexity to our location+situation+bodypart model than Understat's is — which is exactly the model we're replicating for Chile, so this is the correlation that matters.

**Conclusion:** this is a real ceiling, not a bug. A model trained only on shot location + situation + body part cannot fully replicate Understat's own xG, which almost certainly incorporates shot context we deliberately excluded (shot speed, GK/defender positioning, assist type, one-on-one, rebounds — the same `lastAction`-derived signals Sofascore doesn't expose either, so this exclusion is also what keeps Chilean train/serve features consistent). **Decision (with the user): accept 0.68 and proceed** — Task 3's correlation against Sofascore's *own* published xG (threshold ≥0.75) is the metric that actually matters for this trial, since it validates the coordinate/vocabulary translation layer against the same kind of location-based model we're replicating, not a richer proprietary one.

## Task 4 note: two real bugs found during the backfill, both fixed and both matter beyond this trial

The backfill ran three times against real Sofascore data before settling. Two vocabulary/logic gaps surfaced that weren't anticipated in the design, each caught by actually running the real thing rather than by unit tests:

1. **`throw-in-set-piece` situation.** Unmapped, failed loud as designed, extended `SITUATION_MAP` → `SetPiece`. Low-stakes (dropped a shot's xG contribution, not a match outcome).
2. **Own goals silently dropped.** Sofascore represents an own goal with `situation="own-goal"`/`goalType="own"`, attributed to the *scoring-against* player's own team. `translate_shot` rejected this as unknown vocabulary, meaning the shot (and its goal) never got persisted at all — under-counting the final score for any Chilean match with an own goal, corrupting the ground-truth labels this whole trial's backtest depends on. Fixed to mirror the existing Understat convention: flip the scoring team, force `xg=0.0`. Verified after the fix: `SUM(home_goals)+SUM(away_goals)` across the whole Chilean DB exactly equals the count of `is_goal=1` shot rows (4423 = 4423).

A third issue surfaced from the sanity-check numbers, not from a vocabulary error: the red-card rate came in far higher than the plan's rough expectation (1 per ~2.08 matches vs. an assumed 1-per-8-12). Inspecting the highest-count match found two "red cards" attributed to **coaches** (Ariel Holan, Manuel Fernandez — Sofascore marks a manager/staff dismissal with a `manager` key instead of `player`, and a negative `time` as a post-match ejection marker). Fixed in both `backfill.py` and, since it's the identical bug, the **live production poller** (`sofascore/poller.py`) — pushed and redeployed via Dokploy. A 40-match resample after the fix found the remaining red cards have legitimate reason codes (Foul/Violent conduct/Argument) and the manager-card fix only removed 52 of 814 total (762 remain) — the still-elevated rate looks like a genuine characteristic of this dataset (Chilean football, especially Liga de Ascenso, red-carding more often than the rough European-anchored guess), not a further bug.

**Final backfill numbers (after both fixes, clean rerun):** 1,691 matches (Liga de Primera 2022–2026 + Liga de Ascenso 2024–2026; Ascenso 2022–2023 have no Sofascore shotmap coverage at all — 0/280 and 0/252 matches). 43,558 shots (25.76/match, expected 20–30). Mean xG 1.28/team/match (expected 1.0–1.6). 762 player red cards (1 per 2.22 matches). Card vocabulary census across 9,855 real card incidents: `yellow` 9,041 / `red` 500 / `yellowRed` 314 — confirms the poller's existing `RED_CARD_INCIDENT_CLASSES` assumption was correct all along.

## Resultado (Task 5 — the decision gate)

**Dataset:** 43,966 rows built from `data/goles_chile.db` (Liga de Primera + Liga de Ascenso, one row per match/team/cutoff-minute). Split: train 24,986 (seasons 2022–2024) / validation 12,740 (season 2025) / test 6,240 (season 2026-to-date).

**Backtest on the held-out 2026 test season:**
- Brier score: 0.1649 (naive no-skill baseline: 0.1655)
- **Brier Skill Score: 0.0035**
- Poisson baseline BSS (blend=0.1, same test season): **-0.0129** — LightGBM beats it clearly, but that's a low bar
- Calibration: only two probability bins had test data ([0.0–0.2) n=3,325, pred 0.173/real 0.184; [0.2–0.4) n=2,915, pred 0.228/real 0.238) — reasonably well-calibrated within the range that occurs, but the test set never produces higher-probability predictions to check

**Feature importances confirm the architecture worked exactly as designed:** `market_*` (`own_market_wp`, `opp_market_wp`, `market_draw_wp`, `market_over25_wp`) and `own_linebreak_shots`/`own_transition_shots` all show **0.0 gain** — the model never split on them, exactly the "missing market data" convention working as intended, not a bug. Top real signal: `trailing_prior_xg`, `own_box_xg_total`, `own_max_shot_xg`, `own_setpiece_xg`, `own_xg_total` — the own-xG-model pipeline (Tasks 1–3) is clearly producing usable signal.

**xG-model validation recap (Tasks 1 & 3):** 0.68 correlation vs. Understat's proprietary xG (ceiling, investigated, not a bug); **0.9127** correlation vs. Sofascore's own published xG (the metric that actually validates the translation layer used here) — both hold up.

**Decision (per the plan's pre-agreed rule):** BSS ≈ 0 (0.0035, below the ≳0.01 bar for "meaningfully above zero") → **the Chile trial stops here.** The model does not show a clear edge on Chilean data without market odds. This matches the honest expectation set in the design ("without market odds... the Chilean model should land below the EPL's BSS 0.0335 — the trial's question is whether it stays meaningfully above zero") — it stayed essentially flat rather than clearly positive. Phase D (live inference + Telegram) is **not** started. Recommendation: wait for August's EPL/Bundesliga restart, where the full feature set including market odds exists; optionally revisit a market-aware Chilean retrain once Task 6's Betfair odds collection has accumulated enough weeks of data.
