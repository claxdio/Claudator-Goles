# Closing-Line Odds Experiment — Design

**Status:** Approved (quick experiment, not a production change)

## Motivation

The market-odds/rest-days/red-card plan (`docs/superpowers/plans/2026-07-11-goal-predictor-market-rest-features.md`) confirmed `own_market_wp` — the team's own no-vig, market-implied win probability from football-data.co.uk's **pre-match average odds** (`AvgH`/`AvgD`/`AvgA`) — is by far the single most valuable feature in the model (4x the next feature's importance). That plan's own "Próximos pasos" flagged an open question: football-data.co.uk also publishes true **closing-line** odds (`AvgCH`/`AvgCD`/`AvgCA`, plus `AvgC>2.5`/`AvgC<2.5` for the over/under market) which were never tried. Closing lines reflect all pre-match information (including late team news) right up to kickoff and are widely considered sharper than earlier-collected average odds — which is exactly the kind of price a live pipeline could realistically fetch close to kickoff.

Before investing in live-odds infrastructure (Betfair Exchange, delayed key, VPS-hosted), we want a cheap, fast answer: **do closing lines actually beat pre-match average odds on held-out BSS?** If yes, that reframes what "as fresh as possible" means for the live pipeline. If flat or worse, the pre-match-average approach already wired into production is good enough and live-odds work doesn't need to chase closing-line freshness specifically.

## Data availability (verified)

Fetched real CSVs for all 6 tracked seasons (`1819`–`2324`) x 2 leagues (`E0`, `D1`) from football-data.co.uk:

| Season | Closing columns (`AvgCH/AvgCD/AvgCA`, `AvgC>2.5/AvgC<2.5`) present? |
|---|---|
| 1819 | No (only the old Betbrain-branded pre-match columns exist) |
| 1920 – 2324 | Yes, both leagues |

`1819` is never the `TEST_SEASON` (`2324`) or `VALIDATION_SEASON` (`2223`) in either training script, so missing closing data there only reduces training-row signal quality for that one season — it does not block or bias the held-out comparison.

## Scope

**Throwaway experiment, not a production change.** No schema changes, no new columns on `matches`, no persistence of closing odds anywhere, no changes to `FEATURE_NAMES` or `build_dataset`, no automated tests. This matches the repo's existing precedent for I/O-heavy one-off scripts (`ingest_odds.py`, `ingest_cards.py`, `train_gbt_replication.py`) that skip TDD in favor of a single real, manually-verified run. If the result says closing lines win, a *follow-up* plan (full TDD, schema change, wired into production) formalizes it — exactly the same two-step process already used for market-odds/rest-days/red-cards.

## Design

New script: `src/goles/experiment_closing_lines.py`.

1. **Fetch raw odds** — reuse `goles.loaders.football_data.fetch_odds(LEAGUE_CODES, SEASONS)` unchanged. The raw CSV already contains both the pre-match average columns (already used in production) and the closing columns (unused so far) — no changes needed to `football_data.py`.

2. **Match closing odds to match_ids** — a small function local to this script, `_match_closing_odds(conn, odds_df) -> dict[int, tuple[float, float, float, float]]` (`match_id -> (close_home_wp, close_draw_wp, close_away_wp, close_over_wp)`). This deliberately duplicates `persist_odds`'s date/team-name/league/season matching logic (~15 lines) rather than modifying `persist_odds` itself, to keep this experiment fully isolated from production code. Rows missing any of the 5 closing columns, or with no matching `match_id`, are simply absent from the dict (handled as "no closing data" downstream, same as unmatched rows elsewhere in this codebase).

3. **Build two feature variants from one dataset** — call `build_dataset(conn)` once (as `train_gbt.py` does). This produces rows whose `own_market_wp`/`opp_market_wp`/`market_draw_wp`/`market_over25_wp` reflect **pre-match average** odds (variant A — the current production baseline). Build variant B by deep-copying each row's `features` dict and overwriting those same 4 keys with the closing-derived values from step 2, keyed by `(match_id, team)` to resolve which side is "own" vs "opp". A `match_id` absent from the closing dict gets `0.0` for all 4 — the same "no market data available" convention `build_dataset` already uses for its own missing-market case, so both variants use one consistent missingness convention.

4. **Train both variants in the same run** — using the exact same `TEST_SEASON="2324"` / `VALIDATION_SEASON="2223"` split, `train_gbt`/`fit_platt_scaling`/`raw_predictions`/`BacktestResult` calls as `train_gbt.py`, so the comparison is apples-to-apples and isn't confounded by LightGBM's own run-to-run randomness (both variants train in the same process, back to back, rather than diffing against the previously-recorded 0.0335 from a separate historical run).

5. **Report** — print, for each variant: Brier score, no-skill Brier, BSS, and feature importance for the 4 market-probability features specifically (so we can see directly whether closing-derived `own_market_wp` outranks pre-match-average `own_market_wp`). Also print closing-odds match coverage (`matched / total`, mirroring `ingest_odds.py`'s coverage sanity check) so a low match rate is visible rather than silently degrading the comparison.

6. Script does **not** call `save_model` — it must never overwrite the persisted production model (`data/model/booster.txt` / `platt.json`).

## Decision rule

Same ±0.002 band used in prior retrain comparisons: if variant B's BSS beats variant A by more than +0.002, that's a real signal favoring closing-line freshness for the live pipeline design. Within ±0.002 is flat (pre-match average is good enough, no need to chase closing-line recency live). A regression beyond -0.002 favors sticking with pre-match-average-style timing.

## Verification

Manual run only (`python -m goles.experiment_closing_lines`), console output reviewed for both BSS numbers and coverage %. Findings get appended to this doc (a "## Resultado" section) once run, informing the subsequent decision on live-odds pipeline design — no separate implementation plan needed unless the result says closing lines should be formalized into production.
