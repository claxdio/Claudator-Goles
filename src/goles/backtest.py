from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from goles.features import compute_state_at_minute, goal_in_window
from goles.model import dynamic_lambda, prob_goal_in_window
from goles.priors import trailing_xg_per90

CUTOFF_MINUTES = list(range(20, 81, 5))
HORIZON_MINUTES = 15
RECENT_WINDOW_MINUTES = 15
DEFAULT_BLEND = 0.5


@dataclass
class BacktestResult:
    predicted_probs: list[float]
    actual_outcomes: list[bool]

    @property
    def brier_score(self) -> float:
        n = len(self.predicted_probs)
        if n == 0:
            return float("nan")
        return sum(
            (p - float(o)) ** 2 for p, o in zip(self.predicted_probs, self.actual_outcomes)
        ) / n

    @property
    def no_skill_brier_score(self) -> float:
        """Brier score of the naive baseline that always predicts the
        empirical base rate (mean of actual_outcomes) instead of using any
        live signal. This is the reference score a model must beat to have
        any real skill."""
        n = len(self.actual_outcomes)
        if n == 0:
            return float("nan")
        base_rate = sum(float(o) for o in self.actual_outcomes) / n
        return base_rate * (1.0 - base_rate)

    @property
    def brier_skill_score(self) -> float:
        """Brier Skill Score: 1 - (model_brier / no_skill_brier). Positive
        means the model beats the naive base-rate baseline; zero means it's
        exactly as good; negative means it's worse than just guessing the
        base rate for every prediction."""
        ref = self.no_skill_brier_score
        if ref == 0.0 or math.isnan(ref):
            return float("nan")
        return 1.0 - (self.brier_score / ref)

    def calibration_bins(self, n_bins: int = 5) -> list[tuple[float, float, float, int]]:
        """Groups predictions into `n_bins` equal-width probability buckets
        and returns (bin_low, mean_predicted, mean_actual, count) for each
        non-empty bucket."""
        bucketed_preds: list[list[float]] = [[] for _ in range(n_bins)]
        bucketed_actuals: list[list[bool]] = [[] for _ in range(n_bins)]
        for p, o in zip(self.predicted_probs, self.actual_outcomes):
            idx = min(int(p * n_bins), n_bins - 1)
            bucketed_preds[idx].append(p)
            bucketed_actuals[idx].append(o)

        report = []
        for i in range(n_bins):
            if not bucketed_preds[i]:
                continue
            mean_pred = sum(bucketed_preds[i]) / len(bucketed_preds[i])
            mean_actual = sum(bucketed_actuals[i]) / len(bucketed_actuals[i])
            report.append((i / n_bins, mean_pred, mean_actual, len(bucketed_preds[i])))
        return report


def load_match_shots(
    conn: sqlite3.Connection, match_id: int, home_team_id: int, away_team_id: int
) -> list[dict]:
    rows = conn.execute(
        """SELECT minute, team_id, xg, is_goal,
                  location_x, location_y, situation, shot_type, last_action
           FROM shots WHERE match_id = ? ORDER BY minute""",
        (match_id,),
    ).fetchall()
    shots = []
    for minute, team_id, xg, is_goal, loc_x, loc_y, situation, shot_type, last_action in rows:
        team = "home" if team_id == home_team_id else "away"
        shots.append(
            {
                "minute": minute, "team": team, "xg": xg, "is_goal": bool(is_goal),
                "location_x": loc_x, "location_y": loc_y,
                "situation": situation, "shot_type": shot_type, "last_action": last_action,
            }
        )
    return shots


def load_match_cards(
    conn: sqlite3.Connection, match_id: int, home_team_id: int, away_team_id: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT team_id, minute FROM cards WHERE match_id = ? ORDER BY minute",
        (match_id,),
    ).fetchall()
    cards = []
    for team_id, minute in rows:
        team = "home" if team_id == home_team_id else "away"
        cards.append({"team": team, "minute": minute})
    return cards


def run_backtest(
    conn: sqlite3.Connection,
    team: str = "home",
    blend: float = DEFAULT_BLEND,
    cutoff_minutes: list[int] = CUTOFF_MINUTES,
) -> BacktestResult:
    """Replays every stored match at each cutoff minute in `cutoff_minutes`,
    predicting P(goal in the next HORIZON_MINUTES for `team`) with the
    Poisson baseline -- using each team's trailing season-to-date average
    xG as the pre-match prior (never the match's own xG) -- and comparing
    it against what actually happened."""
    predicted_probs: list[float] = []
    actual_outcomes: list[bool] = []

    matches = conn.execute(
        "SELECT match_id, home_team_id, away_team_id, league, season FROM matches"
    ).fetchall()

    for match_id, home_team_id, away_team_id, league, season in matches:
        shots = load_match_shots(conn, match_id, home_team_id, away_team_id)
        if not shots:
            continue
        team_id = home_team_id if team == "home" else away_team_id
        pre_match_xg = trailing_xg_per90(conn, team_id, league, season, match_id)

        for cutoff in cutoff_minutes:
            state = compute_state_at_minute(shots, cutoff, window=RECENT_WINDOW_MINUTES)
            recent_xg = state.home_xg_last15 if team == "home" else state.away_xg_last15
            lam = dynamic_lambda(
                pre_match_xg_per90=pre_match_xg,
                in_match_xg_recent=recent_xg,
                recent_window_minutes=RECENT_WINDOW_MINUTES,
                horizon_minutes=HORIZON_MINUTES,
                blend=blend,
            )
            predicted_probs.append(prob_goal_in_window(lam))
            actual_outcomes.append(goal_in_window(shots, cutoff, HORIZON_MINUTES, team))

    return BacktestResult(predicted_probs=predicted_probs, actual_outcomes=actual_outcomes)


def compare_blends(
    conn: sqlite3.Connection,
    team: str,
    blends: list[float],
    cutoff_minutes: list[int] = CUTOFF_MINUTES,
) -> dict[float, BacktestResult]:
    """Runs run_backtest once per blend value and returns each result keyed
    by the blend used, so different blend settings can be compared on the
    same data."""
    return {
        blend: run_backtest(conn, team=team, blend=blend, cutoff_minutes=cutoff_minutes)
        for blend in blends
    }


def print_comparison(results: dict[float, BacktestResult]) -> None:
    print(f"{'blend':>6} | {'brier':>8} | {'no-skill':>8} | {'BSS':>8} | {'n':>6}")
    for blend, result in sorted(results.items()):
        print(
            f"{blend:6.2f} | {result.brier_score:8.4f} | "
            f"{result.no_skill_brier_score:8.4f} | {result.brier_skill_score:8.4f} | "
            f"{len(result.predicted_probs):6d}"
        )


def print_report(result: BacktestResult, n_bins: int = 5) -> None:
    print(f"Muestras evaluadas: {len(result.predicted_probs)}")
    print(f"Brier score: {result.brier_score:.4f}")
    print(f"Brier score (base ingenua): {result.no_skill_brier_score:.4f}")
    print(f"Brier Skill Score: {result.brier_skill_score:.4f}  (>0 = mejor que la base ingenua)")
    print("Calibracion (bin_low, prob. media predicha, frecuencia real, n):")
    bin_width = 1 / n_bins
    for bin_low, mean_pred, mean_actual, count in result.calibration_bins(n_bins):
        print(f"  [{bin_low:.1f}-{bin_low + bin_width:.1f}) pred={mean_pred:.3f} real={mean_actual:.3f} n={count}")
