from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from goles.features import compute_state_at_minute, goal_in_window
from goles.model import dynamic_lambda, prob_goal_in_window

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


def _load_match_shots(
    conn: sqlite3.Connection, match_id: int, home_team_id: int, away_team_id: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT minute, team_id, xg, is_goal FROM shots WHERE match_id = ? ORDER BY minute",
        (match_id,),
    ).fetchall()
    shots = []
    for minute, team_id, xg, is_goal in rows:
        team = "home" if team_id == home_team_id else "away"
        shots.append({"minute": minute, "team": team, "xg": xg, "is_goal": bool(is_goal)})
    return shots


def _pre_match_xg_per90(shots: list[dict], team: str) -> float:
    """Simplified prior: sums the team's shot xG across the whole match.
    See the 'look-ahead bias' note in this plan's Global Constraints — this
    is a placeholder prior for validating the pipeline, not a production
    pre-match estimate."""
    return sum(s["xg"] for s in shots if s["team"] == team)


def run_backtest(
    conn: sqlite3.Connection, team: str = "home", blend: float = DEFAULT_BLEND
) -> BacktestResult:
    """Replays every stored match at each cutoff minute in CUTOFF_MINUTES,
    predicting P(goal in the next HORIZON_MINUTES for `team`) with the
    Poisson baseline, and comparing it against what actually happened."""
    predicted_probs: list[float] = []
    actual_outcomes: list[bool] = []

    matches = conn.execute("SELECT match_id, home_team_id, away_team_id FROM matches").fetchall()

    for match_id, home_team_id, away_team_id in matches:
        shots = _load_match_shots(conn, match_id, home_team_id, away_team_id)
        if not shots:
            continue
        pre_match_xg = _pre_match_xg_per90(shots, team)

        for cutoff in CUTOFF_MINUTES:
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


def print_report(result: BacktestResult, n_bins: int = 5) -> None:
    print(f"Muestras evaluadas: {len(result.predicted_probs)}")
    print(f"Brier score: {result.brier_score:.4f}")
    print("Calibracion (bin_low, prob. media predicha, frecuencia real, n):")
    bin_width = 1 / n_bins
    for bin_low, mean_pred, mean_actual, count in result.calibration_bins(n_bins):
        print(f"  [{bin_low:.1f}-{bin_low + bin_width:.1f}) pred={mean_pred:.3f} real={mean_actual:.3f} n={count}")
