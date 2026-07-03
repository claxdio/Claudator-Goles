from goles.features import compute_ml_features, compute_state_at_minute, goal_in_window

SAMPLE_SHOTS = [
    {"minute": 10, "team": "home", "xg": 0.10, "is_goal": False},
    {"minute": 34, "team": "home", "xg": 0.35, "is_goal": True},
    {"minute": 50, "team": "away", "xg": 0.20, "is_goal": False},
    {"minute": 63, "team": "away", "xg": 0.55, "is_goal": True},
    {"minute": 70, "team": "home", "xg": 0.18, "is_goal": False},
    {"minute": 74, "team": "home", "xg": 0.60, "is_goal": True},
]


def test_compute_state_at_minute_counts_goals_scored_so_far():
    state = compute_state_at_minute(SAMPLE_SHOTS, cutoff_minute=65, window=15)
    assert state.home_score == 1
    assert state.away_score == 1


def test_compute_state_at_minute_rolling_window_only_includes_recent_shots():
    state = compute_state_at_minute(SAMPLE_SHOTS, cutoff_minute=65, window=15)
    # window is (50, 65]; only the away shot at minute 63 falls inside it
    assert state.away_shots_last15 == 1
    assert abs(state.away_xg_last15 - 0.55) < 1e-9
    assert state.home_shots_last15 == 0
    assert abs(state.home_xg_last15 - 0.0) < 1e-9


def test_goal_in_window_detects_future_goal_for_team():
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=65, horizon=10, team="home") is True
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=65, horizon=10, team="away") is False


def test_goal_in_window_excludes_goals_outside_horizon():
    # the home goal at minute 34 is outside the (20, 30] window
    assert goal_in_window(SAMPLE_SHOTS, cutoff_minute=20, horizon=10, team="home") is False


ML_SAMPLE_SHOTS = [
    {"minute": 5, "team": "home", "xg": 0.05, "is_goal": False},
    {"minute": 12, "team": "home", "xg": 0.30, "is_goal": False},
    {"minute": 20, "team": "away", "xg": 0.10, "is_goal": False},
    {"minute": 34, "team": "home", "xg": 0.40, "is_goal": True},
    {"minute": 50, "team": "away", "xg": 0.15, "is_goal": False},
    {"minute": 63, "team": "away", "xg": 0.55, "is_goal": True},
    {"minute": 70, "team": "home", "xg": 0.18, "is_goal": False},
]


def test_compute_ml_features_home_perspective_basic_totals():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert features["is_home"] == 1.0
    assert features["score_diff"] == 0.0  # 1-1 at minute 65
    assert abs(features["own_xg_total"] - 0.75) < 1e-9  # 0.05+0.30+0.40
    assert abs(features["opp_xg_total"] - 0.80) < 1e-9  # 0.10+0.15+0.55
    assert abs(features["xg_diff"] - (0.75 - 0.80)) < 1e-9


def test_compute_ml_features_away_perspective_mirrors_home():
    home_features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    away_features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="away")
    assert away_features["is_home"] == 0.0
    assert abs(away_features["own_xg_total"] - home_features["opp_xg_total"]) < 1e-9
    assert abs(away_features["opp_xg_total"] - home_features["own_xg_total"]) < 1e-9
    assert away_features["score_diff"] == -home_features["score_diff"]


def test_compute_ml_features_big_chances_uses_xg_threshold():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert features["own_big_chances"] == 2.0  # the 0.30 and 0.40 xg shots
    assert abs(features["own_max_shot_xg"] - 0.40) < 1e-9


def test_compute_ml_features_time_since_shot_and_goal():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="away")
    assert abs(features["own_time_since_shot"] - 2.0) < 1e-9  # away's last shot: minute 63
    assert abs(features["time_since_goal"] - 2.0) < 1e-9  # last goal overall: minute 63


def test_compute_ml_features_never_uses_shots_after_cutoff():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert abs(features["own_xg_total"] - 0.75) < 1e-9  # excludes the 0.18 shot at minute 70


def test_compute_ml_features_minutes_remaining_and_interaction_when_tied():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=65, team="home")
    assert features["minutes_remaining"] == 25.0  # 90 - 65
    assert features["score_diff_x_minutes_remaining"] == 0.0


def test_compute_ml_features_score_diff_interaction_when_leading():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=40, team="home")
    assert features["score_diff"] == 1.0  # 1-0 at minute 40
    assert features["minutes_remaining"] == 50.0
    assert features["score_diff_x_minutes_remaining"] == 50.0


def test_compute_ml_features_trend_ratio_reflects_recent_burst():
    features = compute_ml_features(ML_SAMPLE_SHOTS, cutoff_minute=40, team="home")
    own_xg_rate = 0.75 / 40  # 0.05+0.30+0.40 over 40 minutes elapsed
    own_recent_rate = 0.40 / 15  # only the minute-34 shot falls in (25,40]
    expected_trend = own_recent_rate / own_xg_rate
    assert abs(features["own_trend"] - expected_trend) < 1e-6
