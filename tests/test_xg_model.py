import math

from goles.xg_model import (
    PENALTY_XG,
    XG_FEATURE_NAMES,
    predict_xg,
    shot_to_features,
    train_xg_model,
)


def _shot(x, y, situation="OpenPlay", shot_type="RightFoot", is_goal=False):
    return {
        "location_x": x, "location_y": y,
        "situation": situation, "shot_type": shot_type, "is_goal": is_goal,
    }


def test_shot_to_features_length_matches_names():
    features = shot_to_features(_shot(0.9, 0.5))
    assert len(features) == len(XG_FEATURE_NAMES)


def test_distance_is_zero_at_goal_center_and_grows_with_distance():
    near = shot_to_features(_shot(0.99, 0.5))
    far = shot_to_features(_shot(0.5, 0.5))
    dist_idx = XG_FEATURE_NAMES.index("distance_m")
    assert near[dist_idx] < far[dist_idx]
    assert near[dist_idx] < 2.0  # ~1 meter out, dead center


def test_angle_is_larger_from_the_center_than_from_the_byline():
    center = shot_to_features(_shot(0.88, 0.5))
    wide = shot_to_features(_shot(0.88, 0.1))
    angle_idx = XG_FEATURE_NAMES.index("angle_rad")
    assert center[angle_idx] > wide[angle_idx]


def test_train_and_predict_learns_that_close_beats_far():
    import random

    random.seed(7)
    shots = []
    # synthetic but directionally-real data: close shots score more often
    for _ in range(2000):
        close = random.random() < 0.5
        x = random.uniform(0.88, 0.98) if close else random.uniform(0.55, 0.75)
        goal = random.random() < (0.35 if close else 0.03)
        shots.append(_shot(x, random.uniform(0.35, 0.65), is_goal=goal))
    booster = train_xg_model(shots)
    xg_close = predict_xg(booster, _shot(0.93, 0.5))
    xg_far = predict_xg(booster, _shot(0.6, 0.5))
    assert xg_close > xg_far
    assert 0.0 <= xg_far <= 1.0


def test_penalties_get_fixed_xg_and_are_excluded_from_training():
    shots = [_shot(0.9, 0.5, is_goal=True) for _ in range(50)]
    shots += [_shot(0.9, 0.5, is_goal=False) for _ in range(50)]
    shots += [_shot(0.88, 0.5, situation="Penalty", is_goal=True) for _ in range(400)]
    booster = train_xg_model(shots)
    assert predict_xg(booster, _shot(0.88, 0.5, situation="Penalty")) == PENALTY_XG
    # non-penalty prediction should reflect only the 50/50 non-penalty data,
    # not be dragged toward 1.0 by the 400 penalty goals
    assert predict_xg(booster, _shot(0.9, 0.5)) < 0.8
