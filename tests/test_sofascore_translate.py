import pytest

from goles.sofascore.translate import UnknownVocabularyError, translate_shot


def _sofa_shot(**overrides):
    shot = {
        "id": 7684954, "time": 20, "shotType": "goal", "situation": "corner",
        "isHome": True, "playerCoordinates": {"x": 5.0, "y": 44.1},
        "bodyPart": "head",
    }
    shot.update(overrides)
    return shot


def test_translate_maps_coordinates_to_understat_convention():
    out = translate_shot(_sofa_shot())
    # Sofascore x=5 (5% of pitch length from the goal line) -> Understat 0.95
    assert out["location_x"] == pytest.approx(0.95)
    assert out["location_y"] == pytest.approx(0.441)


def test_translate_maps_vocabularies_and_outcome():
    out = translate_shot(_sofa_shot())
    assert out["situation"] == "FromCorner"
    assert out["shot_type"] == "Head"
    assert out["is_goal"] is True
    assert out["minute"] == 20
    assert out["is_home"] is True


def test_translate_open_play_variants_all_map_to_openplay():
    for sofa_situation in ("regular", "assisted", "fast-break"):
        out = translate_shot(_sofa_shot(situation=sofa_situation, shotType="miss"))
        assert out["situation"] == "OpenPlay"
        assert out["is_goal"] is False


def test_translate_set_piece_vocabulary():
    assert translate_shot(_sofa_shot(situation="set-piece"))["situation"] == "SetPiece"
    assert translate_shot(_sofa_shot(situation="throw-in-set-piece"))["situation"] == "SetPiece"
    assert translate_shot(_sofa_shot(situation="free-kick"))["situation"] == "DirectFreekick"
    assert translate_shot(_sofa_shot(situation="penalty"))["situation"] == "Penalty"


def test_translate_fails_loud_on_unknown_situation():
    with pytest.raises(UnknownVocabularyError, match="volea-imaginaria"):
        translate_shot(_sofa_shot(situation="volea-imaginaria"))


def test_translate_tolerates_missing_body_part():
    out = translate_shot(_sofa_shot(bodyPart=None))
    assert out["shot_type"] is None  # xG model one-hots it as all-zero


def test_translate_own_goal_flips_team_and_flags_zero_xg():
    # Sofascore attributes the shot to the scoring-against player (isHome
    # here is the away team), but the goal counts for the home side.
    out = translate_shot(_sofa_shot(situation="own-goal", goalType="own", isHome=False))
    assert out["is_goal"] is True
    assert out["is_home"] is True  # flipped from the raw isHome=False
    assert out["is_own_goal"] is True


def test_translate_regular_goal_is_not_flagged_as_own_goal():
    out = translate_shot(_sofa_shot())
    assert out["is_own_goal"] is False
