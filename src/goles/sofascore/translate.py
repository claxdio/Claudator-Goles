from __future__ import annotations

# Observed Sofascore situation vocabulary -> Understat's. Extended
# empirically as the backfill logs unknown values (which fail loud below);
# never guess an entry without a real observed value.
SITUATION_MAP = {
    "regular": "OpenPlay",
    "assisted": "OpenPlay",
    "fast-break": "OpenPlay",
    "corner": "FromCorner",
    "set-piece": "SetPiece",
    "free-kick": "DirectFreekick",
    "penalty": "Penalty",
}

BODY_PART_MAP = {
    "right-foot": "RightFoot",
    "left-foot": "LeftFoot",
    "head": "Head",
    "other": "OtherBodyPart",
}


class UnknownVocabularyError(ValueError):
    """A Sofascore vocabulary value we have never observed -- fail loud so
    the mapping table gets extended deliberately, never guessed."""


def translate_shot(sofa_shot: dict) -> dict:
    """Raw Sofascore shot dict -> Understat-convention dict, so everything
    downstream (xg_model, features.py, the goles.db schema) works on
    Chilean data unchanged. Sofascore x is the % of pitch length measured
    from the opponent's goal line (x=5 is point blank); Understat
    location_x is the 0-1 fraction toward the attacking goal."""
    situation_raw = sofa_shot.get("situation")
    if situation_raw not in SITUATION_MAP:
        raise UnknownVocabularyError(f"situacion Sofascore desconocida: {situation_raw!r}")

    body_raw = sofa_shot.get("bodyPart")
    if body_raw is not None and body_raw not in BODY_PART_MAP:
        raise UnknownVocabularyError(f"bodyPart Sofascore desconocido: {body_raw!r}")

    coordinates = sofa_shot.get("playerCoordinates") or {}
    x = coordinates.get("x")
    y = coordinates.get("y")
    if x is None or y is None:
        raise UnknownVocabularyError("tiro sin coordenadas -- no se puede calcular xG")

    return {
        "minute": sofa_shot["time"],
        "location_x": 1.0 - (x / 100.0),
        "location_y": y / 100.0,
        "situation": SITUATION_MAP[situation_raw],
        "shot_type": BODY_PART_MAP[body_raw] if body_raw is not None else None,
        "is_goal": sofa_shot.get("shotType") == "goal",
        "is_home": bool(sofa_shot.get("isHome")),
    }
