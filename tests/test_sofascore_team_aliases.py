from goles.sofascore.team_aliases import SOFASCORE_TEAM_NAME_ALIASES, normalize_sofascore_team_name


def test_normalize_sofascore_team_name_passes_through_unmapped_names():
    assert normalize_sofascore_team_name("Arsenal") == "Arsenal"
    assert normalize_sofascore_team_name("Some Unmapped Team") == "Some Unmapped Team"


def test_sofascore_team_name_aliases_has_no_identity_entries():
    for sofascore_name, our_name in SOFASCORE_TEAM_NAME_ALIASES.items():
        assert sofascore_name != our_name
