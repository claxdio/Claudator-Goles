from goles.betfair.team_aliases import BETFAIR_TEAM_NAME_ALIASES, normalize_betfair_team_name


def test_normalize_betfair_team_name_passes_through_unmapped_names():
    assert normalize_betfair_team_name("Arsenal") == "Arsenal"
    assert normalize_betfair_team_name("Some Unmapped Team") == "Some Unmapped Team"


def test_betfair_team_name_aliases_has_no_identity_entries():
    for betfair_name, our_name in BETFAIR_TEAM_NAME_ALIASES.items():
        assert betfair_name != our_name
