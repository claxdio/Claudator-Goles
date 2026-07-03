from __future__ import annotations

from goles.backtest import compare_blends, print_comparison, print_report, run_backtest
from goles.db import get_connection, init_db
from goles.loaders.understat import fetch_understat_shots, persist_shots, shots_to_records

LEAGUES = ["ENG-Premier League", "GER-Bundesliga"]
SEASONS = ["2324"]
# Cutoffs 75 and 80 project a 15-minute goal window past most matches' real
# playing time (regulation ends at 90 + stoppage), which structurally lowers
# the true positive rate for those two cutoffs specifically -- excluded here
# to see whether that alone explains some of the high-probability overconfidence.
EARLY_CUTOFFS = list(range(20, 71, 5))


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print(f"Descargando datos de Understat para {LEAGUES} temporada {SEASONS}...")
    shots_df = fetch_understat_shots(LEAGUES, SEASONS)
    records = shots_to_records(shots_df)
    print(f"{len(records)} eventos de tiro descargados. Guardando en la base de datos...")
    persist_shots(conn, records)

    print("\n=== Backtest con blend=0.5 (default), todos los cortes ===")
    result = run_backtest(conn, team="home")
    print_report(result)

    print("\n=== Backtest con blend=0.5, excluyendo cortes tardios (75, 80) ===")
    result_early = run_backtest(conn, team="home", cutoff_minutes=EARLY_CUTOFFS)
    print_report(result_early)

    print("\n=== Comparacion de valores de blend (cortes tempranos) ===")
    comparison = compare_blends(
        conn, team="home", blends=[0.1, 0.3, 0.5, 0.7, 0.9], cutoff_minutes=EARLY_CUTOFFS
    )
    print_comparison(comparison)


if __name__ == "__main__":
    main()
