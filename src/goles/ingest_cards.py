from __future__ import annotations

from soccerdata._config import DATA_DIR

from goles.db import get_connection, init_db
from goles.loaders.understat import load_red_cards_from_cache, persist_red_cards


def main() -> None:
    conn = get_connection()
    init_db(conn)

    cache_dir = DATA_DIR / "Understat"
    print(f"Leyendo tarjetas rojas del cache crudo en {cache_dir}...")
    red_cards_by_game_id = load_red_cards_from_cache(cache_dir)
    total_cards = sum(len(events) for events in red_cards_by_game_id.values())
    print(f"{len(red_cards_by_game_id)} partidos con al menos una tarjeta roja ({total_cards} tarjetas en total).")

    processed, not_found = persist_red_cards(conn, red_cards_by_game_id)
    print(f"Procesados: {processed}. Partidos no encontrados en la base: {not_found}.")


if __name__ == "__main__":
    main()
