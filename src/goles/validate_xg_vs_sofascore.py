from __future__ import annotations

import numpy as np
import tls_requests

from goles.sofascore.client import get_shotmap
from goles.sofascore.translate import UnknownVocabularyError, translate_shot
from goles.train_xg import XG_MODEL_PATH
from goles.xg_model import load_xg_model, predict_xg

# A finished top-tier match where Sofascore publishes real per-shot xG
# (FIFA World Cup knockout match observed during design). Any finished
# top-tier event id works -- pass a different one as argv[1] if needed.
DEFAULT_EVENT_ID = 12813015


def main(event_id: int = DEFAULT_EVENT_ID) -> None:
    booster = load_xg_model(XG_MODEL_PATH)
    client = tls_requests.Client()
    shots = get_shotmap(client, event_id)
    print(f"{len(shots)} tiros en el evento {event_id}.")

    ours, theirs = [], []
    skipped = 0
    for shot in shots:
        if shot.get("xg") is None or shot.get("situation") == "penalty":
            continue
        try:
            translated = translate_shot(shot)
        except UnknownVocabularyError as exc:
            print(f"  omitido: {exc}")
            skipped += 1
            continue
        ours.append(predict_xg(booster, translated))
        theirs.append(shot["xg"])

    if len(ours) < 5:
        print("Muy pocos tiros comparables -- probar con otro event id de liga top.")
        return
    ours_np, theirs_np = np.array(ours), np.array(theirs)
    corr = float(np.corrcoef(ours_np, theirs_np)[0, 1])
    print(f"Comparables: {len(ours)} (omitidos: {skipped})")
    print(f"correlacion nuestro-xG vs Sofascore-xG: {corr:.4f}  (esperado >= 0.75)")
    print(f"MAE: {float(np.mean(np.abs(ours_np - theirs_np))):.4f}")
    print(f"medias: nuestro {ours_np.mean():.4f} vs Sofascore {theirs_np.mean():.4f}")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVENT_ID)
