# Model Persistence + Platt-Scaling Convergence Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the two non-optional final-review findings recorded in the GBT-correction-layer plan's Próximos pasos: (1) `fit_platt_scaling` silently returns invalid calibration parameters when scipy's optimizer fails to converge, and (2) the validated LightGBM model (booster + Platt params) has no persistence mechanism, which blocks Phase 2 (live inference) from ever loading it. The bootstrap-CI item from that same review is explicitly out of scope here (the user confirmed skipping it — two independent season replications already point the same direction).

**Architecture:** `fit_platt_scaling` gets a convergence check on scipy's `OptimizeResult.success` that raises `RuntimeError` instead of silently returning garbage `(a, b)`. A new `src/goles/persistence.py` module adds `save_model`/`load_model`, storing the LightGBM booster in its native text format (`booster.txt`) and the two Platt scalars as a small JSON side-car (`platt.json`) inside a directory — this mirrors how LightGBM's own `Booster.save_model`/`Booster(model_file=...)` already work, so no new serialization format is invented. `train_gbt.py` (the main, validated run — not the replication script, which exists only to prove repeatability, not to produce a deployable artifact) is wired to call `save_model` at the end of `main()`, writing to `data/model/`, which is already covered by the repo's blanket `data/` gitignore rule (same precedent as `data/goles.db`).

**Tech Stack:** Unchanged — Python 3.11+, lightgbm, scipy, pytest. No new dependencies (JSON and pathlib are stdlib).

## Global Constraints

- No new dependencies — `json`, `pathlib.Path` (stdlib) and `lgb.Booster.save_model`/`lgb.Booster(model_file=...)` (already a transitive dependency via lightgbm) cover everything needed.
- Must run correctly on Windows/PowerShell — use `pathlib.Path` throughout, no POSIX-only path assumptions (matches every prior plan's constraint).
- No network calls in any unit test (Tasks 1-2). Task 3 (wiring + real run) is a manual-verification task that touches the real `data/goles.db` and writes real artifacts to `data/model/`, following the same precedent as `ingest_history.py` and the two `train_gbt*.py` scripts — no automated test for the script's `main()` itself.
- `data/` is already gitignored at the repo root — `data/model/` needs no new gitignore entry.
- Only `train_gbt.py` (the main run, TEST_SEASON="2324") gets persistence wiring. `train_gbt_replication.py` stays unmodified — its purpose is proving repeatability on a different season split, not producing the deployable artifact.
- All existing tests (64 as of the last commit) must keep passing unmodified.

---

### Task 1: `fit_platt_scaling` raises on optimizer non-convergence

**Files:**
- Modify: `src/goles/gbt_model.py`
- Test: `tests/test_gbt_model.py` (append)

**Interfaces:**
- Produces: `fit_platt_scaling` keeps its existing signature `(raw_probs: list[float], y_true: list[int]) -> tuple[float, float]`, but now raises `RuntimeError` when `scipy.optimize.minimize`'s `OptimizeResult.success` is `False`, instead of silently returning `result.x` regardless.
- Consumes: `scipy.optimize.minimize` (already imported in `gbt_model.py` as `minimize`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gbt_model.py`:

```python
from unittest.mock import patch

import numpy as np
import pytest


def test_fit_platt_scaling_raises_when_optimizer_does_not_converge():
    class FakeResult:
        success = False
        message = "mock optimizer failure"
        x = np.array([1.0, 0.0])

    with patch("goles.gbt_model.minimize", return_value=FakeResult()):
        with pytest.raises(RuntimeError, match="mock optimizer failure"):
            fit_platt_scaling([0.1, 0.9], [0, 1])
```

Add the import at the top of the file (it currently only has `import random` and the `goles.gbt_model` import) — place `from unittest.mock import patch`, `import numpy as np`, and `import pytest` right after `import random`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_gbt_model.py -v -k does_not_converge`
Expected: FAIL — `Failed: DID NOT RAISE <class 'RuntimeError'>` (the current implementation ignores `result.success` entirely and just returns `(1.0, 0.0)`).

- [ ] **Step 3: Implement**

In `src/goles/gbt_model.py`, replace the end of `fit_platt_scaling`:

```python
    result = minimize(neg_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a, b = result.x
    return float(a), float(b)
```

with:

```python
    result = minimize(neg_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(f"Platt scaling optimization did not converge: {result.message}")
    a, b = result.x
    return float(a), float(b)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_gbt_model.py -v`
Expected: all pass (3 existing + 1 new = 4).

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: 65 passed, no regressions.

- [ ] **Step 6: Commit**

```powershell
git add src/goles/gbt_model.py tests/test_gbt_model.py
git commit -m "fix: raise instead of silently returning invalid params when Platt scaling fails to converge"
```

---

### Task 2: Model persistence module (`save_model`/`load_model`)

**Files:**
- Create: `src/goles/persistence.py`
- Test: Create `tests/test_persistence.py`

**Interfaces:**
- Produces: `save_model(booster: lgb.Booster, platt_params: tuple[float, float], model_dir: Path) -> None` (creates `model_dir` and its parents if missing, writes `booster.txt` and `platt.json` inside it); `load_model(model_dir: Path) -> tuple[lgb.Booster, tuple[float, float]]` (reads the same two files back).
- Consumes: `lgb.Booster.save_model`/`lgb.Booster(model_file=...)` (lightgbm, already a dependency); `goles.gbt_model.train_gbt`/`raw_predictions`/`fit_platt_scaling` (existing, used only in the test to produce a real booster to round-trip).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_persistence.py`:

```python
import random

from goles.gbt_model import fit_platt_scaling, raw_predictions, train_gbt
from goles.persistence import load_model, save_model


def _train_tiny_booster():
    random.seed(7)
    X_train, y_train = [], []
    for _ in range(200):
        x0 = random.random()
        X_train.append([x0, random.random()])
        y_train.append(1 if x0 > 0.5 else 0)
    X_valid, y_valid = [], []
    for _ in range(50):
        x0 = random.random()
        X_valid.append([x0, random.random()])
        y_valid.append(1 if x0 > 0.5 else 0)
    booster = train_gbt(X_train, y_train, X_valid, y_valid)
    return booster, X_valid, y_valid


def test_save_and_load_model_round_trips_predictions_and_platt_params(tmp_path):
    booster, X_valid, y_valid = _train_tiny_booster()
    raw_valid = raw_predictions(booster, X_valid)
    a, b = fit_platt_scaling(raw_valid, y_valid)

    model_dir = tmp_path / "model"
    save_model(booster, (a, b), model_dir)
    loaded_booster, (loaded_a, loaded_b) = load_model(model_dir)

    assert abs(loaded_a - a) < 1e-9
    assert abs(loaded_b - b) < 1e-9
    original_preds = raw_predictions(booster, X_valid)
    loaded_preds = raw_predictions(loaded_booster, X_valid)
    assert len(original_preds) == len(loaded_preds)
    for p_orig, p_loaded in zip(original_preds, loaded_preds):
        assert abs(p_orig - p_loaded) < 1e-6


def test_save_model_creates_missing_parent_directories(tmp_path):
    booster, X_valid, y_valid = _train_tiny_booster()
    a, b = fit_platt_scaling(raw_predictions(booster, X_valid), y_valid)

    nested_dir = tmp_path / "nested" / "model"
    save_model(booster, (a, b), nested_dir)
    assert (nested_dir / "booster.txt").exists()
    assert (nested_dir / "platt.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_persistence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'goles.persistence'`.

- [ ] **Step 3: Implement**

Create `src/goles/persistence.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb


def save_model(
    booster: lgb.Booster, platt_params: tuple[float, float], model_dir: Path
) -> None:
    """Persists a trained + calibrated model to `model_dir`: the LightGBM
    booster in its native text format (`booster.txt`) and the Platt
    scaling parameters as `platt.json` (`{"a": ..., "b": ...}`). Creates
    `model_dir` and any missing parent directories."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "booster.txt"))
    a, b = platt_params
    with open(model_dir / "platt.json", "w", encoding="utf-8") as fh:
        json.dump({"a": a, "b": b}, fh)


def load_model(model_dir: Path) -> tuple[lgb.Booster, tuple[float, float]]:
    """Loads a model previously written by `save_model`. Returns
    `(booster, (a, b))`."""
    model_dir = Path(model_dir)
    booster = lgb.Booster(model_file=str(model_dir / "booster.txt"))
    with open(model_dir / "platt.json", encoding="utf-8") as fh:
        data = json.load(fh)
    return booster, (data["a"], data["b"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_persistence.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv\Scripts\pytest.exe -q`
Expected: 67 passed, no regressions.

- [ ] **Step 6: Commit**

```powershell
git add src/goles/persistence.py tests/test_persistence.py
git commit -m "feat: add booster + Platt-scaling model persistence"
```

---

### Task 3: Wire persistence into `train_gbt.py` and produce the real artifact

**Files:**
- Modify: `src/goles/train_gbt.py`

**Interfaces:**
- Consumes: Task 2's `save_model`.
- Produces: a saved model at `data/model/` (`booster.txt` + `platt.json`) after every real run of `python -m goles.train_gbt`.

No automated test — same precedent as `ingest_history.py` and the two `train_gbt*.py` scripts (real-data, manual-verification tasks).

- [ ] **Step 1: Update the script**

In `src/goles/train_gbt.py`, add the import and a `MODEL_DIR` constant, then save at the end of `main()`.

Change the import block:

```python
from __future__ import annotations

from goles.backtest import BacktestResult
from goles.dataset import FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt
```

to:

```python
from __future__ import annotations

from pathlib import Path

from goles.backtest import BacktestResult
from goles.dataset import FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt
from goles.persistence import save_model
```

Add the constant next to the other module-level constants (after `POISSON_COMPARISON_BLEND = 0.1`):

```python
MODEL_DIR = Path("data") / "model"
```

At the very end of `main()`, after the feature-importance print loop, add:

```python
    save_model(booster, (a, b), MODEL_DIR)
    print(f"\nModelo guardado en {MODEL_DIR} (booster.txt + platt.json).")
```

- [ ] **Step 2: Run the script for real**

Run (PowerShell, venv activated):
```powershell
python -m goles.train_gbt
```
Expected: same training/evaluation output as before (BSS around 0.0190 per the feature-enrichment plan's recorded result), plus a final line confirming the model was saved. Verify the files exist:
```powershell
python -c "from pathlib import Path; d = Path('data/model'); print((d / 'booster.txt').exists(), (d / 'platt.json').exists())"
```
Expected: `True True`.

- [ ] **Step 3: Verify the saved model round-trips against real predictions**

```powershell
python -c "from goles.persistence import load_model; from goles.gbt_model import raw_predictions, apply_platt_scaling; from goles.dataset import build_dataset, rows_to_arrays, split_by_season; from goles.db import get_connection; from pathlib import Path; conn = get_connection(); rows = build_dataset(conn, blend=0.1); _, _, test_rows = split_by_season(rows, '2324', '2223'); X_test, y_test = rows_to_arrays(test_rows); booster, (a, b) = load_model(Path('data/model')); preds = apply_platt_scaling(raw_predictions(booster, X_test), a, b); print(len(preds), preds[:3])"
```
Expected: prints `17836 [...]` (or the current row count) with three calibrated probabilities between 0 and 1 — confirms the persisted model loads and produces predictions without error.

- [ ] **Step 4: Confirm no regressions, then commit**

Run: `.venv\Scripts\pytest.exe -q` → expected 67 passed (unchanged from Task 2 — this task adds no new automated tests).

```powershell
git add src/goles/train_gbt.py
git commit -m "feat: persist the validated GBT model after training"
```

## Próximos pasos (fuera de alcance de este plan)

Con el modelo persistido, la Fase 2 (pipeline en vivo) puede ahora cargar `data/model/booster.txt` + `platt.json` vía `goles.persistence.load_model` en vez de tener que reentrenar en cada arranque. Sigue pendiente de revisiones anteriores: wiring de ClubElo como señal independiente adicional (requiere emparejar nombres de equipos entre ClubElo y Understat); guardas contra fechas vacías/no-ISO en `trailing_xg_per90`; una decisión documentada para cuando un partido anterior tiene cero tiros registrados; y, del plan de enriquecimiento de features, evaluar si las dos features derivadas de `lastAction` (sin equivalente en vivo) vale la pena excluirlas antes de construir el pipeline de features en vivo. El intervalo de confianza bootstrap por partido para el BSS quedó explícitamente fuera de alcance de este plan (confirmado con el usuario) — sigue disponible como mejora de rigor estadístico si se quiere retomar más adelante.
