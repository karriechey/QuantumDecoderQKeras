# Python setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and every
dependency from a single lockfile. The committed environment targets **macOS Apple
Silicon, CPU-only**, on **Python 3.11** (uv installs the interpreter for you).

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other methods (Homebrew, etc.) are in the
[uv install docs](https://docs.astral.sh/uv/getting-started/installation/).

## 2. Create the environment

From the repository root:

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and builds a `.venv/` with the full stack:
Stim, PyMatching, sinter, stimbposd, TensorFlow 2.15 / Keras 2.15, QKeras, and
numpy / scipy / scikit-learn / pandas / matplotlib, plus the Jupyter runtime. You do
not need to activate the venv — prefix commands with `uv run`.

## 3. Verify the install

```bash
uv run pytest
```

Expected: **28 passed**. This imports every dependency and local module, runs a
TensorFlow op on CPU, and trains a small QKeras quantized model — a quick confirmation
that the stack works on your machine.

## 4. Use the notebooks

```bash
uv run jupyter lab
```

Or execute a notebook headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute Stim_playground.ipynb
```

Self-contained notebooks such as `Stim_playground.ipynb` and
`surface_code_d4_r2_CNN.ipynb` generate their data via Stim and run end to end.

## Notes

- **Scope:** the lock is scoped to macOS (`sys_platform == 'darwin'`). A Linux + NVIDIA
  GPU environment for longer training runs is planned as separate work.
- **Versions:** the stack is pinned to the TensorFlow 2.15 / numpy 1.26 generation so
  QKeras (which requires Keras 2) works; this also matches the era the notebooks were
  written against.
- **Heavier notebooks:** some notebooks use the slow BP+OSD decoder (`stimbposd`) or
  expect saved data / model files, so they may not yet run out of the box.
