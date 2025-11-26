# WARP

This file documents how to work with this project inside Warp.

> Note: This project appears to be a Python / Streamlit machine-learning app based on `requirements.txt`.

---

## Project Overview

- **Language:** Python
- **Key libraries:** `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `reportlab`, `joblib`
- **Typical use-case:** Interactive data app / ML model UI, likely served with Streamlit.

If there is a dedicated README (`README.md`) or docs directory, prefer that as the source of truth and keep this file in sync.

---

## Environment Setup

1. **Create and activate a virtual environment**

   PowerShell (Windows):

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **(Optional) Save new dependencies**

   When you add new packages:

   ```bash
   pip install <package>
   pip freeze > requirements.txt
   ```

---

## Common Commands

These are the commands you are most likely to run from the project root.

### Run the Streamlit app

```bash
streamlit run app.py
```

- If your main entrypoint file is not `app.py`, replace it with the correct filename (for example `main.py` or `src/app.py`).

### Ad-hoc scripts / experimentation

If you have notebooks or experiment scripts, they are usually in folders like `notebooks/` or `experiments/`. You can run a script with:

```bash
python path/to/script.py
```

---

## Testing

If you add tests, prefer `pytest`. Typical usage:

```bash
pytest
```

If you use the standard library `unittest`, you can instead run:

```bash
python -m unittest
```

Once a concrete test framework is configured (e.g. `pytest.ini`, `tests/` folder), update this section with the exact command.

---

## Linting & Formatting

This project does not advertise a specific linter/formatter in `requirements.txt`, but common choices are:

- **ruff** or **flake8** for linting
- **black** for formatting

Example setup (optional):

```bash
pip install ruff black
ruff check .
black .
```

If you add tooling config files (e.g. `pyproject.toml`, `.ruff.toml`, `setup.cfg`), reference the exact commands here.

---

## Project Structure (to fill in)

Fill this section with a short description of your layout once it stabilizes. For example:

- `app.py` – Streamlit entrypoint
- `models/` – Saved models and training code
- `data/` – Input datasets (gitignored if large/sensitive)
- `utils/` – Helper functions

---

## Working in Warp

### Useful Warp Workflows

- **Run app & watch logs:** open a Warp tab and run `streamlit run app.py`.
- **Edit code & rerun:** when Streamlit detects changes, it can auto-reload the app.
- **Run tests quickly:** have a dedicated tab running `pytest --maxfail=1 -q` (once tests exist).

### Recommended shell snippets

You can create reusable Warp workflows / snippets for:

- **Start app:** `streamlit run app.py`
- **Setup env:** `python -m venv .venv; .venv\\Scripts\\Activate.ps1; pip install -r requirements.txt`

---

## Maintenance Notes

- Keep `requirements.txt` in sync with the environment by running `pip freeze > requirements.txt` after installing/removing packages.
- Document any long-running or scheduled jobs (e.g., model training scripts) here once they exist.
- If you introduce Docker, CI, or deployment scripts, add the primary commands and links in this file.
