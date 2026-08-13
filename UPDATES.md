# What was added to OmniSeg-Audio-Pipeline

All files are **additive** — none of your existing Python code, configs, assets
or requirements were modified.

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Code-quality CI. Since the pipeline needs a GPU + multi-GB models, CI does **not** run inference. Instead it compiles every module and runs a critical-error lint (undefined names, syntax) on Python 3.11 & 3.12. Fast, always-green on valid code. |
| `LICENSE` | MIT license (matches the README badge). |
| `pyproject.toml` | Ruff config: excludes the vendored `sam2_source.py` snippet and pins the critical-lint ruleset. |
| `README.md` | Polished, badge-topped README (keeps your real showcase images). |

## Recommended cleanup (optional)

- **Delete `sam2_source.py`** — it's a copied snippet of Meta's internal SAM 2
  code, not part of your project. Removing it keeps the repo clean and avoids
  mixing third-party code into your own. The CI already excludes it either way.

## Push it

```powershell
git add -A
git commit -m "Add CI, LICENSE, ruff config, and polished README"
git push
```
