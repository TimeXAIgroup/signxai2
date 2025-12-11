# SignXAI2 Documentation setup (GitHub Pages + Sphinx)

This file was generated together with `.github/workflows/docs-pages.yml`.

## Expected layout

The workflow assumes a standard Sphinx layout:

    docs/
      source/
        conf.py
        index.rst (or index.md)
      _build/
        html/        # generated output

If your layout differs, update the `sphinx-build` line in
`.github/workflows/docs-pages.yml` accordingly.

### Recommended `conf.py` tweaks

In `docs/source/conf.py` you probably want something like:

```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

html_theme = "sphinx_rtd_theme"
html_baseurl = "https://timexaigroup.github.io/signxai2/"
```

Make sure you have at least:

```toml
# In pyproject.toml (optional)
[project.optional-dependencies]
docs = [
  "sphinx",
  "sphinx-rtd-theme",
  "myst-parser",
]
```

Then you can locally test the docs with:

```bash
pip install .[docs]
sphinx-build -b html docs/source docs/_build/html
```

## GitHub configuration

1. Commit and push this folder structure:
   - `.github/workflows/docs-pages.yml`
   - `docs/requirements-docs.txt` (optional helper for local builds)

2. In the repository on GitHub, go to:
   **Settings → Pages → Build and deployment → Source**
   and select **GitHub Actions**.

3. Push a change under `docs/` (or manually run the workflow via
   **Actions → Build and Deploy Sphinx Docs → Run workflow**).

Your docs will be available at:

    https://timexaigroup.github.io/signxai2/
```