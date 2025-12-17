set -e

ver=$(grep -E '^version *= *"' pyproject.toml | sed -E 's/^version *= *"([^"]+)".*/\1/')
[[ -n "$ver" ]] || { echo "Could not read version from pyproject.toml"; exit 1; }

git stage --all
git commit -a -m "release: v$ver"
git tag "v$ver"
git push origin main
git push origin "v$ver"

echo "Pushed v$ver — workflow will build and publish to PyPI."