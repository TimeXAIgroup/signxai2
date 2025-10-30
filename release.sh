set -e
[[ $# -eq 1 ]] || { echo "Usage: $0 <version>"; exit 1; }
ver="$1"
sed -i -E "s/^(version *= *\").*(\")$/\1$ver\2/" pyproject.toml
git stage --all
git commit -a -m "chore(release): v$ver"
git tag "v$ver"
git push origin main
git push origin "v$ver"
echo "Pushed v$ver — workflow will build and publish to PyPI."

