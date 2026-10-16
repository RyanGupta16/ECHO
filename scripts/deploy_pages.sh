#!/bin/zsh
# Build the static web demo and publish it to the gh-pages branch (served by GitHub Pages).
#   scripts/deploy_pages.sh
set -e
cd "$(dirname "$0")/.."
REPO=$(basename -s .git "$(git remote get-url origin)")
URL=$(git remote get-url origin)
(cd web && NEXT_PUBLIC_BASE_PATH=/$REPO npm run build)
touch web/out/.nojekyll
TMP=$(mktemp -d)
cp -R web/out/. "$TMP"
cd "$TMP"
git init -q -b gh-pages
git add -A
git commit -q -m "Deploy live demo"
git push -f "$URL" gh-pages
rm -rf "$TMP"
echo "deployed → https://$(git -C "$OLDPWD" remote get-url origin | sed -E 's#.*github.com[:/]([^/]+)/.*#\1#' | tr A-Z a-z).github.io/$REPO/"
