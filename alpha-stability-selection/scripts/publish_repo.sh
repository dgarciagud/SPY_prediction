#!/usr/bin/env bash
# Publica esta carpeta como el repo privado independiente dgarciagud/alpha-stability-selection.
# Requiere la CLI `gh` autenticada (gh auth login) en tu máquina.
#
# Se ejecuta desde dentro de alpha-stability-selection/.
set -euo pipefail

REPO="alpha-stability-selection"
OWNER="dgarciagud"

if ! command -v gh >/dev/null 2>&1; then
  echo "Necesitas la CLI de GitHub (gh). Instálala y ejecuta 'gh auth login'." >&2
  exit 1
fi

# git init aislado en esta carpeta (historia propia, separada de spy_prediction)
git init
git add .
git commit -m "alpha-stability-selection: motor de selección por estabilidad"

# Crea el repo privado y empuja
gh repo create "${OWNER}/${REPO}" --private --source=. --remote=origin --push

echo "Listo: https://github.com/${OWNER}/${REPO}"
