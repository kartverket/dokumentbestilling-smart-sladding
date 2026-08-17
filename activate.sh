#!/usr/bin/env bash
# activate.sh — aktiver venv + last inn server-variabler i én kommando
#
# Bruk (merk: source, ikke kjør):
#   source activate.sh
#
# Kan IKKE kjøres som ./activate.sh — venv aktiveres da i en subshell
# og forsvinner når skriptet er ferdig.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) Aktiver venv
source "$_SCRIPT_DIR/venv/bin/activate"

# 2) Last inn server-variabler
source "$_SCRIPT_DIR/server.env"

echo "✓ venv aktivert + server-variabler lastet"
echo "  SLADD_REPO=$SLADD_REPO"

unset _SCRIPT_DIR

