#!/usr/bin/env bash
# activate.sh: venv + server variables in one command. Must be sourced ("source activate.sh"),
# because running it activates the venv in a subshell that dies with the script.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$_SCRIPT_DIR/venv/bin/activate"
source "$_SCRIPT_DIR/server.env"

# The repo's git hooks include the fnr guard, so turn them on if they are not already.
if [ "$(git -C "$_SCRIPT_DIR" config --get core.hooksPath)" != ".githooks" ]; then
    git -C "$_SCRIPT_DIR" config core.hooksPath .githooks
    echo "✓ git hooks enabled (.githooks)"
fi

echo "✓ venv activated + server variables loaded"
echo "  SLADD_REPO=$SLADD_REPO"

unset _SCRIPT_DIR
