#!/usr/bin/env bash
# Back up models as GitHub releases in SLADD_BACKUP_REPO. See docs/SERVER.md.
#
# Usage:
#   ./backup_model.sh               Every model in SLADD_WEIGHTS without a release.
#   ./backup_model.sh <dir> [tag]   One directory. The tag defaults to its name.

set -euo pipefail

if [[ -f "$(dirname "$0")/server.env" ]]; then
    # shellcheck source=server.env
    source "$(dirname "$0")/server.env"
fi

REPO=${SLADD_BACKUP_REPO:?Set SLADD_BACKUP_REPO in server.env}
# MiB, so each part stays below GitHub's limit of 2 GiB per release file.
PART_SIZE=1900M

work=
trap 'rm -rf "$work"' EXIT

backup() {
    [[ -d $1 ]] || { echo "ERROR: no such directory: $1" >&2; return 1; }
    # Absolute, because the restore command in the notes extracts into its parent.
    local dir
    dir=$(cd "$1" && pwd)
    local tag=${2:-$(basename "$dir")}

    # A draft means an upload stopped halfway, so it must not count as a backup.
    case $(gh release view "$tag" -R "$REPO" --json isDraft -q .isDraft 2>/dev/null || true) in
        false)
            echo "$tag: already backed up, skipped"
            return 0 ;;
        true)
            echo "ERROR: $tag has a draft release from an interrupted upload." >&2
            echo "       Delete it and run again: gh release delete $tag -R $REPO" >&2
            return 1 ;;
    esac

    local parent name
    parent=$(dirname "$dir")
    name=$(basename "$dir")
    # Next to the source, since /data2 has the room and /tmp may not.
    work=$(mktemp -d "$parent/.backup-XXXXXX")

    echo "$tag: packing $dir ($(du -sh "$dir" | cut -f1))"
    tar -cf - -C "$parent" "$name" | split -b "$PART_SIZE" -d -a 2 - "$work/$tag.tar.part-"
    (cd "$work" && sha256sum "$tag".tar.part-* > SHA256SUMS)

    local notes
    notes="Backup of \`$dir\` on $(hostname -s), $(date -u +%F).

Restore:

\`\`\`bash
gh release download $tag -R $REPO -D $tag
(cd $tag && sha256sum -c SHA256SUMS)
cat $tag/$tag.tar.part-* | tar -xf - -C $parent
\`\`\`"

    local parts=("$work/$tag".tar.part-*)
    echo "$tag: uploading ${#parts[@]} part(s)"
    gh release create "$tag" -R "$REPO" --draft --title "$tag" --notes "$notes" \
        "${parts[@]}" "$work/SHA256SUMS"
    gh release edit "$tag" -R "$REPO" --draft=false >/dev/null
    echo "$tag: done"

    rm -rf "$work"
    work=
}

# Fails before any packing when gh is not logged in or cannot see the repo.
gh repo view "$REPO" --json name >/dev/null

if (($# == 0)); then
    for dir in "${SLADD_WEIGHTS:?}"/*/; do
        backup "$dir"
    done
else
    backup "$1" "${2:-}"
fi
