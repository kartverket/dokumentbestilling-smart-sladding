#!/usr/bin/env bash
# Build, test and promote the app image on the GPU server.
#
# An image is built once and gets an immutable tag. After that, only *which*
# tag runs where changes. Prod never builds, and rollback means pointing back
# at a tag that has already run. Images exist only on this server, so a tag
# you delete is gone for good. See "prune".
#
# Usage:
#   ./deploy.sh build [tag] [weights=PATH]
#                                Build an image. Weights default to
#                                SLADD_PRODWEIGHTS; the tag is derived from
#                                git and the model name.
#   ./deploy.sh test <tag>       Run the tag on the test port (5072), health check.
#   ./deploy.sh promote <tag>    Put the tag in prod (5071). Rolls back on failure.
#   ./deploy.sh rollback         Back to the previous tag that ran in prod.
#   ./deploy.sh status           What runs where, and is it healthy.
#   ./deploy.sh versions         Locally built tags, newest first.
#   ./deploy.sh start prod|test  Start again with the tag recorded in .env.
#   ./deploy.sh stop prod|test   Stop the container and free the GPU memory.
#   ./deploy.sh logs prod|test   Follow the log.
#   ./deploy.sh prune [keep]     Delete old images. Cannot be undone.

set -euo pipefail
cd "$(dirname "$0")"

# SLADD_LOGS maps to LOG_ROOT because that is the name compose mounts by.
if [[ -f server.env ]]; then
    # shellcheck source=server.env
    source ./server.env
    [[ -n ${SLADD_LOGS:-} ]]     && export LOG_ROOT="$SLADD_LOGS"
    [[ -n ${SLADD_LOG_DAYS:-} ]] && export LOG_BACKUP_DAYS="$SLADD_LOG_DAYS"
fi

IMAGE=${IMAGE:-smart-sladding}
ENV_FILE=.env
HISTORY=.deploy-historikk
PROD_PORT=5071

# The weights live outside the build context, so they are staged here. Path is fixed by the Dockerfile.
STAGING=.byggvekter

LABEL_MODEL=no.kartverket.smsl.modell
LABEL_MODEL_SHA=no.kartverket.smsl.modell.sha256
LABEL_MODEL_SRC=no.kartverket.smsl.modell.kilde

# ── helpers ──────────────────────────────────────────────────────────

die() { echo "ERROR: $*" >&2; exit 1; }

get_env() {
    [[ -f $ENV_FILE ]] || return 0
    # "|| true" keeps a missing key from killing the caller under set -e + pipefail.
    { grep "^${1}=" "$ENV_FILE" 2>/dev/null || true; } | tail -1 | cut -d= -f2-
}

set_env() {
    local key=$1 value=$2 tmp
    touch "$ENV_FILE"
    tmp=$(mktemp)
    if grep -q "^${key}=" "$ENV_FILE"; then
        sed "s|^${key}=.*|${key}=${value}|" "$ENV_FILE" > "$tmp"
    else
        cat "$ENV_FILE" > "$tmp"
        echo "${key}=${value}" >> "$tmp"
    fi
    mv "$tmp" "$ENV_FILE"
}

# date + commit + model: the commit alone would not distinguish two models built from the same code.
auto_tag() {
    local model=$1 sha suffix=""
    sha=$(git rev-parse --short=8 HEAD 2>/dev/null) || die "not a git repo. Give the tag manually"
    git diff --quiet && git diff --cached --quiet || suffix="-dirty"
    echo "$(date +%Y%m%d)-${sha}-$(tag_slug "$model")${suffix}"
}

# Docker tags only accept [a-zA-Z0-9._-]; model names come from arbitrary directory names.
tag_slug() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed 's/[^a-z0-9._-]/-/g; s/--*/-/g; s/^[-._]*//; s/[-._]*$//' \
        | cut -c1-40
}

# sha256sum on Linux, shasum on macOS.
file_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d" " -f1
    else
        shasum -a 256 "$1" | cut -d" " -f1
    fi
}

# Empty for images built before the labels existed.
image_model() {
    docker image inspect "${IMAGE}:${1}" \
        --format "{{index .Config.Labels \"${LABEL_MODEL}\"}}" 2>/dev/null || true
}

# Published models are <name>/<name>.pt; raw runs are <run>/weights/best.pt, where only the run dir names it.
model_name_from_path() {
    local path=$1 name
    name=$(basename "$path"); name=${name%.pt}
    if [[ $name == best || $name == last ]]; then
        name=$(basename "$(dirname "$(dirname "$path")")")
    fi
    echo "$name"
}

image_exists() {
    docker image inspect "${IMAGE}:${1}" >/dev/null 2>&1
}

require_image() {
    image_exists "$1" \
        || die "cannot find ${IMAGE}:${1}. Images exist only locally, so it must be built: ./deploy.sh build ${1}"
}

# "prod" or "test". Anything else is a typo we must not guess at.
require_target() {
    case ${1:-} in
        prod|test) return 0 ;;
        '') die "give a target: \"prod\" or \"test\"" ;;
        *) die "unknown target \"${1}\". Use \"prod\" or \"test\"" ;;
    esac
}

# The test service sits behind a compose profile; prod has none.
profile_for() {
    [[ $1 == test ]] && echo "--profile test" || true
}

# The port a target is exposed on. 5071 is prod and only prod.
port_for() {
    if [[ $1 == prod ]]; then
        echo "$PROD_PORT"
    else
        local p; p=$(get_env TEST_PORT); p=${p:-5072}
        [[ $p == "$PROD_PORT" ]] && die "TEST_PORT is set to ${PROD_PORT}, which is the prod port"
        echo "$p"
    fi
}

# The tag .env says should run for a target.
tag_for() {
    if [[ $1 == prod ]]; then get_env PROD_TAG; else get_env TEST_TAG; fi
}

# Models load on the first /model call, so a healthy container answers within seconds.
health_check() {
    local port=$1 tries=${2:-45} i
    for (( i=1; i<=tries; i++ )); do
        if curl -fsS --max-time 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  /health answers on port ${port} (after ${i} tries)"
            return 0
        fi
        sleep 2
    done
    echo "  /health did not answer on port ${port} within $(( tries * 2 )) seconds" >&2
    return 1
}

# ── commands ─────────────────────────────────────────────────────────

cmd_build() {
    local tag="" weights="" arg
    for arg in "$@"; do
        case $arg in
            weights=*) weights=${arg#weights=} ;;
            -*)  die "unknown flag \"${arg}\". Usage: ./deploy.sh build [tag] [weights=PATH]" ;;
            *)   [[ -z $tag ]] || die "too many arguments: \"${arg}\""
                 tag=$arg ;;
        esac
    done

    local proxy=${PROXY:-http://159.162.48.7:3128}

    # Without a model the image starts fine and fails on the first /model call.
    weights=${weights:-${SLADD_PRODWEIGHTS:-}}
    [[ -n $weights ]] \
        || die "no weights chosen. Set SLADD_PRODWEIGHTS in server.env, or pass them: ./deploy.sh build weights=\$SLADD_WEIGHTS/<model>/<model>.pt"
    [[ -f $weights ]] \
        || die "cannot find the weights file \"${weights}\". See what is published: ls \$SLADD_WEIGHTS"

    local bundle name sha
    bundle=$(cd "$(dirname "$weights")" && pwd)
    name=$(model_name_from_path "$weights")
    sha=$(file_sha256 "$weights")
    tag=${tag:-$(auto_tag "$name")}

    if image_exists "$tag"; then
        echo "${IMAGE}:${tag} already exists. Rebuilding and overwriting the tag."
    fi

    echo "Model: ${name}"
    echo "  file:   ${weights}"
    echo "  sha256: ${sha:0:16}…"

    # Cleaned up however the build ends, so it is never mistaken for a model store.
    rm -rf "$STAGING"
    mkdir -p "$STAGING"
    trap 'rm -rf "$STAGING"' EXIT
    cp "$weights" "$STAGING/modell.pt"
    if [[ -f $bundle/modell.json ]]; then
        cp "$bundle/modell.json" "$STAGING/modell.json"
    else
        echo "  WARNING: no modell.json next to the weights file. The image will not"
        echo "           know what the model was trained on. Publish the model with"
        echo "           \"make -C \$SLADD_TRAIN publiser\" instead of copying the .pt file."
    fi

    echo
    echo "Building ${IMAGE}:${tag} ..."
    docker build \
        --build-arg HTTP_PROXY="$proxy" \
        --build-arg HTTPS_PROXY="$proxy" \
        --label org.opencontainers.image.revision="$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
        --label org.opencontainers.image.version="$tag" \
        --label "${LABEL_MODEL}=${name}" \
        --label "${LABEL_MODEL_SHA}=${sha}" \
        --label "${LABEL_MODEL_SRC}=${weights}" \
        -t "${IMAGE}:${tag}" .

    echo
    echo "Done: ${IMAGE}:${tag}  (model ${name})"
    echo "Test it:  ./deploy.sh test ${tag}"
}

cmd_test() {
    local tag=${1:-}
    [[ -n $tag ]] || die "give a tag: ./deploy.sh test <tag>   (see ./deploy.sh versions)"
    require_image "$tag"

    local port; port=$(port_for test)

    set_env TEST_TAG "$tag"
    echo "Starting ${IMAGE}:${tag} on port ${port} ..."
    docker compose --profile test up -d --force-recreate test

    if health_check "$port"; then
        echo
        echo "Test runs on http://localhost:${port}"
        echo "  curl -X POST http://localhost:${port}/model -H 'Content-Type: application/pdf' --data-binary @document.pdf"
        echo "Happy with it?  ./deploy.sh promote ${tag}"
        echo "Stop the test:  ./deploy.sh stop test"
    else
        echo
        echo "See what went wrong: ./deploy.sh logs test" >&2
        exit 1
    fi
}

cmd_promote() {
    local tag=${1:-}
    [[ -n $tag ]] || die "give the tag explicitly: ./deploy.sh promote <tag>   (see ./deploy.sh versions)"
    require_image "$tag"

    if [[ $tag == *-dirty ]]; then
        die "\"${tag}\" was built on uncommitted changes and must not go to prod. Commit, rebuild, and promote that tag."
    fi

    local previous; previous=$(get_env PROD_TAG)
    echo "Prod (port ${PROD_PORT}): ${previous:-nothing}  ->  ${tag}"
    echo "Model:                   $(image_model "$previous")  ->  $(image_model "$tag")"
    read -r -p "Continue? [j/N] " answer
    case $answer in
        j|J|yes|Ja|JA) ;;
        *) echo "Aborted."; exit 0 ;;
    esac

    set_env PROD_TAG "$tag"
    docker compose up -d --force-recreate prod

    if health_check "$PROD_PORT"; then
        printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$tag" "${previous:--}" >> "$HISTORY"
        echo
        echo "${tag} is in prod on port ${PROD_PORT}."
        [[ -n $previous ]] && echo "Back to ${previous}:  ./deploy.sh rollback"
    else
        echo >&2
        if [[ -n $previous ]] && image_exists "$previous"; then
            echo "Rolling back to ${previous} ..." >&2
            set_env PROD_TAG "$previous"
            docker compose up -d --force-recreate prod
            health_check "$PROD_PORT" || echo "WARNING: ${previous} does not answer either. Prod is down." >&2
        else
            echo "No previous tag to roll back to. Prod is down." >&2
        fi
        exit 1
    fi
}

cmd_rollback() {
    [[ -s $HISTORY ]] || die "no deploy history in ${HISTORY}"
    local previous; previous=$(tail -1 "$HISTORY" | cut -f3)
    [[ -n $previous && $previous != "-" ]] || die "the previous deploy had no tag to go back to"
    image_exists "$previous" \
        || die "${previous} is in the history, but the image no longer exists locally. Rebuild it from the commit the tag names."
    echo "Rolling prod back to ${previous}."
    cmd_promote "$previous"
}

cmd_status() {
    local prod_tag test_tag
    prod_tag=$(get_env PROD_TAG || true)
    test_tag=$(get_env TEST_TAG || true)

    echo "Image:   ${IMAGE}  (local to this server only)"
    echo "Logs:    ${LOG_ROOT:-/data/docker}  (${LOG_BACKUP_DAYS:-30} days of history)"
    echo "Weights: ${SLADD_WEIGHTS:-(not set)}  (default model: ${SLADD_PRODWEIGHTS:-none})"
    echo "Prod  (port ${PROD_PORT}):  ${prod_tag}  model: $(image_model "$prod_tag")"
    echo "Test  (port $(port_for test)):  ${test_tag}  model: $(image_model "$test_tag")"
    echo
    docker compose --profile test ps -a --format 'table {{.Name}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
    echo
    echo "Last deploys to prod (time, tag, previous tag):"
    tail -5 "$HISTORY" 2>/dev/null || echo "  (none)"
}

# The tag shortens the model name, so the image label is read per tag as the authority.
cmd_versions() {
    local tag created size
    printf '%-46s %-16s %-9s %s\n' TAG AGE SIZE MODEL
    while IFS=$'\t' read -r tag created size; do
        [[ -z $tag || $tag == "<none>" ]] && continue
        printf '%-46s %-16s %-9s %s\n' \
            "$tag" "$created" "$size" "$(image_model "$tag")"
    done < <(docker images "$IMAGE" \
                --format '{{.Tag}}\t{{.CreatedSince}}\t{{.Size}}' \
                --filter 'dangling=false')
}

cmd_logs() {
    local target=${1:-prod}; require_target "$target"
    local -a profile=(); read -r -a profile <<< "$(profile_for "$target")"
    docker compose ${profile[@]+"${profile[@]}"} logs -f "$target"
}

cmd_start() {
    local target=${1:-}; require_target "$target"
    local -a profile=(); read -r -a profile <<< "$(profile_for "$target")"

    # Starts what .env points at; switching version is "promote" or "test <tag>".
    local tag; tag=$(tag_for "$target")
    [[ -n $tag && $tag != "ikke-satt" ]] \
        || die "no tag set for ${target}. Run \"./deploy.sh $( [[ $target == prod ]] && echo 'promote' || echo 'test' ) <tag>\" first."
    require_image "$tag"

    local port; port=$(port_for "$target")
    echo "Starting ${target} (${IMAGE}:${tag}) on port ${port} ..."
    docker compose ${profile[@]+"${profile[@]}"} up -d "$target"

    health_check "$port" || { echo "See what went wrong: ./deploy.sh logs ${target}" >&2; exit 1; }
}

cmd_stop() {
    local target=${1:-}; require_target "$target"
    local -a profile=(); read -r -a profile <<< "$(profile_for "$target")"

    # Stopping prod takes production down. Test can be stopped without asking.
    if [[ $target == prod ]]; then
        echo "This takes production down on port ${PROD_PORT}."
        read -r -p "Continue? [j/N] " answer
        case $answer in
            j|J|yes|Ja|JA) ;;
            *) echo "Aborted."; return 0 ;;
        esac
    fi

    # "stop", not "rm", so "start" brings up exactly the same setup again.
    docker compose ${profile[@]+"${profile[@]}"} stop "$target"
    echo "${target} is stopped. Start again with: ./deploy.sh start ${target}"
}

# Deletion is irreversible, so the N newest, prod, test and everything in the history are protected.
cmd_prune() {
    local keep=${1:-5}
    local prod_tag; prod_tag=$(get_env PROD_TAG)
    local test_tag; test_tag=$(get_env TEST_TAG)
    local historic=""
    [[ -f $HISTORY ]] && historic=$(cut -f2,3 "$HISTORY" | tr '\t' '\n' | sort -u)

    local -a doomed=()
    while read -r tag; do
        [[ -z $tag || $tag == "<none>" ]] && continue
        [[ $tag == "$prod_tag" || $tag == "$test_tag" ]] && continue
        grep -qxF "$tag" <<< "$historic" && continue
        doomed+=("$tag")
    done < <(docker images "$IMAGE" --format '{{.Tag}}' | tail -n "+$(( keep + 1 ))")

    if [[ ${#doomed[@]} -eq 0 ]]; then
        echo "Nothing to clean up (protecting the ${keep} newest, prod, test and everything in the history)."
        return 0
    fi

    echo "Deleting ${#doomed[@]} image(s) permanently:"
    printf '  %s\n' "${doomed[@]}"
    echo
    echo "The images exist only locally. The only way back is to rebuild from"
    echo "the commit the tag names."
    read -r -p "Continue? [j/N] " answer
    case $answer in
        j|J|yes|Ja|JA) ;;
        *) echo "Aborted."; return 0 ;;
    esac
    for tag in "${doomed[@]}"; do docker rmi "${IMAGE}:${tag}" || true; done
}

# ── routing ──────────────────────────────────────────────────────────

cmd=${1:-status}
shift || true
case $cmd in
    build)      cmd_build "$@" ;;
    test)       cmd_test "$@" ;;
    promote)    cmd_promote "$@" ;;
    rollback)   cmd_rollback ;;
    status)     cmd_status ;;
    versions)   cmd_versions ;;
    start)      cmd_start "$@" ;;
    stop)       cmd_stop "$@" ;;
    logs)       cmd_logs "$@" ;;
    prune)      cmd_prune "$@" ;;
    -h|--help|help)
        awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0" ;;
    *) die "unknown command \"${cmd}\". Run ./deploy.sh --help" ;;
esac
