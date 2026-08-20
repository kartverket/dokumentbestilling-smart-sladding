#!/usr/bin/env bash
# Bygg, test og promoter app-imaget på GPU-serveren.
#
# Prinsippet: et image bygges én gang og får en uforanderlig tag. Etter
# det flyttes bare *hvilken* tag som kjører hvor. Prod bygger aldri, så
# det som står på port 5071 endrer seg ikke av at noen bygger noe nytt,
# og rollback er å peke tilbake på en tag som allerede har kjørt.
#
# Imagene lagres bare lokalt på serveren. Det betyr at en tag du sletter
# er borte for godt — se «prune» nedenfor.
#
# Bruk:
#   ./deploy.sh build [tag]      Bygg image. Tag utledes fra git hvis den utelates.
#   ./deploy.sh test <tag>       Start taggen på testporten (5072) og helsesjekk.
#   ./deploy.sh promote <tag>    Sett taggen i prod (5071). Ruller tilbake ved feil.
#   ./deploy.sh rollback         Tilbake til forrige tag som kjørte i prod.
#   ./deploy.sh status           Hva kjører hvor, og er det friskt.
#   ./deploy.sh versions         Lokalt bygde tagger, nyest først.
#   ./deploy.sh start prod|test  Start igjen med taggen som står i .env.
#   ./deploy.sh stop prod|test   Stopp containeren og frigi GPU-minnet.
#   ./deploy.sh logs prod|test   Følg loggen.
#   ./deploy.sh prune [antall]   Slett gamle images. Kan ikke angres.

set -euo pipefail
cd "$(dirname "$0")"

# server.env eier serverstiene, inkludert hvor loggene skal ligge.
# Vi mapper SLADD_LOGS -> LOG_ROOT, som er navnet compose bruker i
# volume-monteringene. Finnes ikke filen, faller compose tilbake på
# /data/docker.
if [[ -f server.env ]]; then
    # shellcheck source=server.env
    source ./server.env
    [[ -n ${SLADD_LOGS:-} ]]      && export LOG_ROOT="$SLADD_LOGS"
    [[ -n ${SLADD_LOGG_DAGER:-} ]] && export LOG_BACKUP_DAYS="$SLADD_LOGG_DAGER"
fi

IMAGE=${IMAGE:-smart-sladding}
ENV_FIL=.env
HISTORIKK=.deploy-historikk
PROD_PORT=5071

# ── hjelpere ─────────────────────────────────────────────────────────

feil() { echo "FEIL: $*" >&2; exit 1; }

hent_env() {
    [[ -f $ENV_FIL ]] || return 0
    # grep-en må ikke få velte kalleren: uten «|| true» gjør pipefail en
    # manglende nøkkel til exit 1, og «x=$(hent_env NOKKEL)» dreper da
    # skriptet under set -e før feilmeldingen vår rekker å bli skrevet.
    { grep "^${1}=" "$ENV_FIL" 2>/dev/null || true; } | tail -1 | cut -d= -f2-
}

sett_env() {
    local nokkel=$1 verdi=$2 tmp
    touch "$ENV_FIL"
    tmp=$(mktemp)
    if grep -q "^${nokkel}=" "$ENV_FIL"; then
        sed "s|^${nokkel}=.*|${nokkel}=${verdi}|" "$ENV_FIL" > "$tmp"
    else
        cat "$ENV_FIL" > "$tmp"
        echo "${nokkel}=${verdi}" >> "$tmp"
    fi
    mv "$tmp" "$ENV_FIL"
}

# Sorterbar og sporbar: dato + commit. "-dirty" hvis treet ikke er rent,
# så et image bygget på ucommittede endringer aldri kan forveksles med
# en commit — og aldri havner i prod ved et uhell (se cmd_promote).
auto_tag() {
    local sha suffiks=""
    sha=$(git rev-parse --short=8 HEAD 2>/dev/null) || feil "ikke et git-repo — oppgi tag manuelt"
    git diff --quiet && git diff --cached --quiet || suffiks="-dirty"
    echo "$(date +%Y%m%d)-${sha}${suffiks}"
}

image_finnes() {
    docker image inspect "${IMAGE}:${1}" >/dev/null 2>&1
}

krev_image() {
    image_finnes "$1" \
        || feil "finner ikke ${IMAGE}:${1}. Imagene finnes bare lokalt, så den må bygges: ./deploy.sh build ${1}"
}

# «prod» eller «test» — alt annet er en skrivefeil vi ikke skal gjette på.
krev_mal() {
    case ${1:-} in
        prod|test) return 0 ;;
        '') feil "oppgi mål: «prod» eller «test»" ;;
        *) feil "ukjent mål «${1}». Bruk «prod» eller «test»" ;;
    esac
}

# Test-tjenesten ligger bak en compose-profil; prod har ingen.
profil_for() {
    [[ $1 == test ]] && echo "--profile test" || true
}

# Porten et mål eksponeres på. 5071 er prod og bare prod.
port_for() {
    if [[ $1 == prod ]]; then
        echo "$PROD_PORT"
    else
        local p; p=$(hent_env TEST_PORT); p=${p:-5072}
        [[ $p == "$PROD_PORT" ]] && feil "TEST_PORT er satt til ${PROD_PORT}, som er prod-porten"
        echo "$p"
    fi
}

# Taggen .env sier skal kjøre for et mål.
tag_for() {
    if [[ $1 == prod ]]; then hent_env PROD_TAG; else hent_env TEST_TAG; fi
}

# Venter til /health svarer. Modellene lastes først ved første /model-kall,
# så en frisk container svarer normalt innen sekunder.
helsesjekk() {
    local port=$1 forsok=${2:-45} i
    for (( i=1; i<=forsok; i++ )); do
        if curl -fsS --max-time 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
            echo "  /health svarer på port ${port} (etter ${i} forsøk)"
            return 0
        fi
        sleep 2
    done
    echo "  /health svarte ikke på port ${port} innen $(( forsok * 2 )) sekunder" >&2
    return 1
}

# ── kommandoer ───────────────────────────────────────────────────────

cmd_build() {
    local tag=${1:-$(auto_tag)}
    local proxy=${PROXY:-http://159.162.48.7:3128}

    # Submodulet app/weights har best.pt. Bygger vi uten det, får vi et
    # image som starter fint og feiler først ved første /model-kall.
    [[ -f app/weights/best.pt ]] \
        || feil "app/weights/best.pt mangler. Hent modell-submodulet: git submodule update --init app/weights"

    if image_finnes "$tag"; then
        echo "${IMAGE}:${tag} finnes allerede. Bygger på nytt og overskriver taggen."
    fi

    echo "Bygger ${IMAGE}:${tag} ..."
    docker build \
        --build-arg HTTP_PROXY="$proxy" \
        --build-arg HTTPS_PROXY="$proxy" \
        --label org.opencontainers.image.revision="$(git rev-parse HEAD 2>/dev/null || echo ukjent)" \
        --label org.opencontainers.image.version="$tag" \
        -t "${IMAGE}:${tag}" .

    echo
    echo "Ferdig: ${IMAGE}:${tag}"
    echo "Test den:      ./deploy.sh test ${tag}"
}

cmd_test() {
    local tag=${1:-}
    [[ -n $tag ]] || feil "oppgi tag: ./deploy.sh test <tag>   (se ./deploy.sh versions)"
    krev_image "$tag"

    local port; port=$(port_for test)

    sett_env TEST_TAG "$tag"
    echo "Starter ${IMAGE}:${tag} på port ${port} ..."
    docker compose --profile test up -d --force-recreate test

    if helsesjekk "$port"; then
        echo
        echo "Test kjører på http://localhost:${port}"
        echo "  curl -X POST http://localhost:${port}/model -H 'Content-Type: application/pdf' --data-binary @dokument.pdf"
        echo "Fornøyd?       ./deploy.sh promote ${tag}"
        echo "Stopp testen:  ./deploy.sh stop test"
    else
        echo
        echo "Se hva som gikk galt: ./deploy.sh logs test" >&2
        exit 1
    fi
}

cmd_promote() {
    local tag=${1:-}
    [[ -n $tag ]] || feil "oppgi tag eksplisitt: ./deploy.sh promote <tag>   (se ./deploy.sh versions)"
    krev_image "$tag"

    if [[ $tag == *-dirty ]]; then
        feil "«${tag}» er bygget på ucommittede endringer og skal ikke i prod. Commit, bygg på nytt, og promoter den taggen."
    fi

    local forrige; forrige=$(hent_env PROD_TAG)
    echo "Prod (port ${PROD_PORT}): ${forrige:-ingenting}  ->  ${tag}"
    read -r -p "Fortsette? [j/N] " svar
    case $svar in
        j|J|ja|Ja|JA) ;;
        *) echo "Avbrutt."; exit 0 ;;
    esac

    sett_env PROD_TAG "$tag"
    docker compose up -d --force-recreate prod

    if helsesjekk "$PROD_PORT"; then
        printf '%s\t%s\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)" "$tag" "${forrige:--}" >> "$HISTORIKK"
        echo
        echo "${tag} er i prod på port ${PROD_PORT}."
        [[ -n $forrige ]] && echo "Tilbake til ${forrige}:  ./deploy.sh rollback"
    else
        echo >&2
        if [[ -n $forrige ]] && image_finnes "$forrige"; then
            echo "Ruller tilbake til ${forrige} ..." >&2
            sett_env PROD_TAG "$forrige"
            docker compose up -d --force-recreate prod
            helsesjekk "$PROD_PORT" || echo "ADVARSEL: ${forrige} svarer heller ikke. Prod er nede." >&2
        else
            echo "Ingen forrige tag å rulle tilbake til. Prod er nede." >&2
        fi
        exit 1
    fi
}

cmd_rollback() {
    [[ -s $HISTORIKK ]] || feil "ingen deploy-historikk i ${HISTORIKK}"
    local forrige; forrige=$(tail -1 "$HISTORIKK" | cut -f3)
    [[ -n $forrige && $forrige != "-" ]] || feil "forrige deploy hadde ingen tag å gå tilbake til"
    image_finnes "$forrige" \
        || feil "${forrige} står i historikken, men imaget finnes ikke lenger lokalt. Bygg det på nytt fra commiten taggen navngir."
    echo "Ruller prod tilbake til ${forrige}."
    cmd_promote "$forrige"
}

cmd_status() {
    echo "Image: ${IMAGE}  (bare lokalt på denne serveren)"
    echo "Logger: ${LOG_ROOT:-/data/docker}  (${LOG_BACKUP_DAYS:-30} døgn historikk)"
    echo "Prod  (port ${PROD_PORT}):  $(hent_env PROD_TAG || true)"
    echo "Test  (port $(port_for test)):  $(hent_env TEST_TAG || true)"
    echo
    docker compose --profile test ps -a --format 'table {{.Name}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
    echo
    echo "Siste deployer til prod (tid, tag, forrige tag):"
    tail -5 "$HISTORIKK" 2>/dev/null || echo "  (ingen)"
}

cmd_versions() {
    docker images "$IMAGE" \
        --format 'table {{.Tag}}\t{{.CreatedSince}}\t{{.Size}}' \
        --filter 'dangling=false'
}

cmd_logs() {
    local mal=${1:-prod}; krev_mal "$mal"
    local -a profil=(); read -r -a profil <<< "$(profil_for "$mal")"
    docker compose ${profil[@]+"${profil[@]}"} logs -f "$mal"
}

cmd_start() {
    local mal=${1:-}; krev_mal "$mal"
    local -a profil=(); read -r -a profil <<< "$(profil_for "$mal")"

    # Starter det .env allerede peker på — bytter aldri versjon. Skal du
    # bytte, er det «promote» (prod) eller «test <tag>» (test).
    local tag; tag=$(tag_for "$mal")
    [[ -n $tag && $tag != "ikke-satt" ]] \
        || feil "ingen tag satt for ${mal}. Kjør «./deploy.sh $( [[ $mal == prod ]] && echo 'promote' || echo 'test' ) <tag>» først."
    krev_image "$tag"

    local port; port=$(port_for "$mal")
    echo "Starter ${mal} (${IMAGE}:${tag}) på port ${port} ..."
    docker compose ${profil[@]+"${profil[@]}"} up -d "$mal"

    helsesjekk "$port" || { echo "Se hva som gikk galt: ./deploy.sh logs ${mal}" >&2; exit 1; }
}

cmd_stop() {
    local mal=${1:-}; krev_mal "$mal"
    local -a profil=(); read -r -a profil <<< "$(profil_for "$mal")"

    # Å stoppe prod er å ta ned produksjon. Test kan stoppes uten spørsmål.
    if [[ $mal == prod ]]; then
        echo "Dette tar ned produksjon på port ${PROD_PORT}."
        read -r -p "Fortsette? [j/N] " svar
        case $svar in
            j|J|ja|Ja|JA) ;;
            *) echo "Avbrutt."; return 0 ;;
        esac
    fi

    # «stop», ikke «rm»: containeren blir stående, så «start» henter opp
    # nøyaktig samme oppsett igjen. GPU-minnet frigis uansett, siden
    # prosessen dør. restart-policyen tar den ikke opp av seg selv etter
    # en eksplisitt stopp.
    docker compose ${profil[@]+"${profil[@]}"} stop "$mal"
    echo "${mal} er stoppet. Start igjen med: ./deploy.sh start ${mal}"
}

# Imagene er store (titalls GB), så disken fylles fort. Men de finnes
# bare her: sletter du en tag, er den eneste veien tilbake å bygge den
# på nytt fra commiten taggen navngir. Vi verner de N nyeste, taggen i
# prod, taggen i test, og alt som står i deploy-historikken.
cmd_prune() {
    local behold=${1:-5}
    local prod_tag; prod_tag=$(hent_env PROD_TAG)
    local test_tag; test_tag=$(hent_env TEST_TAG)
    local historiske=""
    [[ -f $HISTORIKK ]] && historiske=$(cut -f2,3 "$HISTORIKK" | tr '\t' '\n' | sort -u)

    local -a slett=()
    while read -r tag; do
        [[ -z $tag || $tag == "<none>" ]] && continue
        [[ $tag == "$prod_tag" || $tag == "$test_tag" ]] && continue
        grep -qxF "$tag" <<< "$historiske" && continue
        slett+=("$tag")
    done < <(docker images "$IMAGE" --format '{{.Tag}}' | tail -n "+$(( behold + 1 ))")

    if [[ ${#slett[@]} -eq 0 ]]; then
        echo "Ingenting å rydde (verner de ${behold} nyeste, prod, test og alt i historikken)."
        return 0
    fi

    echo "Sletter ${#slett[@]} image(r) permanent:"
    printf '  %s\n' "${slett[@]}"
    echo
    echo "Imagene finnes bare lokalt. Eneste vei tilbake er å bygge på nytt"
    echo "fra commiten taggen navngir."
    read -r -p "Fortsette? [j/N] " svar
    case $svar in
        j|J|ja|Ja|JA) ;;
        *) echo "Avbrutt."; return 0 ;;
    esac
    for tag in "${slett[@]}"; do docker rmi "${IMAGE}:${tag}" || true; done
}

# ── ruting ───────────────────────────────────────────────────────────

kmd=${1:-status}
shift || true
case $kmd in
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
    *) feil "ukjent kommando «${kmd}». Kjør ./deploy.sh --help" ;;
esac
