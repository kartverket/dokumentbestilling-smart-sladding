"""Maskineri for å kjøre tunge dokumentjobber parallelt mot én GPU.

Hovedprosessen koordinerer bare, og alt arbeid skjer i uavhengige prosesser
som henter filer fra en delt teller. Grunnen er målt på V100S + Xeon 6230:
~77 % av tiden i én pipeline gikk til enkelttrådet CPU-forbehandling
(normalisering, bildekopier) og bare ~11 % til GPU-arbeid, så kortet sto
stille halve tiden. Fire prosesser mot samme kort ga 3,3× gjennomstrømning
og mettet GPU-en (228 av 250 W).

Minnestyringen følger av det: hver prosess får sitt eget tak på kortet
(FLAGS_gpu_memory_limit_mb), batchstørrelsen finner taket selv med
slow-start og halvering ved minnefeil, og en minnefeil koster en omkjøring
av de samme dokumentene — aldri et tapt dokument.

Bruk fra et verktøy:

    import parallell_pipeline as pp

    def lag_behandler(oppgave):
        # Kjører i arbeidsprosessen. Importer tunge moduler HER.
        from paddle_ocr_model_fnr import les_tokens_batched
        tider = oppgave["tider"]

        def behandle(gruppe, sider_om_gangen=None):
            # gruppe: [(navn, [bilder]), ...]
            # returner én dict per dokument: {"navn", "sider", "tekst"}
            ...
        return behandle

    p = argparse.ArgumentParser()
    pp.legg_til_argumenter(p)
    args = p.parse_args()
    opts = pp.oppsett(args)
    pp.kjør(filer, lag_behandler, opts)
"""

import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

_UTILS = os.path.dirname(os.path.abspath(__file__))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
APP = os.path.join(_UTILS, "..", "app")
if APP not in sys.path:
    sys.path.insert(0, APP)

import numpy as np

# Merk: ingenting som drar inn paddle eller torch importeres her. En fork()
# etter at CUDA er berørt gir ubrukelige CUDA-kontekster i barneprosessene,
# så koordinatoren må holde seg unna. Arbeidsprosessene importerer selv.
from load_pdf import les_sider


# ── Logging ──────────────────────────────────────────────────────

LOG_MAPPE = "/data2/tmp"


class _Tee:
    """Skriv til både konsoll og loggfil samtidig."""

    def __init__(self, loggfil_sti):
        self._stdout = sys.stdout
        self._fil = open(loggfil_sti, "a", encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._fil.write(data)
        self._fil.flush()

    def flush(self):
        self._stdout.flush()
        self._fil.flush()

    def fileno(self):
        return self._stdout.fileno()


def sett_opp_loggfil(navn):
    """Tee stdout til /data2/tmp/<navn>_<tidsstempel>.log."""
    try:
        os.makedirs(LOG_MAPPE, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        loggfil = os.path.join(LOG_MAPPE, f"{navn}_{ts}.log")
        sys.stdout = _Tee(loggfil)
        print(f"Loggfil: {loggfil}")
        return loggfil
    except OSError as e:
        print(f"Advarsel: Kunne ikke opprette loggfil i {LOG_MAPPE}: {e}")
        return None


# ── Tid og ressurser ─────────────────────────────────────────────

def fmt_tid(sekunder):
    """Formater sekunder til lesbar t:mm:ss / m:ss / Xs."""
    if sekunder < 0:
        return "?"
    s = int(sekunder)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}t {(s % 3600) // 60:02d}m {s % 60:02d}s"


def frigjør_gpu_cache():
    """Gi ubrukt cachet GPU-minne tilbake til CUDA.

    Paddle sin allokator gjenbruker blokker og gir dem ikke tilbake av seg
    selv. Med varierende bildestørrelser ser det ut som en lekkasje.
    """
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def minne_info():
    """(brukt_gb, tilgjengelig_gb, prosent_brukt) for systemminne."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / 1e9, mem.available / 1e9, mem.percent
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for linje in f:
                deler = linje.split()
                if len(deler) >= 2:
                    info[deler[0].rstrip(":")] = int(deler[1]) * 1024
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            brukt = total - available
            return brukt / 1e9, available / 1e9, (brukt / total * 100) if total else 0
    except (FileNotFoundError, KeyError):
        return 0, 0, 0


def gpu_minne_info():
    """(brukt_mb, total_mb) for GPU, eller None.

    Bevisst kun nvidia-smi: paddle-API-et ville initialisert CUDA i
    koordinatoren, og da blir fork() av arbeidsprosessene ugyldig.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            linje = result.stdout.strip().split("\n")[0]
            brukt, total = [int(x.strip()) for x in linje.split(",")]
            return brukt, total
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def gpu_prosent_brukt():
    gpu = gpu_minne_info()
    if gpu and gpu[1] > 0:
        return gpu[0] / gpu[1] * 100
    return None


def er_minne_trygt(grense_prosent=88):
    """Er det nok RAM til å fortsette å pre-rendre?"""
    _, _, prosent = minne_info()
    if prosent == 0:
        return True  # klarte ikke å lese minne, anta OK
    return prosent < grense_prosent


def skriv_ressursstatus():
    brukt, tilg, pct = minne_info()
    deler = []
    if pct > 0:
        deler.append(f"RAM: {brukt:.1f}/{brukt + tilg:.1f} GB ({pct:.0f}%)")
    gpu = gpu_minne_info()
    if gpu:
        deler.append(f"GPU: {gpu[0]}/{gpu[1]} MB ({gpu[0] / gpu[1] * 100:.0f}%)")
    if deler:
        print(f"  Ressurser: {' | '.join(deler)}")


def antall_cpu_kjerner():
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _auto_prefetch():
    """Prefetch-dybde ut fra ledig RAM (~150 MB per pre-rendret PDF)."""
    _, tilg, _ = minne_info()
    if tilg <= 0:
        return 16
    return min(max(int(tilg * 0.10 * 1000 / 150), 12), 64)


def _last_pdf(sti):
    """Rendr en PDF til bildeliste. Returnerer (sti, bilder|Exception)."""
    try:
        return sti, les_sider(sti)
    except Exception as e:
        return sti, e


# ── Batchstørrelse ───────────────────────────────────────────────

# Gulv for AIMD. Gulvet styrer bare hvor lav den *vedvarende* batchen
# kan bli mens kortet er opptatt av noe annet — selve gjenopprettingen
# under en minnefeil deler gruppa lokalt (12→6→3→1) og går side for side
# til slutt, uavhengig av dette tallet. Da er poenget med gulvet å gi
# fra seg mest mulig, ikke å holde på gjennomstrømning.
MIN_BATCH = 1


class AIMDBatchStørrelse:
    """Batchstørrelse styrt av faktiske minnefeil, ikke av nvidia-smi.

    Når flere prosesser deler kortet med hvert sitt minnetak er den globale
    målingen ubrukelig som styringssignal: den ligger *ved* taket når alt
    går som planlagt, og en måling-styrt regulator bremser da hele tiden.

    Samme mønster som TCP, men vi starter på taket i stedet for å lete
    oss opp til det: minnetaket per prosess er alt regnet ut fra kortet
    (FLAGS_gpu_memory_limit_mb), så slow-start fant aldri annet enn den
    grensen vi selv satte — den kostet bare de første minuttene på små
    batcher. Kommer det en minnefeil likevel, typisk fordi noe annet tok
    kortet underveis, halverer vi og vokser forsiktig lineært tilbake.
    """

    def __init__(self, start=12, minimum=MIN_BATCH, maksimum=12):
        self.nåværende = max(min(start, maksimum), minimum)
        self.minimum = minimum
        self.maksimum = maksimum
        self._ok_siden = 0
        self._oom = 0
        self._slow_start = True

    def neste(self):
        if self.nåværende >= self.maksimum:
            return self.nåværende
        if self._slow_start:
            if self._ok_siden >= 1:
                self.nåværende = min(self.maksimum, self.nåværende * 2)
                self._ok_siden = 0
        elif self._ok_siden >= 2:
            self.nåværende += 1
            self._ok_siden = 0
        return self.nåværende

    def ok(self):
        self._ok_siden += 1

    def oom(self):
        self.nåværende = max(self.minimum, self.nåværende // 2)
        self._slow_start = False   # over til forsiktig lineær vekst
        self._ok_siden = -4        # hold igjen noen batcher først
        self._oom += 1


# ── Samplende profiler (py-spy-erstatning uten nettverk) ─────────

class Profil:
    """Teller hvor hovedtråden står, ved å ta stakk-prøver utenfra.

    Den innerste Python-rammen sier hva tiden går til: paddle sine
    run/infer-kall er GPU-arbeid, mens resize, unclip, boxes_from_bitmap,
    rot90 og ascontiguousarray er enkelttrådet CPU-arbeid som GPU-en må
    vente på.
    """

    def __init__(self, intervall=0.01):
        self._intervall = intervall
        self._teller = {}
        self._hovedtråd = threading.get_ident()
        self._stopp = threading.Event()

    def start(self):
        threading.Thread(target=self._løkke, daemon=True).start()

    def _løkke(self):
        while not self._stopp.wait(self._intervall):
            ramme = sys._current_frames().get(self._hovedtråd)
            if ramme is None:
                continue
            kode = ramme.f_code
            nøkkel = f"{kode.co_name}  ({os.path.basename(kode.co_filename)}:{ramme.f_lineno})"
            self._teller[nøkkel] = self._teller.get(nøkkel, 0) + 1

    def stopp(self):
        self._stopp.set()

    def snapshot(self):
        return dict(self._teller)


# ── Arbeidsfordeling mellom prosessene ───────────────────────────

class Arbeidsfordeler:
    """Delt filliste som alle prosessene henter fra.

    Hver tar neste ledige fil fra samme teller, så fordelingen balanserer
    seg selv: den raskeste prosessen tar flest filer. En fast fordeling
    gjettet på forhånd gir alltid en hale der den ene siden er ferdig og
    venter på den andre.

    Telleren er en delt mp.Value; fillisten i hver prosess er en
    fork()-kopi. Derfor må prosessene startes med fork-kontekst.
    """

    def __init__(self, filer, ctx):
        self._filer = filer
        self.totalt = len(filer)
        self._neste = ctx.Value("i", 0)

    def hent(self):
        """Neste filsti, eller None når alle filer er delt ut."""
        with self._neste.get_lock():
            i = self._neste.value
            if i >= self.totalt:
                return None
            self._neste.value = i + 1
        return self._filer[i]

    def tom(self):
        return self._neste.value >= self.totalt


# ── Feilmeldinger ────────────────────────────────────────────────

def kort_feil(e, maks=200):
    """Kort én-linjes sammendrag av en paddle-feil.

    Paddle sine minnefeil er 4-5 KB C++-traceback. Uforkortet fyller de
    loggen fullstendig når flere prosesser treffer taket samtidig.
    """
    tekst = " ".join(str(e).split())
    if "Error Message Summary" in tekst:
        tekst = tekst.split("Error Message Summary")[-1]
    tekst = tekst.strip(" -:")
    return (tekst or repr(e))[:maks]


def er_minnefeil(e):
    tekst = str(e)
    return ("Out of memory" in tekst or "ResourceExhausted" in tekst
            or "OutOfMemory" in tekst or "CUDA out of memory" in tekst
            or isinstance(e, MemoryError))


# ── Arbeidsprosess (én prosess = én komplett pipeline) ───────────

def _worker(oppgave):
    """Prosess-inngang: steng utskrift, kjør pipelinen, rapporter alltid.

    PaddlePaddle og ultralytics skriver modell- og oneDNN-meldinger direkte
    fra C++ til fd 1/2. Med flere prosesser drukner loggen i det, så alt går
    via resultatkøen i stedet.
    """
    kø = oppgave["resultat_kø"]
    wid = oppgave["id"]
    devnull = open(os.devnull, "w")
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    except OSError:
        pass
    sys.stdout = devnull
    sys.stderr = devnull

    antall = 0
    try:
        antall = _pipeline(oppgave) or 0
    except BaseException as e:  # rapporteres — ellers dør prosessen stille
        import traceback
        kø.put(("worker-feil", wid, "", 0, f"{e!r}\n{traceback.format_exc()}"))
    finally:
        kø.put(("ferdig", wid, "", antall, ""))


def _pipeline(oppgave):
    """Rendr PDF-er i tråder, kjør behandleren i batch, rapporter per dokument."""
    wid = oppgave["id"]
    kø = oppgave["resultat_kø"]
    fordeler = oppgave["fordeler"]
    prefetch = oppgave["prefetch"]
    minne_grense = oppgave["minne_grense"]

    # Eget minnetak per prosess. Målt: FLAGS_fraction_of_gpu_memory_to_use
    # blir IKKE respektert av paddle sin inferens-prediktor, mens
    # FLAGS_gpu_memory_limit_mb blir det — uten den vokser prosessene inn i
    # hverandre til kortet er tomt. Må settes før noe rører GPU-en.
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    if oppgave.get("gpu_mb"):
        os.environ["FLAGS_gpu_memory_limit_mb"] = str(int(oppgave["gpu_mb"]))

    tider = {"vente": 0.0}
    oppgave["tider"] = tider
    behandle = oppgave["lag_behandler"](oppgave)
    kø.put(("klar", wid, "", 0, ""))

    if oppgave["dokument_batch"]:
        adaptiv, maks_batch_fast = None, oppgave["dokument_batch"]
    else:
        adaptiv = AIMDBatchStørrelse(
            start=oppgave["start_batch"] or oppgave["maks_batch"],
            minimum=MIN_BATCH, maksimum=oppgave["maks_batch"])
        maks_batch_fast = None

    profil = Profil() if oppgave["profil"] else None
    if profil:
        profil.start()

    executor = ThreadPoolExecutor(max_workers=oppgave["workers"])
    prefetch_kø = deque()

    def _fyll_kø():
        while len(prefetch_kø) < prefetch and er_minne_trygt(minne_grense):
            sti = fordeler.hent()
            if sti is None:
                break
            prefetch_kø.append((sti, executor.submit(_last_pdf, sti)))

    _fyll_kø()

    ferdig = 0
    kø_dybde_sum = 0
    antall_batches = 0
    sist_rapport = 0

    def _stats(ferdig_nå):
        return ("stats", wid, "", ferdig_nå, dict(
            tider, kø=kø_dybde_sum / max(antall_batches, 1),
            batch=maks_batch_fast or adaptiv.nåværende,
            profil=profil.snapshot() if profil else None))

    def _rapporter(resultater, tid_batch, n_dok, n_sider):
        for r in resultater:
            kø.put(("ok", wid, r["navn"], r["sider"], {
                "tekst": r.get("tekst", ""),
                "batch_dok": n_dok,
                "batch_sider": n_sider,
                "batch_tid": tid_batch,
            }))
        return len(resultater)

    while True:
        _fyll_kø()
        if not prefetch_kø:
            if not fordeler.tom():
                time.sleep(2)  # minnetak nådd — vent på ledig RAM
                _fyll_kø()
                if not prefetch_kø:
                    sti = fordeler.hent()
                    if sti is not None:
                        prefetch_kø.append((sti, executor.submit(_last_pdf, sti)))
            if not prefetch_kø:
                break

        maks_batch = maks_batch_fast or adaptiv.neste()
        kø_dybde_sum += len(prefetch_kø)
        antall_batches += 1

        batch_docs = []
        t0 = time.perf_counter()
        while prefetch_kø and len(batch_docs) < maks_batch:
            sti, fut = prefetch_kø.popleft()
            sti, resultat = fut.result()
            navn = os.path.basename(sti)
            if isinstance(resultat, Exception):
                kø.put(("feil", wid, navn, 0, kort_feil(resultat)))
                continue
            batch_docs.append((navn, resultat))
            _fyll_kø()
        tider["vente"] += time.perf_counter() - t0

        if not batch_docs:
            continue

        # Kjør batchen. Tom-for-minne håndteres ved å halvere gruppen og
        # prøve igjen — et dokument skal ikke gå tapt fordi en annen
        # prosess tilfeldigvis holdt minnet i det øyeblikket.
        arbeid = [batch_docs]
        while arbeid:
            gruppe = arbeid.pop()
            n_sider = sum(len(b) for _, b in gruppe)
            start_gruppe = time.perf_counter()
            try:
                resultater = behandle(gruppe)
            except Exception as e:
                frigjør_gpu_cache()
                minnefeil = er_minnefeil(e)
                if minnefeil and adaptiv:
                    adaptiv.oom()
                if minnefeil and len(gruppe) > 1:
                    midt = len(gruppe) // 2
                    arbeid.append(gruppe[midt:])
                    arbeid.append(gruppe[:midt])
                    kø.put(("oom", wid, "", len(gruppe),
                            f"deler batchen (→ {midt} + {len(gruppe) - midt} dok)"))
                    continue
                if not minnefeil:
                    for navn, _b in gruppe:
                        kø.put(("feil", wid, navn, 0, kort_feil(e)))
                    continue
                # Ett dokument alene, side for side. Stigen har to trinn av
                # en grunn: er trykket fra de andre prosessene, hjelper det å
                # vente. Er det prosessens egen allokator som har spist opp
                # minnetaket sitt, hjelper ingen venting — da må modellene
                # rives ned og bygges opp igjen for å gi minnet tilbake.
                nullstill = getattr(behandle, "nullstill", None)
                resultater, siste = None, e
                trinn = ((0, False), (5, False), (0, True), (30, False))
                for forsøk, (pause, riv_ned) in enumerate(trinn, start=1):
                    if pause:
                        time.sleep(pause)
                    if riv_ned and nullstill:
                        kø.put(("oom", wid, gruppe[0][0], 1,
                                "bygger modellene på nytt for å frigi minne"))
                        nullstill()
                    frigjør_gpu_cache()
                    try:
                        resultater = behandle(gruppe, sider_om_gangen=1)
                        break
                    except Exception as e2:
                        siste = e2
                        if not er_minnefeil(e2):
                            break
                if resultater is None:
                    kø.put(("feil", wid, gruppe[0][0], 0, kort_feil(siste)))
                    continue
                kø.put(("oom", wid, gruppe[0][0], 1,
                        f"side for side, klarte det på forsøk {forsøk}"))
            if adaptiv:
                adaptiv.ok()
            ferdig += _rapporter(resultater, time.perf_counter() - start_gruppe,
                                 len(gruppe), n_sider)

        frigjør_gpu_cache()

        if ferdig - sist_rapport >= 20:
            sist_rapport = ferdig
            kø.put(_stats(ferdig))

    executor.shutdown(wait=True)
    if profil:
        profil.stopp()
    kø.put(_stats(ferdig))
    return ferdig


# ── CLI og oppsett ───────────────────────────────────────────────

def legg_til_argumenter(p):
    """Legg til flaggene som styrer parallellkjøringen."""
    p.add_argument("--gpu-prosesser", type=int, nargs="?", default=4, const=-1,
                   metavar="N",
                   help="antall parallelle pipeliner mot samme GPU (default: 4, "
                        "uten tall = auto). Målt på V100S: ~77 %% av tiden i én "
                        "pipeline er enkelttrådet CPU-forbehandling, så én "
                        "prosess klarer ikke å mette kortet")
    p.add_argument("--workers", type=int, default=0,
                   help="tråder for PDF-rendering PER prosess (0=auto)")
    p.add_argument("--prefetch", type=int, default=0,
                   help="pre-rendrede PDF-er i kø per prosess (0=auto fra RAM)")
    p.add_argument("--minne-grense", type=int, default=88,
                   help="maks RAM-bruk i prosent før pre-rendering pauser")
    p.add_argument("--dokument-batch", type=int, default=0,
                   help="maks dokumenter per batch per prosess (0=adaptiv)")
    p.add_argument("--start-batch", type=int, default=0,
                   help="startverdi for den adaptive batchen (0=maks). Sett "
                        "den lavere om du vet at noe annet deler kortet")
    p.add_argument("--gpu-grense", type=int, default=90,
                   help="hvor mye av GPU-minnet prosessene får dele, i prosent. "
                        "Deles likt mellom dem")
    p.add_argument("--profil", action="store_true",
                   help="mål hvor prosessene bruker tiden (faser + stakk-prøver)")
    p.add_argument("--vis-ressurser", action="store_true",
                   help="vis RAM/GPU-status underveis")


def oppsett(args, ekstra=None):
    """Regn ut og skriv ut prosess-, minne- og trådoppsett."""
    kjerner = antall_cpu_kjerner()
    n = args.gpu_prosesser
    if n < 0:
        # Hver pipeline trenger ~1 kjerne til forbehandling + render-tråder
        n = max(min(kjerner // 12, 6), 1)
    n = max(n, 1)

    # Uten et tak per prosess vokser allokatorene uavhengig til kortet er
    # fullt (målt: 30,9 av 32 GB), og da feiler den prosessen som
    # tilfeldigvis ber om minne sist. Det som allerede er i bruk trekkes
    # fra: kjører det en annen jobb på kortet, må vi dele på resten.
    gpu = gpu_minne_info()
    gpu_mb = None
    opptatt = 0
    if gpu and gpu[1] > 0:
        brukt, total = gpu
        opptatt = brukt
        # ~700 MB per prosess går til CUDA-kontekst og modellvekter
        gpu_mb = max((total * args.gpu_grense / 100 - brukt - 700 * n) / n, 512)

    opts = dict(
        n_prosesser=n,
        workers=args.workers or min(max(kjerner // (4 * n), 2), 16),
        prefetch=args.prefetch or max(_auto_prefetch() // n, 8),
        minne_grense=args.minne_grense,
        dokument_batch=args.dokument_batch,
        start_batch=args.start_batch,
        maks_batch=12,   # tak for AIMD; minnetaket over er den reelle grensen
        gpu_mb=gpu_mb,
        profil=args.profil,
        vis_ressurser=args.vis_ressurser,
        ekstra=ekstra or {},
    )

    print(f"  Pipeliner:  {n} × (1 hovedtråd + {opts['workers']} render-tråder), "
          f"prefetch {opts['prefetch']}/prosess")
    if gpu_mb:
        print(f"  GPU-minne:  {gpu_mb:.0f} MB per prosess (FLAGS_gpu_memory_limit_mb)"
              + (f", {opptatt} MB alt i bruk av noe annet" if opptatt > 1000 else ""))
        if gpu_mb < 3000:
            print(f"  !! Lite GPU-minne per prosess. Kjører det en annen jobb på "
                  f"kortet, vent til den er ferdig eller bruk --gpu-prosesser 1")
    if args.dokument_batch:
        print(f"  Batch:      fast {args.dokument_batch} dok per prosess")
    else:
        print(f"  Batch:      AIMD fra {args.start_batch or opts['maks_batch']} "
              f"(maks {opts['maks_batch']}), halveres ved minnefeil "
              f"ned mot {MIN_BATCH}")
    return opts


# ── Koordinator ──────────────────────────────────────────────────

class Gjennomstrømning:
    """Farten akkurat nå, målt over et glidende vindu.

    Snittet siden start kan ikke svare på «går det bra nå?». Hvert nytt
    tall drukner i historikken, så snittet bruker en time på å ta igjen
    virkeligheten: en kjøring som er i full fart etter fem minutter ser
    ut til å bruke halvannen time på å komme dit, og en kjøring som
    halverer farten halvveis ser fin ut lenge etterpå. Vinduet svarer på
    hvordan det går nå; snittet blir stående fordi det er det anslaget
    for gjenstående tid hviler på.
    """

    def __init__(self, vindu_sek=60, maks_punkter=512):
        self._punkter = deque(maxlen=maks_punkter)
        self.vindu_sek = vindu_sek

    def registrer(self, t, dok, sider):
        self._punkter.append((t, dok, sider))

    def vindu(self):
        """(sider/s, dok/s) over vinduet, eller None før vi har nok."""
        if len(self._punkter) < 2:
            return None
        t1, d1, s1 = self._punkter[-1]
        # Nyeste punkt som er minst vindu_sek gammelt. Finnes det ikke —
        # vi er så vidt i gang — bruker vi det eldste vi har, så tallet
        # finnes fra andre statuslinje av selv om vinduet er kort.
        t0, d0, s0 = self._punkter[0]
        for t, d, sd in self._punkter:
            if t1 - t < self.vindu_sek:
                break
            t0, d0, s0 = t, d, sd
        dt = t1 - t0
        if dt <= 0:
            return None
        return (s1 - s0) / dt, (d1 - d0) / dt


def _skriv_status(stats, elapsed, ferdig, sider, med_profil, n_profil=12,
                  fart=None):
    """Fasefordeling per prosess og samlet gjennomstrømning."""
    if not stats:
        return
    gpu_mem = gpu_prosent_brukt()
    gpu_str = f" | GPU-minne: {gpu_mem:.0f}%" if gpu_mem else ""
    if elapsed > 0:
        snitt = (f"{sider / elapsed:.2f} sider/s | "
                 f"{ferdig / elapsed * 3600:.0f} dok/time")
        v = fart.vindu() if fart else None
        if v:
            print(f"  ⚡ {v[0]:.2f} sider/s | {v[1] * 3600:.0f} dok/time"
                  f"  (snitt siden start: {snitt}){gpu_str}")
        else:
            print(f"  ⚡ {snitt}{gpu_str}")
    for wid in sorted(stats):
        d = stats[wid]
        faser = {k: v for k, v in d.items()
                 if k not in ("kø", "batch", "profil", "vente")}
        sum_fase = sum(faser.values())
        if sum_fase <= 0:
            continue
        deler = " | ".join(f"{navn} {100 * v / sum_fase:.0f}%"
                           for navn, v in sorted(faser.items()))
        print(f"     [{wid}] batch={d['batch']} kø={d['kø']:.0f} | vente-render "
              f"{100 * d['vente'] / (sum_fase + d['vente']):.0f}% | {deler}")
    if not med_profil:
        return
    samlet = {}
    for d in stats.values():
        for navn, ant in (d.get("profil") or {}).items():
            samlet[navn] = samlet.get(navn, 0) + ant
    totalt = sum(samlet.values())
    if not totalt:
        return
    print(f"     Profil (alle prosesser) — {totalt} prøver:")
    for navn, ant in sorted(samlet.items(), key=lambda kv: -kv[1])[:n_profil]:
        print(f"       {100 * ant / totalt:5.1f}%  {navn}")


def kjør(filer, lag_behandler, opts):
    """Kjør `lag_behandler` sin behandler over `filer` i parallelle prosesser.

    Returnerer (ferdig, total_sider, feilet) der feilet er [(navn, feil), ...].
    """
    ctx = mp.get_context("fork")   # fordeleren deles via fork
    fordeler = Arbeidsfordeler(filer, ctx)
    kø = ctx.Queue()

    felles = dict(
        resultat_kø=kø, fordeler=fordeler, lag_behandler=lag_behandler,
        workers=opts["workers"], prefetch=opts["prefetch"],
        minne_grense=opts["minne_grense"], dokument_batch=opts["dokument_batch"],
        start_batch=opts["start_batch"], maks_batch=opts["maks_batch"],
        gpu_mb=opts["gpu_mb"], profil=opts["profil"], ekstra=opts["ekstra"],
    )
    prosesser = []
    for i in range(opts["n_prosesser"]):
        pr = ctx.Process(target=_worker, args=(dict(felles, id=f"p{i}"),),
                         daemon=True)
        pr.start()
        prosesser.append(pr)

    totalt = len(filer)
    print(f"\nStarter på {totalt} dokumenter i {len(prosesser)} prosess(er). "
          f"Laster modeller...\n")

    ferdig = 0
    total_sider = 0
    feilet = []
    per_prosess = {}
    stats = {}
    klar = avsluttet = oom_hendelser = 0
    start_alle = None
    fart = Gjennomstrømning()
    neste_status = 20
    neste_ressurs = 100

    try:
        while avsluttet < len(prosesser):
            try:
                status, wid, navn, n, data = kø.get(timeout=30)
            except Exception:
                if all(not pr.is_alive() for pr in prosesser):
                    print("!! Alle prosesser er borte — avslutter")
                    break
                continue

            if status == "klar":
                klar += 1
                print(f"  [{wid}] modeller lastet")
                if klar == len(prosesser):
                    start_alle = time.perf_counter()
                    print(f"\nAlle {klar} prosesser i gang.\n")
            elif status == "feil":
                feilet.append((navn, data))
                ferdig += 1
                print(f"[{ferdig}/{totalt}] ✗ {navn} [{wid}]: {data}")
            elif status == "oom":
                oom_hendelser += 1
                print(f"  [{wid}] GPU tom for minne på {n} dok — {data}")
            elif status == "worker-feil":
                print(f"  [{wid}] KRASJET: {data}")
            elif status == "ferdig":
                avsluttet += 1
                print(f"  [{wid}] avsluttet — {n} dokumenter")
            elif status == "stats":
                stats[wid] = data
            else:  # "ok"
                if start_alle is None:
                    start_alle = time.perf_counter()
                ferdig += 1
                total_sider += n
                teller = per_prosess.setdefault(wid, [0, 0])
                teller[0] += 1
                teller[1] += n

                elapsed = time.perf_counter() - start_alle
                fart.registrer(elapsed, ferdig, total_sider)
                snitt = elapsed / max(ferdig, 1)
                print(f"[{ferdig}/{totalt}] ✓ {navn} [{wid}]: "
                      f"{n} side(r), {data['tekst']} — "
                      f"batch {data['batch_dok']} dok/{data['batch_sider']} sider "
                      f"på {data['batch_tid']:.1f}s "
                      f"(gått: {fmt_tid(elapsed)}, "
                      f"gjenstår: {fmt_tid(snitt * (totalt - ferdig))})")

                if ferdig >= neste_status:
                    neste_status += 20
                    _skriv_status(stats, elapsed, ferdig, total_sider,
                                  opts["profil"], fart=fart)
                if opts["vis_ressurser"] and ferdig >= neste_ressurs:
                    neste_ressurs += 100
                    skriv_ressursstatus()
    except KeyboardInterrupt:
        print("\n!! Avbrutt — skriver oppsummering for det som er gjort")

    for pr in prosesser:
        pr.join(timeout=5)

    vegg_tid = time.perf_counter() - (start_alle or time.perf_counter())
    print(f"\n{'=' * 60}")
    print(f"Ferdig! {ferdig}/{totalt} dokumenter, {total_sider} sider")
    for wid in sorted(per_prosess):
        dok, sider = per_prosess[wid]
        print(f"  [{wid}] {dok} dok, {sider} sider "
              f"({100 * dok / max(ferdig, 1):.0f}%)")
    if oom_hendelser:
        print(f"  Tom-for-minne: {oom_hendelser} ganger (batchen ble delt og "
              f"kjørt om — ingen dokumenter tapt av det)")
    print(f"  Vegg-tid:   {fmt_tid(vegg_tid)}")
    if vegg_tid > 0:
        print(f"  Throughput: {total_sider / vegg_tid:.2f} sider/s, "
              f"{ferdig / vegg_tid * 3600:.0f} dok/time")
    _skriv_status(stats, vegg_tid, ferdig, total_sider, opts["profil"],
                  n_profil=20)

    if feilet:
        print(f"\nFeilet ({len(feilet)}):")
        for navn, feil in feilet[:10]:
            print(f"  {navn}: {feil}")
        if len(feilet) > 10:
            print(f"  ... og {len(feilet) - 10} til")

    skriv_ressursstatus()
    return ferdig, total_sider, feilet
