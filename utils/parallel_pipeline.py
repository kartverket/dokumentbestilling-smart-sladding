"""Machinery for running heavy document jobs in parallel against one GPU.

The main process only coordinates; all work happens in independent processes
pulling files from a shared counter. Measured on V100S + Xeon 6230: ~77 % of
the time in a single pipeline went to single-threaded CPU preprocessing and
only ~11 % to GPU work, so the card idled half the time. Four processes against
the same card gave 3.3x throughput and saturated it (228 of 250 W).

Memory management follows from that: each process gets its own cap on the card
(FLAGS_gpu_memory_limit_mb), the batch size finds that cap itself via slow
start and halving on memory errors, and a memory error costs a rerun of the
same documents, never a lost document.

Run from a tool:

    import parallel_pipeline as pp

    def make_handler(task):
        # Runs in the worker process. Import heavy modules HERE.
        from paddle_ocr_model_fnr import read_tokens_batched
        timings = task["timings"]

        def handle(group, pages_at_a_time=None):
            # group: [(name, [images]), ...]
            # return one dict per document: {"navn", "sider", "text"}
            ...
        return handle

    p = argparse.ArgumentParser()
    pp.add_arguments(p)
    args = p.parse_args()
    opts = pp.setup(args)
    pp.run(files, make_handler, opts)
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

# Nothing that pulls in paddle or torch may be imported here: a fork() after
# CUDA has been touched leaves unusable CUDA contexts in the children. The
# workers do their own importing.
from load_pdf import read_pages


# ── Logging ──────────────────────────────────────────────────────

LOG_DIR = "/data2/tmp"


class _Tee:
    """Write to console and log file at once."""

    def __init__(self, log_file_path):
        self._stdout = sys.stdout
        self._file = open(log_file_path, "a", encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def fileno(self):
        return self._stdout.fileno()


def setup_logfile(name):
    """Tee stdout to /data2/tmp/<name>_<timestamp>.log."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(LOG_DIR, f"{name}_{ts}.log")
        sys.stdout = _Tee(log_file)
        print(f"Log file: {log_file}")
        return log_file
    except OSError as e:
        print(f"Warning: could not create a log file in {LOG_DIR}: {e}")
        return None


# ── Time and resources ───────────────────────────────────────────

def fmt_time(seconds):
    """Format seconds as a readable h:mm:ss / m:ss / Xs."""
    if seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}t {(s % 3600) // 60:02d}m {s % 60:02d}s"


def free_gpu_cache():
    """Return unused cached GPU memory to CUDA.

    Paddle's allocator reuses blocks and never hands them back on its own.
    """
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def memory_info():
    """(used_gb, available_gb, percent_used) for system memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / 1e9, mem.available / 1e9, mem.percent
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            used_mem = total - available
            return used_mem / 1e9, available / 1e9, (used_mem / total * 100) if total else 0
    except (FileNotFoundError, KeyError):
        return 0, 0, 0


def gpu_memory_info():
    """(used_mb, total_mb) for the GPU, or None.

    nvidia-smi only: the paddle API would initialise CUDA in the coordinator,
    which makes fork() of the workers invalid.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            used_mem, total = [int(x.strip()) for x in line.split(",")]
            return used_mem, total
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def gpu_percent_used():
    gpu = gpu_memory_info()
    if gpu and gpu[1] > 0:
        return gpu[0] / gpu[1] * 100
    return None


def is_memory_safe(limit_percent=88):
    """Is there enough RAM to keep pre-rendering?"""
    _, _, pct = memory_info()
    if pct == 0:
        return True  # could not read memory, assume OK
    return pct < limit_percent


def write_resource_status():
    used_mem, avail, pct = memory_info()
    parts = []
    if pct > 0:
        parts.append(f"RAM: {used_mem:.1f}/{used_mem + avail:.1f} GB ({pct:.0f}%)")
    gpu = gpu_memory_info()
    if gpu:
        parts.append(f"GPU: {gpu[0]}/{gpu[1]} MB ({gpu[0] / gpu[1] * 100:.0f}%)")
    if parts:
        print(f"  Resources: {' | '.join(parts)}")


def cpu_core_count():
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def _auto_prefetch():
    """Prefetch depth from free RAM (~150 MB per pre-rendered PDF)."""
    _, avail, _ = memory_info()
    if avail <= 0:
        return 16
    return min(max(int(avail * 0.10 * 1000 / 150), 12), 64)


def _last_pdf(path):
    """Render a PDF to a list of images. Returns (path, images|Exception)."""
    try:
        return path, read_pages(path)
    except Exception as e:
        return path, e


# ── Batch size ───────────────────────────────────────────────────

# Floor for AIMD. It only bounds how low the *sustained* batch goes while the
# card is busy with something else; recovery from a memory error splits the
# group locally (12→6→3→1) and ends up page by page regardless of this number.
MIN_BATCH = 1


class AIMDBatchSize:
    """Batch size driven by actual memory errors, not by nvidia-smi.

    With several processes under separate memory caps the global measurement is
    useless as a control signal: it sits *at* the cap when all is well. Like
    TCP, but starting at the cap rather than searching up to it, since the
    per-process cap is already derived from the card.
    """

    def __init__(self, start=12, minimum=MIN_BATCH, maximum=12):
        self.current = max(min(start, maximum), minimum)
        self.minimum = minimum
        self.maximum = maximum
        self._ok_siden = 0
        self._oom = 0
        self._slow_start = True

    def next_one(self):
        if self.current >= self.maximum:
            return self.current
        if self._slow_start:
            if self._ok_siden >= 1:
                self.current = min(self.maximum, self.current * 2)
                self._ok_siden = 0
        elif self._ok_siden >= 2:
            self.current += 1
            self._ok_siden = 0
        return self.current

    def ok(self):
        self._ok_siden += 1

    def oom(self):
        self.current = max(self.minimum, self.current // 2)
        self._slow_start = False   # switch to cautious linear growth
        self._ok_siden = -4        # hold back for a few batches first
        self._oom += 1


# ── Sampling profiler (a py-spy replacement without networking) ──

class Profile:
    """Count where the main thread is, by sampling its stack from outside.

    The innermost Python frame says where time goes: paddle's run/infer calls
    are GPU work, resize/unclip/boxes_from_bitmap/rot90/ascontiguousarray are
    single-threaded CPU work the GPU waits for.
    """

    def __init__(self, interval=0.01):
        self._intervall = interval
        self._counter = {}
        self._main_thread = threading.get_ident()
        self._stopp = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stopp.wait(self._intervall):
            frame = sys._current_frames().get(self._main_thread)
            if frame is None:
                continue
            code = frame.f_code
            stat_key = f"{code.co_name}  ({os.path.basename(code.co_filename)}:{frame.f_lineno})"
            self._counter[stat_key] = self._counter.get(stat_key, 0) + 1

    def stop(self):
        self._stopp.set()

    def snapshot(self):
        return dict(self._counter)


# ── Work distribution across the processes ───────────────────────

class WorkDistributor:
    """Shared file list that every process pulls from.

    Self-balancing, where a split guessed up front always leaves a tail. The
    counter is a shared mp.Value and each process holds a fork() copy of the
    list, so processes must be started with the fork context.
    """

    def __init__(self, files, ctx):
        self._files = files
        self.total = len(files)
        self._next = ctx.Value("i", 0)

    def fetch(self):
        """Next file path, or None once every file has been handed out."""
        with self._next.get_lock():
            i = self._next.value
            if i >= self.total:
                return None
            self._next.value = i + 1
        return self._files[i]

    def empty(self):
        return self._next.value >= self.total


# ── Error messages ───────────────────────────────────────────────

def short_error(e, max_items=200):
    """Short one-line summary of a paddle error.

    Paddle memory errors are 4-5 KB of C++ traceback and swamp the log.
    """
    line_text = " ".join(str(e).split())
    if "Error Message Summary" in line_text:
        line_text = line_text.split("Error Message Summary")[-1]
    line_text = line_text.strip(" -:")
    return (line_text or repr(e))[:max_items]


def is_memory_error(e):
    line_text = str(e)
    return ("Out of memory" in line_text or "ResourceExhausted" in line_text
            or "OutOfMemory" in line_text or "CUDA out of memory" in line_text
            or isinstance(e, MemoryError))


# ── Worker process (one process = one complete pipeline) ─────────

def _worker(task):
    """Process entry: silence output, run the pipeline, always report.

    PaddlePaddle and ultralytics write straight from C++ to fd 1/2, which
    drowns the log, so everything goes through the result queue instead.
    """
    queue = task["result_queue"]
    wid = task["id"]
    devnull = open(os.devnull, "w")
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    except OSError:
        pass
    sys.stdout = devnull
    sys.stderr = devnull

    count = 0
    try:
        count = _pipeline(task) or 0
    except BaseException as e:  # reported, else the process dies silently
        import traceback
        queue.put(("worker-feil", wid, "", 0, f"{e!r}\n{traceback.format_exc()}"))
    finally:
        queue.put(("ferdig", wid, "", count, ""))


def _pipeline(task):
    """Render PDFs in threads, run the handler in batches, report per document."""
    wid = task["id"]
    queue = task["result_queue"]
    distributor = task["distributor"]
    prefetch = task["prefetch"]
    memory_limit = task["memory_limit"]

    # Own memory cap per process. Measured: FLAGS_fraction_of_gpu_memory_to_use
    # is NOT respected by paddle's inference predictor, while
    # FLAGS_gpu_memory_limit_mb is. Without it the processes grow into each
    # other until the card is empty. Must be set before anything touches the GPU.
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    if task.get("gpu_mb"):
        os.environ["FLAGS_gpu_memory_limit_mb"] = str(int(task["gpu_mb"]))

    timings = {"vente": 0.0}
    task["timings"] = timings
    handle = task["make_handler"](task)
    queue.put(("ready", wid, "", 0, ""))

    if task["document_batch"]:
        adaptive, max_batch_fixed = None, task["document_batch"]
    else:
        adaptive = AIMDBatchSize(
            start=task["start_batch"] or task["max_batch"],
            minimum=MIN_BATCH, maximum=task["max_batch"])
        max_batch_fixed = None

    profile = Profile() if task["profile"] else None
    if profile:
        profile.start()

    executor = ThreadPoolExecutor(max_workers=task["workers"])
    prefetch_queue = deque()

    def _fill_queue():
        while len(prefetch_queue) < prefetch and is_memory_safe(memory_limit):
            path = distributor.fetch()
            if path is None:
                break
            prefetch_queue.append((path, executor.submit(_last_pdf, path)))

    _fill_queue()

    done = 0
    queue_depth_total = 0
    count_batches = 0
    last_report = 0

    def _stats(done_now):
        return ("stats", wid, "", done_now, dict(
            timings, queue=queue_depth_total / max(count_batches, 1),
            batch=max_batch_fixed or adaptive.current,
            profile=profile.snapshot() if profile else None))

    def _report(results, time_batch, n_doc, n_pages):
        for r in results:
            queue.put(("ok", wid, r["navn"], r["sider"], {
                "text": r.get("text", ""),
                "batch_dok": n_doc,
                "batch_sider": n_pages,
                "batch_tid": time_batch,
            }))
        return len(results)

    while True:
        _fill_queue()
        if not prefetch_queue:
            if not distributor.empty():
                time.sleep(2)  # RAM cap reached, wait for free RAM
                _fill_queue()
                if not prefetch_queue:
                    path = distributor.fetch()
                    if path is not None:
                        prefetch_queue.append((path, executor.submit(_last_pdf, path)))
            if not prefetch_queue:
                break

        max_batch = max_batch_fixed or adaptive.next_one()
        queue_depth_total += len(prefetch_queue)
        count_batches += 1

        batch_docs = []
        t0 = time.perf_counter()
        while prefetch_queue and len(batch_docs) < max_batch:
            path, fut = prefetch_queue.popleft()
            path, outcome = fut.result()
            name = os.path.basename(path)
            if isinstance(outcome, Exception):
                queue.put(("feil", wid, name, 0, short_error(outcome)))
                continue
            batch_docs.append((name, outcome))
            _fill_queue()
        timings["vente"] += time.perf_counter() - t0

        if not batch_docs:
            continue

        # Out of memory halves the group and retries: no document may be lost
        # because another process happened to hold the memory at that moment.
        work = [batch_docs]
        while work:
            group = work.pop()
            n_pages = sum(len(b) for _, b in group)
            start_group = time.perf_counter()
            try:
                results = handle(group)
            except Exception as e:
                free_gpu_cache()
                memory_error = is_memory_error(e)
                if memory_error and adaptive:
                    adaptive.oom()
                if memory_error and len(group) > 1:
                    mid = len(group) // 2
                    work.append(group[mid:])
                    work.append(group[:mid])
                    queue.put(("oom", wid, "", len(group),
                            f"splitting the batch (→ {mid} + {len(group) - mid} docs)"))
                    continue
                if not memory_error:
                    for name, _b in group:
                        queue.put(("feil", wid, name, 0, short_error(e)))
                    continue
                # One document alone, page by page. The ladder has two kinds
                # of step on purpose: waiting helps when the pressure comes from
                # the other processes, but not when this process's own allocator
                # has eaten its cap. Then the models must be torn down and
                # rebuilt to give the memory back.
                reset = getattr(handle, "reset", None)
                results, last = None, e
                trinn = ((0, False), (5, False), (0, True), (30, False))
                for attempt, (pause, tear_ned) in enumerate(trinn, start=1):
                    if pause:
                        time.sleep(pause)
                    if tear_ned and reset:
                        queue.put(("oom", wid, group[0][0], 1,
                                "rebuilding the models to free memory"))
                        reset()
                    free_gpu_cache()
                    try:
                        results = handle(group, pages_at_a_time=1)
                        break
                    except Exception as e2:
                        last = e2
                        if not is_memory_error(e2):
                            break
                if results is None:
                    queue.put(("feil", wid, group[0][0], 0, short_error(last)))
                    continue
                queue.put(("oom", wid, group[0][0], 1,
                        f"page by page, succeeded on attempt {attempt}"))
            if adaptive:
                adaptive.ok()
            done += _report(results, time.perf_counter() - start_group,
                                 len(group), n_pages)

        free_gpu_cache()

        if done - last_report >= 20:
            last_report = done
            queue.put(_stats(done))

    executor.shutdown(wait=True)
    if profile:
        profile.stop()
    queue.put(_stats(done))
    return done


# ── CLI and setup ────────────────────────────────────────────────

def add_arguments(p):
    """Add the flags that control the parallel run."""
    p.add_argument("--gpu-processes", type=int, nargs="?", default=4, const=-1,
                   metavar="N",
                   help="parallel pipelines against the same GPU (default: 4, "
                        "no number = auto). Measured on V100S: ~77 %% of the "
                        "time in one pipeline is single-threaded CPU "
                        "preprocessing, so one process cannot saturate the card")
    p.add_argument("--workers", type=int, default=0,
                   help="threads for PDF rendering PER process (0=auto)")
    p.add_argument("--prefetch", type=int, default=0,
                   help="pre-rendered PDFs queued per process (0=auto from RAM)")
    p.add_argument("--memory-limit", type=int, default=88,
                   help="max RAM use in percent before pre-rendering pauses")
    p.add_argument("--document-batch", type=int, default=0,
                   help="max documents per batch per process (0=adaptive)")
    p.add_argument("--start-batch", type=int, default=0,
                   help="starting value for the adaptive batch (0=max). Lower "
                        "it if you know something else shares the card")
    p.add_argument("--gpu-limit", type=int, default=90,
                   help="percent of GPU memory the processes may share, split "
                        "equally between them")
    p.add_argument("--profile", action="store_true",
                   help="measure where the processes spend time (phases + stack samples)")
    p.add_argument("--show-resources", action="store_true",
                   help="show RAM/GPU status along the way")


def setup(args, extra=None):
    """Compute and print the process, memory and thread setup."""
    cores = cpu_core_count()
    n = args.gpu_processes
    if n < 0:
        # Each pipeline needs ~1 core for preprocessing + render threads.
        n = max(min(cores // 12, 6), 1)
    n = max(n, 1)

    # Without a per-process cap the allocators grow independently until the
    # card is full (measured: 30.9 of 32 GB), and whichever process asks for
    # memory last is the one that fails. Memory already in use is subtracted:
    # if another job is on the card, we share what is left.
    gpu = gpu_memory_info()
    gpu_mb = None
    in_use = 0
    if gpu and gpu[1] > 0:
        used_mem, total = gpu
        in_use = used_mem
        # ~700 MB per process goes to CUDA context and model weights.
        gpu_mb = max((total * args.gpu_limit / 100 - used_mem - 700 * n) / n, 512)

    opts = dict(
        n_processes=n,
        workers=args.workers or min(max(cores // (4 * n), 2), 16),
        prefetch=args.prefetch or max(_auto_prefetch() // n, 8),
        memory_limit=args.memory_limit,
        document_batch=args.document_batch,
        start_batch=args.start_batch,
        max_batch=12,   # AIMD ceiling; the memory cap above is the real limit
        gpu_mb=gpu_mb,
        profile=args.profile,
        show_resources=args.show_resources,
        extra=extra or {},
    )

    print(f"  Pipelines:  {n} × (1 main thread + {opts['workers']} render threads), "
          f"prefetch {opts['prefetch']}/process")
    if gpu_mb:
        print(f"  GPU memory: {gpu_mb:.0f} MB per process (FLAGS_gpu_memory_limit_mb)"
              + (f", {in_use} MB already used by something else" if in_use > 1000 else ""))
        if gpu_mb < 3000:
            print(f"  !! Little GPU memory per process. If another job is on the "
                  f"card, wait for it to finish or use --gpu-processes 1")
    if args.document_batch:
        print(f"  Batch:      fixed {args.document_batch} docs per process")
    else:
        print(f"  Batch:      AIMD from {args.start_batch or opts['max_batch']} "
              f"(max {opts['max_batch']}), halved on memory errors "
              f"down towards {MIN_BATCH}")
    return opts


# ── Coordinator ──────────────────────────────────────────────────

class Throughput:
    """Current speed, over a sliding window.

    The average since start takes an hour to catch up with reality, so it
    cannot say whether things are going well now. It stays anyway, because the
    remaining-time estimate rests on it.
    """

    def __init__(self, window_sec=60, max_points=512):
        self._punkter = deque(maxlen=max_points)
        self.window_sec = window_sec

    def registrer(self, t, doc, pages):
        self._punkter.append((t, doc, pages))

    def window(self):
        """(pages/s, docs/s) over the window, or None until we have enough."""
        if len(self._punkter) < 2:
            return None
        t1, d1, s1 = self._punkter[-1]
        # Newest point at least vindu_sek old; before one exists, fall back to
        # the oldest so a number is available from the second status line on.
        t0, d0, s0 = self._punkter[0]
        for t, d, sd in self._punkter:
            if t1 - t < self.window_sec:
                break
            t0, d0, s0 = t, d, sd
        dt = t1 - t0
        if dt <= 0:
            return None
        return (s1 - s0) / dt, (d1 - d0) / dt


def _write_status(worker_stats, elapsed, done, pages, with_profile, n_profile=12,
                  speed=None):
    """Phase split per process, plus overall throughput."""
    if not worker_stats:
        return
    gpu_mem = gpu_percent_used()
    gpu_str = f" | GPU memory: {gpu_mem:.0f}%" if gpu_mem else ""
    if elapsed > 0:
        mean = (f"{pages / elapsed:.2f} pages/s | "
                 f"{done / elapsed * 3600:.0f} docs/h")
        v = speed.window() if speed else None
        if v:
            print(f"  ⚡ {v[0]:.2f} pages/s | {v[1] * 3600:.0f} docs/h"
                  f"  (avg since start: {mean}){gpu_str}")
        else:
            print(f"  ⚡ {mean}{gpu_str}")
    for wid in sorted(worker_stats):
        d = worker_stats[wid]
        phases = {k: v for k, v in d.items()
                 if k not in ("queue", "batch", "profile", "vente")}
        total_phase = sum(phases.values())
        if total_phase <= 0:
            continue
        parts = " | ".join(f"{name} {100 * v / total_phase:.0f}%"
                           for name, v in sorted(phases.items()))
        print(f"     [{wid}] batch={d['batch']} queue={d['queue']:.0f} | wait-render "
              f"{100 * d['vente'] / (total_phase + d['vente']):.0f}% | {parts}")
    if not with_profile:
        return
    combined = {}
    for d in worker_stats.values():
        for name, cnt in (d.get("profile") or {}).items():
            combined[name] = combined.get(name, 0) + cnt
    total = sum(combined.values())
    if not total:
        return
    print(f"     Profile (all processes), {total} samples:")
    for name, cnt in sorted(combined.items(), key=lambda kv: -kv[1])[:n_profile]:
        print(f"       {100 * cnt / total:5.1f}%  {name}")


def run(files, make_handler, opts):
    """Run the handler from `make_handler` over `files` in parallel processes.

    Returns (done, total_pages, failed) where failed is [(name, error), ...].
    """
    ctx = mp.get_context("fork")   # the distributor is shared through fork
    distributor = WorkDistributor(files, ctx)
    queue = ctx.Queue()

    common = dict(
        result_queue=queue, distributor=distributor, make_handler=make_handler,
        workers=opts["workers"], prefetch=opts["prefetch"],
        memory_limit=opts["memory_limit"], document_batch=opts["document_batch"],
        start_batch=opts["start_batch"], max_batch=opts["max_batch"],
        gpu_mb=opts["gpu_mb"], profile=opts["profile"], extra=opts["extra"],
    )
    processes = []
    for i in range(opts["n_processes"]):
        pr = ctx.Process(target=_worker, args=(dict(common, id=f"p{i}"),),
                         daemon=True)
        pr.start()
        processes.append(pr)

    total = len(files)
    print(f"\nStarting on {total} documents in {len(processes)} process(es). "
          f"Loading models...\n")

    done = 0
    total_pages = 0
    failed = []
    per_process = {}
    worker_stats = {}
    ready = finished = oom_events = 0
    start_every = None
    speed = Throughput()
    next_status = 20
    next_resource = 100

    try:
        while finished < len(processes):
            try:
                status, wid, name, n, data = queue.get(timeout=30)
            except Exception:
                if all(not pr.is_alive() for pr in processes):
                    print("!! Every process is gone, stopping")
                    break
                continue

            if status == "ready":
                ready += 1
                print(f"  [{wid}] models loaded")
                if ready == len(processes):
                    start_every = time.perf_counter()
                    print(f"\nAll {ready} processes running.\n")
            elif status == "feil":
                failed.append((name, data))
                done += 1
                print(f"[{done}/{total}] ✗ {name} [{wid}]: {data}")
            elif status == "oom":
                oom_events += 1
                print(f"  [{wid}] GPU out of memory on {n} doc(s), {data}")
            elif status == "worker-feil":
                print(f"  [{wid}] CRASHED: {data}")
            elif status == "ferdig":
                finished += 1
                print(f"  [{wid}] finished, {n} documents")
            elif status == "stats":
                worker_stats[wid] = data
            else:  # "ok"
                if start_every is None:
                    start_every = time.perf_counter()
                done += 1
                total_pages += n
                counter = per_process.setdefault(wid, [0, 0])
                counter[0] += 1
                counter[1] += n

                elapsed = time.perf_counter() - start_every
                speed.registrer(elapsed, done, total_pages)
                mean = elapsed / max(done, 1)
                print(f"[{done}/{total}] ✓ {name} [{wid}]: "
                      f"{n} page(s), {data['text']}, "
                      f"batch {data['batch_dok']} docs/{data['batch_sider']} pages "
                      f"in {data['batch_tid']:.1f}s "
                      f"(elapsed: {fmt_time(elapsed)}, "
                      f"remaining: {fmt_time(mean * (total - done))})")

                if done >= next_status:
                    next_status += 20
                    _write_status(worker_stats, elapsed, done, total_pages,
                                  opts["profile"], speed=speed)
                if opts["show_resources"] and done >= next_resource:
                    next_resource += 100
                    write_resource_status()
    except KeyboardInterrupt:
        print("\n!! Interrupted, printing a summary of what was done")

    for pr in processes:
        pr.join(timeout=5)

    wall_time = time.perf_counter() - (start_every or time.perf_counter())
    print(f"\n{'=' * 60}")
    print(f"Done! {done}/{total} documents, {total_pages} pages")
    for wid in sorted(per_process):
        doc, pages = per_process[wid]
        print(f"  [{wid}] {doc} docs, {pages} pages "
              f"({100 * doc / max(done, 1):.0f}%)")
    if oom_events:
        print(f"  Out of memory: {oom_events} time(s) (the batch was split "
              f"and rerun, no documents lost to it)")
    print(f"  Wall time:  {fmt_time(wall_time)}")
    if wall_time > 0:
        print(f"  Throughput: {total_pages / wall_time:.2f} pages/s, "
              f"{done / wall_time * 3600:.0f} docs/h")
    _write_status(worker_stats, wall_time, done, total_pages, opts["profile"],
                  n_profile=20)

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, error in failed[:10]:
            print(f"  {name}: {error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    write_resource_status()
    return done, total_pages, failed
