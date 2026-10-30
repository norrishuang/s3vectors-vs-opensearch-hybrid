"""
Query load test for the two hybrid-search backends.

For each backend, in order (OpenSearch first, then S3 Vectors):
  1. warmup     - every query once, not measured
  2. serial     - one thread, every query `serial_rounds` times; average latency
  3. concurrent - `concurrency` Python threads loop over the queries until
                  `duration_s` has passed; QPS and latency percentiles

Latency is client-side wall-clock time for one search call, measured the same
way for both backends. Query embedding happens once before the test and is not
included. OpenSearch's server-side `took` is recorded as well, for reference.
"""

import threading
import time
import uuid
from collections import Counter

PHASES = ("warmup", "serial", "concurrent")


def percentile(sorted_values, pct):
    """Linear-interpolated percentile of an already sorted list (pct in 0..100)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def summarize(latencies_ms):
    values = sorted(latencies_ms)
    if not values:
        return {"count": 0}
    r = lambda x: None if x is None else round(x, 1)  # noqa: E731
    return {
        "count": len(values),
        "avg": r(sum(values) / len(values)),
        "min": r(values[0]),
        "p50": r(percentile(values, 50)),
        "p90": r(percentile(values, 90)),
        "p95": r(percentile(values, 95)),
        "p99": r(percentile(values, 99)),
        "max": r(values[-1]),
    }


class CallError(Exception):
    """A failed search call. `kind` is a short label such as 'HTTP 429'."""

    def __init__(self, kind, throttled=False):
        super().__init__(kind)
        self.kind = kind
        self.throttled = throttled


class Target:
    """
    One backend under test.

    `new_worker()` returns a callable `call(text, vector) -> server_ms or None`
    bound to its own connection resources; each benchmark thread gets one.
    The callable raises CallError on failure.
    """

    def __init__(self, key, label, new_worker):
        self.key = key
        self.label = label
        self.new_worker = new_worker


class Benchmark:
    def __init__(self, targets, queries, concurrency, duration_s, serial_rounds, warmup, settings):
        self.id = uuid.uuid4().hex[:12]
        self.targets = targets
        self.queries = queries            # list of (text, vector)
        self.concurrency = concurrency
        self.duration_s = duration_s
        self.serial_rounds = serial_rounds
        self.warmup = warmup
        self.settings = settings          # echoed back to the UI
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state = "running"            # running | finished | stopped | failed
        self.error = None
        self.phase = None
        self.phase_started = None
        self.progress_count = 0
        self.results = {t.key: {"label": t.label} for t in targets}
        self.started = time.time()
        self.finished = None

    # -- status ------------------------------------------------------------
    def status(self):
        with self.lock:
            live = None
            if self.phase and self.phase_started:
                elapsed = time.perf_counter() - self.phase_started
                live = {
                    "phase": self.phase,
                    "elapsed_s": round(elapsed, 1),
                    "requests": self.progress_count,
                    "qps": round(self.progress_count / elapsed, 1) if elapsed > 0 else 0.0,
                }
            return {
                "id": self.id,
                "state": self.state,
                "error": self.error,
                "settings": self.settings,
                "live": live,
                "results": self.results,
                "started": self.started,
                "finished": self.finished,
            }

    def _set_phase(self, name):
        with self.lock:
            self.phase = name
            self.phase_started = time.perf_counter()
            self.progress_count = 0

    def _tick(self):
        with self.lock:
            self.progress_count += 1

    # -- phases --------------------------------------------------------------
    def _run_serial(self, target, rounds, record):
        call = target.new_worker()
        latencies, server, errors = [], [], Counter()
        for _ in range(rounds):
            for text, vector in self.queries:
                if self.stop_event.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    server_ms = call(text, vector)
                except CallError as e:
                    errors[e.kind] += 1
                    continue
                finally:
                    self._tick()
                latencies.append((time.perf_counter() - t0) * 1000.0)
                if server_ms is not None:
                    server.append(server_ms)
        if not record:
            return None
        out = {"latency_ms": summarize(latencies), "errors": dict(errors)}
        if server:
            out["server_took_ms"] = summarize(server)
        return out

    def _run_concurrent(self, target):
        n = len(self.queries)
        latencies, server = [], []
        errors, throttled = Counter(), [0]
        sink_lock = threading.Lock()
        start = time.perf_counter()
        deadline = start + self.duration_s

        def worker(index):
            call = target.new_worker()
            i = index % n
            mine_lat, mine_srv, mine_err, mine_thr = [], [], Counter(), 0
            while time.perf_counter() < deadline and not self.stop_event.is_set():
                text, vector = self.queries[i]
                i = (i + 1) % n
                t0 = time.perf_counter()
                try:
                    server_ms = call(text, vector)
                except CallError as e:
                    mine_err[e.kind] += 1
                    mine_thr += int(e.throttled)
                    self._tick()
                    continue
                mine_lat.append((time.perf_counter() - t0) * 1000.0)
                if server_ms is not None:
                    mine_srv.append(server_ms)
                self._tick()
            with sink_lock:
                latencies.extend(mine_lat)
                server.extend(mine_srv)
                errors.update(mine_err)
                throttled[0] += mine_thr

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(self.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        ok = len(latencies)
        total_err = sum(errors.values())
        out = {
            "concurrency": self.concurrency,
            "elapsed_s": round(elapsed, 2),
            "requests_ok": ok,
            "requests_failed": total_err,
            "throttled": throttled[0],
            "error_rate_pct": round(100.0 * total_err / (ok + total_err), 2) if ok + total_err else 0.0,
            "qps": round(ok / elapsed, 2) if elapsed > 0 else 0.0,
            "latency_ms": summarize(latencies),
            "errors": dict(errors),
        }
        if server:
            out["server_took_ms"] = summarize(server)
        return out

    # -- driver --------------------------------------------------------------
    def run(self):
        try:
            for target in self.targets:
                res = self.results[target.key]
                if self.warmup and not self.stop_event.is_set():
                    self._set_phase(f"{target.label}: warmup")
                    self._run_serial(target, 1, record=False)
                if not self.stop_event.is_set():
                    self._set_phase(f"{target.label}: serial")
                    serial = self._run_serial(target, self.serial_rounds, record=True)
                    with self.lock:
                        res["serial"] = serial
                if not self.stop_event.is_set():
                    self._set_phase(f"{target.label}: concurrent x{self.concurrency}")
                    conc = self._run_concurrent(target)
                    with self.lock:
                        res["concurrent"] = conc
            with self.lock:
                self.state = "stopped" if self.stop_event.is_set() else "finished"
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.state = "failed"
                self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.phase = None
                self.finished = time.time()

    def start(self):
        threading.Thread(target=self.run, daemon=True, name=f"bench-{self.id}").start()

    def stop(self):
        self.stop_event.set()
