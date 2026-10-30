#!/usr/bin/env python3
"""
Hybrid Search comparison WebUI: S3 Vectors (us-east-2) vs OpenSearch (us-east-1).

Flow for each query:
  1. Embed the user's text via an OpenSearch ML connector model (Bedrock Cohere).
  2. Feed that SAME embedding into both backends' hybrid queries so the comparison is fair.
  3. Return ranked results + latency from each source.

Latency:
  - OpenSearch: server-side `took` (ms) from the response.
  - S3 Vectors: no server-side timing field in QueryVectors, so we measure the
    client-side wall-clock round-trip and label it accordingly.

Requires the S3 Vectors Hybrid Search *beta* boto3/botocore (1.43.106) — the
public SDK has no queryTexts/returnScores on QueryVectors.
"""

import argparse
import os
import sys
import threading
import time
import traceback

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from flask import Flask, jsonify, render_template, request
from requests.auth import HTTPBasicAuth

import bench

# ---------------------------------------------------------------------------
# Configuration: read from environment variables (see .env.example).
# CLI flags override everything except the password, which is env-only so it
# never appears in the process list.
# ---------------------------------------------------------------------------
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER", "admin")
OPENSEARCH_PASS = os.environ.get("OPENSEARCH_PASSWORD", "")
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "")
OPENSEARCH_PIPELINE = os.environ.get("OPENSEARCH_SEARCH_PIPELINE", "")
ML_MODEL_ID = os.environ.get("ML_MODEL_ID", "")
EMBED_INPUT_TYPE = os.environ.get("EMBED_INPUT_TYPE", "search_query")

S3V_REGION = os.environ.get("S3V_REGION", "us-east-2")
S3V_BUCKET = os.environ.get("S3V_BUCKET", "")
S3V_INDEX = os.environ.get("S3V_INDEX", "")
S3V_TEXT_KEY = os.environ.get("S3V_TEXT_KEY", "passage_body")

VECTOR_DIM = 1024

# OpenSearch fusion pipelines selectable from the UI (whitelist; first = default).
PIPELINES = {
    "rrf-hybrid-pipeline": {
        "label": "RRF (rank_constant=60)",
    },
    "nlp-search-pipeline": {
        "label": "min_max + weighted mean",
    },
}
DEFAULT_PIPELINE = "rrf-hybrid-pipeline"
# For this pipeline the UI supplies the BM25/kNN weights; they are sent as a
# temporary (request-body) pipeline so the stored pipeline is never modified.
WEIGHTED_PIPELINE = "nlp-search-pipeline"


def weighted_pipeline_body(bm25_weight):
    """min_max + arithmetic_mean with weights [BM25, kNN] (hybrid query order)."""
    return {
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [bm25_weight, round(1.0 - bm25_weight, 4)]},
                    },
                }
            }
        ]
    }

app = Flask(__name__)
_cfg = {}


# ---------------------------------------------------------------------------
# Embedding via OpenSearch ML connector
# ---------------------------------------------------------------------------
def embed(text):
    """Return (vector, latency_ms) using the OpenSearch ML model predict API."""
    url = f"{_cfg['os_endpoint']}/_plugins/_ml/models/{_cfg['ml_model_id']}/_predict"
    t0 = time.perf_counter()
    resp = requests.post(
        url,
        json={"parameters": {"texts": [text], "input_type": _cfg["embed_input_type"]}},
        auth=HTTPBasicAuth(_cfg["os_user"], _cfg["os_pass"]),
        timeout=30,
        verify=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    output = data["inference_results"][0]["output"][0]
    vector = output["data"]
    if len(vector) != VECTOR_DIM:
        raise ValueError(f"Embedding dim {len(vector)} != expected {VECTOR_DIM}")
    return [float(x) for x in vector], latency_ms


# ---------------------------------------------------------------------------
# OpenSearch hybrid search (BM25 on text + knn on embedding, RRF-fused)
# ---------------------------------------------------------------------------
def build_os_search(query_text, vector, k, pipeline=None, bm25_weight=None):
    """Return (url, body, params, mode_label) for the OpenSearch hybrid query."""
    pipeline = pipeline or _cfg.get("os_search_pipeline")
    url = f"{_cfg['os_endpoint']}/{_cfg['os_index']}/_search"
    body = {
        "size": k,
        "_source": ["document_id", "title", "text", "url"],
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title", "text"],
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": vector,
                                "k": k,
                            }
                        }
                    },
                ]
            }
        },
    }
    params = {}
    if pipeline == WEIGHTED_PIPELINE and bm25_weight is not None:
        body["search_pipeline"] = weighted_pipeline_body(bm25_weight)
        mode_label = (f"min_max + weighted mean (BM25 {bm25_weight:g} / "
                      f"kNN {round(1.0 - bm25_weight, 4):g}, per-request)")
    else:
        if pipeline:
            params["search_pipeline"] = pipeline
        mode_label = PIPELINES.get(pipeline, {}).get("label", pipeline or "no pipeline")
    return url, body, params, mode_label


def search_opensearch(query_text, vector, k, pipeline=None, bm25_weight=None):
    """
    Hybrid query combining BM25 full-text and kNN vector search, fused by the
    given search pipeline (RRF or min_max weighted). Falls back to a bool
    should (BM25 + knn) if the hybrid query is rejected.
    OpenSearch returns `took` in ms (server-side).
    """
    pipeline = pipeline or _cfg.get("os_search_pipeline")
    url, body, params, mode_label = build_os_search(query_text, vector, k, pipeline, bm25_weight)

    t0 = time.perf_counter()
    resp = requests.post(
        url,
        json=body,
        params=params,
        auth=HTTPBasicAuth(_cfg["os_user"], _cfg["os_pass"]),
        timeout=60,
        verify=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if resp.status_code >= 400:
        # Hybrid query requires a normalization pipeline; fall back to bool should.
        return _search_opensearch_fallback(query_text, vector, k, note=resp.text[:300])

    data = resp.json()
    took = data.get("took")
    hits = data.get("hits", {}).get("hits", [])
    results = []
    for h in hits:
        src = h.get("_source", {})
        results.append(
            {
                "id": src.get("document_id") or h.get("_id"),
                "title": src.get("title", ""),
                "text": (src.get("text", "") or "")[:400],
                "url": src.get("url", ""),
                "score": h.get("_score"),
            }
        )
    return {
        "ok": True,
        "took_ms": took,
        "wall_ms": round(wall_ms, 1),
        "latency_source": "server (took)",
        "count": len(results),
        "results": results,
        "mode": f"hybrid (BM25 + kNN) · {mode_label}",
        "pipeline": pipeline,
    }


def _search_opensearch_fallback(query_text, vector, k, note=""):
    """Bool should: BM25 over title/text + knn on embedding (score-summed)."""
    url = f"{_cfg['os_endpoint']}/{_cfg['os_index']}/_search"
    body = {
        "size": k,
        "_source": ["document_id", "title", "text", "url"],
        "query": {
            "bool": {
                "should": [
                    {"multi_match": {"query": query_text, "fields": ["title", "text"]}},
                    {"knn": {"embedding": {"vector": vector, "k": k}}},
                ]
            }
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(
        url,
        json=body,
        auth=HTTPBasicAuth(_cfg["os_user"], _cfg["os_pass"]),
        timeout=60,
        verify=False,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])
    results = []
    for h in hits:
        src = h.get("_source", {})
        results.append(
            {
                "id": src.get("document_id") or h.get("_id"),
                "title": src.get("title", ""),
                "text": (src.get("text", "") or "")[:400],
                "url": src.get("url", ""),
                "score": h.get("_score"),
            }
        )
    return {
        "ok": True,
        "took_ms": data.get("took"),
        "wall_ms": round(wall_ms, 1),
        "latency_source": "server (took)",
        "count": len(results),
        "results": results,
        "mode": "bool should (BM25 + kNN, score-summed) — no hybrid pipeline",
        "note": "Hybrid pipeline unavailable; used bool fallback. " + note,
    }


# ---------------------------------------------------------------------------
# S3 Vectors hybrid search (vector + queryTexts, server-side RRF)
# ---------------------------------------------------------------------------
_s3v_client = None


def s3v_client():
    global _s3v_client
    if _s3v_client is None:
        _s3v_client = boto3.client(
            "s3vectors",
            region_name=_cfg["s3v_region"],
            config=Config(retries={"max_attempts": 2, "mode": "standard"}),
        )
    return _s3v_client


def build_s3v_query(query_text, vector, k):
    """Keyword arguments for a hybrid QueryVectors call."""
    return {
        "vectorBucketName": _cfg["s3v_bucket"],
        "indexName": _cfg["s3v_index"],
        "topK": k,
        "queryVector": {"float32": vector},
        "queryTexts": [{"text": query_text, "metadataKey": _cfg["s3v_text_key"]}],
        "returnMetadata": True,
        "returnDistance": True,
        "returnScores": True,
    }


def search_s3vectors(query_text, vector, k):
    """
    Hybrid QueryVectors: query vector + queryTexts (BM25), fused server-side via RRF.
    QueryVectors returns NO server-side timing, so latency is client-side wall-clock.
    """
    client = s3v_client()
    kwargs = build_s3v_query(query_text, vector, k)
    t0 = time.perf_counter()
    try:
        resp = client.query_vectors(**kwargs)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "mode": "hybrid (vector + BM25, server-side RRF)",
        }
    wall_ms = (time.perf_counter() - t0) * 1000.0

    results = []
    for v in resp.get("vectors", []):
        md = v.get("metadata", {}) or {}
        sc = v.get("scores", {}) or {}
        bm25 = None
        bm25_list = sc.get("bm25Scores") or []
        if bm25_list:
            bm25 = bm25_list[0].get("score")
        results.append(
            {
                "id": v.get("key"),
                "title": md.get("title", ""),
                "text": (md.get(_cfg["s3v_text_key"], "") or "")[:400],
                "url": md.get("url", ""),
                # fused RRF score is the ranking score; fall back to bm25 for text-only
                "score": sc.get("fused"),
                "bm25": bm25,
                "distance": sc.get("vectorDistance", v.get("distance")),
            }
        )
    return {
        "ok": True,
        "took_ms": None,
        "wall_ms": round(wall_ms, 1),
        "latency_source": "client round-trip (no server timing)",
        "count": len(results),
        "results": results,
        "mode": "hybrid (vector + BM25, server-side RRF)",
    }


# ---------------------------------------------------------------------------
# Load test (see bench.py)
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "why do leaves change color in autumn",
    "how do vaccines train the immune system",
    "the economic causes of the Great Depression",
    "animals that can survive extreme cold",
    "TCP three-way handshake SYN ACK",
    "Apollo 13 oxygen tank explosion",
    "CRISPR Cas9 gene editing",
    "Treaty of Westphalia 1648",
    "who painted the ceiling of the Sistine Chapel",
    "difference between nuclear fission and fusion",
]
BENCH_LIMITS = {
    "concurrency": (1, 64),
    "duration_s": (5, 600),
    "serial_rounds": (1, 10),
    "queries": (1, 50),
}
S3V_THROTTLE_CODES = {"TooManyRequestsException", "ThrottlingException", "SlowDown",
                      "ServiceUnavailableException"}

_bench_lock = threading.Lock()
_bench_current = None
_embed_cache = {}


def cached_embed(text):
    if text not in _embed_cache:
        _embed_cache[text] = embed(text)[0]
    return _embed_cache[text]


def opensearch_target(k, pipeline, bm25_weight, pool_size):
    def new_worker():
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=pool_size)
        session.mount("https://", adapter)
        session.auth = HTTPBasicAuth(_cfg["os_user"], _cfg["os_pass"])
        session.verify = False

        def call(text, vector):
            url, body, params, _ = build_os_search(text, vector, k, pipeline, bm25_weight)
            try:
                resp = session.post(url, json=body, params=params, timeout=60)
            except requests.Timeout:
                raise bench.CallError("timeout")
            except requests.RequestException as e:
                raise bench.CallError(type(e).__name__)
            if resp.status_code >= 400:
                raise bench.CallError(f"HTTP {resp.status_code}",
                                      throttled=resp.status_code == 429)
            return resp.json().get("took")
        return call

    return bench.Target("opensearch", "OpenSearch", new_worker)


def s3vectors_target(k, pool_size):
    # One client shared by all threads (boto3 clients are thread-safe). Retries
    # are disabled so throttling shows up as errors instead of hidden latency.
    client = boto3.client(
        "s3vectors",
        region_name=_cfg["s3v_region"],
        config=Config(retries={"max_attempts": 1, "mode": "standard"},
                      max_pool_connections=pool_size, read_timeout=60),
    )

    def new_worker():
        def call(text, vector):
            try:
                client.query_vectors(**build_s3v_query(text, vector, k))
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "ClientError")
                status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                raise bench.CallError(code, throttled=code in S3V_THROTTLE_CODES or status == 429)
            except BotoCoreError as e:
                raise bench.CallError(type(e).__name__)
            return None
        return call

    return bench.Target("s3vectors", "S3 Vectors", new_worker)


def _int_in(payload, name, default):
    low, high = BENCH_LIMITS[name]
    value = int(payload.get(name, default))
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


@app.route("/bench")
def bench_page():
    return render_template(
        "bench.html",
        pipelines=PIPELINES,
        default_pipeline=_cfg.get("os_search_pipeline") or DEFAULT_PIPELINE,
        sample_queries=SAMPLE_QUERIES,
        limits=BENCH_LIMITS,
    )


@app.route("/api/bench/start", methods=["POST"])
def bench_start():
    global _bench_current
    payload = request.get_json(force=True)
    try:
        concurrency = _int_in(payload, "concurrency", 4)
        duration_s = _int_in(payload, "duration_s", 60)
        serial_rounds = _int_in(payload, "serial_rounds", 3)
        k = max(1, min(int(payload.get("k", 10)), 100))
        queries = [q.strip() for q in payload.get("queries") or SAMPLE_QUERIES if q and q.strip()]
        low, high = BENCH_LIMITS["queries"]
        if not low <= len(queries) <= high:
            raise ValueError(f"provide between {low} and {high} queries")
        if any(len(q.encode("utf-8")) > 1000 for q in queries):
            raise ValueError("each query must be at most 1000 bytes")
        pipeline = payload.get("pipeline") or _cfg.get("os_search_pipeline") or DEFAULT_PIPELINE
        if pipeline not in PIPELINES:
            raise ValueError(f"unknown pipeline: {pipeline}")
        bm25_weight = None
        if pipeline == WEIGHTED_PIPELINE and payload.get("bm25_weight") is not None:
            bm25_weight = round(float(payload["bm25_weight"]), 4)
            if not 0.0 <= bm25_weight <= 1.0:
                raise ValueError("bm25_weight must be between 0 and 1")
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    with _bench_lock:
        if _bench_current and _bench_current.state == "running":
            return jsonify({"error": "a load test is already running", "id": _bench_current.id}), 409
        try:
            vectors = [(q, cached_embed(q)) for q in queries]
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"embedding failed: {type(e).__name__}: {e}"}), 502
        pool = concurrency + 2
        settings = {
            "concurrency": concurrency, "duration_s": duration_s,
            "serial_rounds": serial_rounds, "k": k, "queries": len(queries),
            "pipeline": pipeline, "bm25_weight": bm25_weight,
            "warmup": bool(payload.get("warmup", True)),
        }
        _bench_current = bench.Benchmark(
            targets=[opensearch_target(k, pipeline, bm25_weight, pool), s3vectors_target(k, pool)],
            queries=vectors,
            concurrency=concurrency,
            duration_s=duration_s,
            serial_rounds=serial_rounds,
            warmup=settings["warmup"],
            settings=settings,
        )
        _bench_current.start()
        return jsonify({"id": _bench_current.id})


@app.route("/api/bench/<run_id>")
def bench_status(run_id):
    run = _bench_current
    if not run or run.id != run_id:
        return jsonify({"error": "unknown run"}), 404
    return jsonify(run.status())


@app.route("/api/bench/<run_id>/stop", methods=["POST"])
def bench_stop(run_id):
    run = _bench_current
    if not run or run.id != run_id:
        return jsonify({"error": "unknown run"}), 404
    run.stop()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        cfg=_cfg,
        pipelines=PIPELINES,
        default_pipeline=_cfg.get("os_search_pipeline") or DEFAULT_PIPELINE,
    )


@app.route("/api/search", methods=["POST"])
def api_search():
    payload = request.get_json(force=True)
    query_text = (payload.get("q") or "").strip()
    k = int(payload.get("k", 10))
    k = max(1, min(k, 100))
    if not query_text:
        return jsonify({"error": "empty query"}), 400
    pipeline = payload.get("pipeline") or _cfg.get("os_search_pipeline") or DEFAULT_PIPELINE
    if pipeline not in PIPELINES:
        return jsonify({"error": f"unknown pipeline: {pipeline}"}), 400
    bm25_weight = None
    if pipeline == WEIGHTED_PIPELINE and payload.get("bm25_weight") is not None:
        try:
            bm25_weight = round(float(payload["bm25_weight"]), 4)
        except (TypeError, ValueError):
            return jsonify({"error": "bm25_weight must be a number"}), 400
        if not 0.0 <= bm25_weight <= 1.0:
            return jsonify({"error": "bm25_weight must be between 0 and 1"}), 400

    out = {"query": query_text, "k": k, "pipeline": pipeline, "bm25_weight": bm25_weight}

    # 1. Embed once, share with both backends.
    try:
        vector, embed_ms = embed(query_text)
        out["embed_ms"] = round(embed_ms, 1)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"embedding failed: {type(e).__name__}: {e}"}), 502

    # 2. Query both sources.
    try:
        out["opensearch"] = search_opensearch(query_text, vector, k, pipeline, bm25_weight)
    except Exception as e:  # noqa: BLE001
        out["opensearch"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-500:]}

    out["s3vectors"] = search_s3vectors(query_text, vector, k)

    return jsonify(out)


@app.route("/api/health")
def health():
    status = {"opensearch": {}, "s3vectors": {}}
    try:
        r = requests.get(
            f"{_cfg['os_endpoint']}/{_cfg['os_index']}/_count",
            auth=HTTPBasicAuth(_cfg["os_user"], _cfg["os_pass"]),
            timeout=10, verify=False,
        )
        status["opensearch"] = {"ok": r.ok, "count": r.json().get("count")}
    except Exception as e:  # noqa: BLE001
        status["opensearch"] = {"ok": False, "error": str(e)}
    try:
        client = s3v_client()
        client.get_index(vectorBucketName=_cfg["s3v_bucket"], indexName=_cfg["s3v_index"])
        status["s3vectors"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        status["s3vectors"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return jsonify(status)


def build_cfg(a):
    return {
        "os_endpoint": a.os_endpoint.rstrip("/"),
        "os_user": a.os_user,
        "os_pass": OPENSEARCH_PASS,
        "os_index": a.os_index,
        "os_search_pipeline": a.os_search_pipeline,
        "ml_model_id": a.ml_model_id,
        "embed_input_type": a.embed_input_type,
        "s3v_region": a.s3v_region,
        "s3v_bucket": a.s3v_bucket,
        "s3v_index": a.s3v_index,
        "s3v_text_key": a.s3v_text_key,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--os-endpoint", default=OPENSEARCH_ENDPOINT)
    p.add_argument("--os-user", default=OPENSEARCH_USER)
    p.add_argument("--os-index", default=OPENSEARCH_INDEX)
    p.add_argument("--os-search-pipeline", default=OPENSEARCH_PIPELINE)
    p.add_argument("--ml-model-id", default=ML_MODEL_ID)
    p.add_argument("--embed-input-type", default=EMBED_INPUT_TYPE)
    p.add_argument("--s3v-region", default=S3V_REGION)
    p.add_argument("--s3v-bucket", default=S3V_BUCKET)
    p.add_argument("--s3v-index", default=S3V_INDEX)
    p.add_argument("--s3v-text-key", default=S3V_TEXT_KEY)
    a = p.parse_args()

    global _cfg
    _cfg = build_cfg(a)

    required = {
        "OPENSEARCH_ENDPOINT": _cfg["os_endpoint"],
        "OPENSEARCH_PASSWORD": _cfg["os_pass"],
        "OPENSEARCH_INDEX": _cfg["os_index"],
        "ML_MODEL_ID": _cfg["ml_model_id"],
        "S3V_BUCKET": _cfg["s3v_bucket"],
        "S3V_INDEX": _cfg["s3v_index"],
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        sys.exit(f"Missing required settings: {', '.join(missing)} (see .env.example)")
    if _cfg["os_search_pipeline"] and _cfg["os_search_pipeline"] not in PIPELINES:
        sys.exit(f"OPENSEARCH_SEARCH_PIPELINE must be one of: {', '.join(PIPELINES)}")

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print(f"Serving on http://{a.host}:{a.port}  (OpenSearch={_cfg['os_index']}, "
          f"S3Vectors={_cfg['s3v_bucket']}/{_cfg['s3v_index']})")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
