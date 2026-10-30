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
import time
import traceback

import boto3
import requests
from botocore.config import Config
from flask import Flask, jsonify, render_template, request
from requests.auth import HTTPBasicAuth

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
def search_opensearch(query_text, vector, k, pipeline=None, bm25_weight=None):
    """
    Hybrid query combining BM25 full-text and kNN vector search, fused by the
    given search pipeline (RRF or min_max weighted). Falls back to a bool
    should (BM25 + knn) if the hybrid query is rejected.
    OpenSearch returns `took` in ms (server-side).
    """
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


def search_s3vectors(query_text, vector, k):
    """
    Hybrid QueryVectors: query vector + queryTexts (BM25), fused server-side via RRF.
    QueryVectors returns NO server-side timing, so latency is client-side wall-clock.
    """
    client = s3v_client()
    kwargs = {
        "vectorBucketName": _cfg["s3v_bucket"],
        "indexName": _cfg["s3v_index"],
        "topK": k,
        "queryVector": {"float32": vector},
        "queryTexts": [{"text": query_text, "metadataKey": _cfg["s3v_text_key"]}],
        "returnMetadata": True,
        "returnDistance": True,
        "returnScores": True,
    }
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
