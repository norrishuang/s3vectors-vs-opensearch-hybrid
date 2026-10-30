# s3vectors-vs-opensearch-hybrid

A small web UI for comparing **hybrid search** (full-text + vector) on
**Amazon S3 Vectors** and **Amazon OpenSearch Service**, side by side, on the
same dataset.

Type a query once. The app embeds it once, sends the same query text and the
same query vector to both backends, and shows the two ranked result lists with
their latency next to each other.

## How it works

```
                       ┌──────────────────────────────┐
  query text ──► Flask │ 1. embed via OpenSearch ML   │──► Bedrock Cohere
                       │    model (_predict)          │
                       │ 2. same text + same vector   │
                       └───────┬──────────────┬───────┘
                               │              │
              OpenSearch hybrid query    S3 Vectors QueryVectors
              (BM25 + k-NN, fused by a   (queryVector + queryTexts,
               search pipeline)           fused server-side with RRF)
```

1. **Embedding.** The query text is embedded once through an OpenSearch ML
   Commons model backed by a Bedrock Cohere connector. The app passes
   `input_type=search_query` on each `_predict` call, overriding the
   connector's default.
2. **OpenSearch.** A `hybrid` query combines `multi_match` (BM25 on `title`
   and `text`) with a `knn` query on `embedding`. The results are fused by a
   search pipeline that you pick in the UI:
   - `rrf-hybrid-pipeline`: Reciprocal Rank Fusion, `rank_constant` 60.
   - `nlp-search-pipeline` ("min_max + weighted mean"): min_max normalization
     and a weighted arithmetic mean of BM25 and k-NN. When you pick this option,
     a slider sets the weights (default BM25 0.5 / k-NN 0.5). The app sends the
     weights with each query as a temporary pipeline in the request body, so
     the pipeline stored on the cluster is never changed.
3. **S3 Vectors.** A single `QueryVectors` call with both `queryVector` and
   `queryTexts`. The service runs vector search and BM25 search and fuses them
   with RRF. Each hit has `scores.fused`, `scores.bm25Scores` and
   `scores.vectorDistance`, and the UI shows all three.

### Overlap marks

Results are matched across the two lists by document ID (one Wikipedia passage
per ID):

- **Green**: the document is at the same rank in both lists.
- **Orange**: the document is in both lists at different ranks. Its tag shows
  the rank on the other side, for example `S3 Vectors #4`.
- **Unmarked**: the document is only in this list.

The line above the results shows how many documents are in both lists and how
many share the same rank.

### Latency

| Backend | What is shown |
|---|---|
| OpenSearch | Server-side `took` from the search response |
| S3 Vectors | Client-side round-trip time of the `QueryVectors` call. The API returns no server-side timing, so this number includes network time. |

The embedding time is shown separately and is not included in either number.

## Requirements

- Python 3.9+
- An OpenSearch Service domain (2.19+ for the RRF `score-ranker-processor`;
  developed against 3.7) with the neural-search and ML Commons plugins, an
  index with a `knn_vector` field, and a deployed Bedrock Cohere embedding model.
- An S3 Vectors vector index with a searchable text field
  (`textSearchConfigurations`) in `us-east-2`.
- The **S3 Vectors Hybrid Search beta SDK** (boto3/botocore wheels). The public
  boto3 has no `queryTexts` or `returnScores` on `QueryVectors`. The beta SDK is
  available to enrolled beta participants through their AWS account team and is
  not included in this repository.
- AWS credentials with `s3vectors:QueryVectors` and `s3vectors:GetIndex` on
  the vector index.

The query embedding model must be the same model that produced the stored
document vectors. The demo dataset (Cohere Wikipedia embeddings) was embedded
with `cohere.embed-multilingual-v3`. Querying it with
`cohere.embed-english-v3` gives near-random vector results.

## Setup

```bash
git clone https://github.com/norrishuang/s3vectors-vs-opensearch-hybrid.git
cd s3vectors-vs-opensearch-hybrid

python3 -m venv .venv
. .venv/bin/activate
pip install /path/to/hybrid-pythonsdk/botocore-*.whl /path/to/hybrid-pythonsdk/boto3-*.whl
pip install -r requirements.txt

cp .env.example .env    # then fill in your endpoints, password and IDs
```

Create the OpenSearch search pipelines (the definitions are in `opensearch/`):

```bash
curl -u "admin:$OPENSEARCH_PASSWORD" -XPUT "$OPENSEARCH_ENDPOINT/_search/pipeline/rrf-hybrid-pipeline" \
  -H 'Content-Type: application/json' -d @opensearch/rrf-hybrid-pipeline.json
curl -u "admin:$OPENSEARCH_PASSWORD" -XPUT "$OPENSEARCH_ENDPOINT/_search/pipeline/nlp-search-pipeline" \
  -H 'Content-Type: application/json' -d @opensearch/nlp-search-pipeline.json
```

## Configuration

All settings come from environment variables, which `run.sh` and `restart.sh`
load from `.env`. See `.env.example`.

| Variable | Description |
|---|---|
| `OPENSEARCH_ENDPOINT` | HTTPS endpoint of the domain |
| `OPENSEARCH_USER` / `OPENSEARCH_PASSWORD` | Basic-auth user (fine-grained access control) |
| `OPENSEARCH_INDEX` | Index with `title`, `text` and an `embedding` knn_vector field |
| `OPENSEARCH_SEARCH_PIPELINE` | Default fusion pipeline (`rrf-hybrid-pipeline` or `nlp-search-pipeline`) |
| `ML_MODEL_ID` | Deployed ML Commons model used to embed the query |
| `EMBED_INPUT_TYPE` | Cohere `input_type` for the query, default `search_query` |
| `S3V_REGION` | S3 Vectors region, default `us-east-2` |
| `S3V_BUCKET` / `S3V_INDEX` | Vector bucket and index |
| `S3V_TEXT_KEY` | Searchable metadata key, default `passage_body` |

Non-secret settings can also be passed as command-line flags; run
`python app.py --help`. The password is read from the environment only.

## Run

```bash
./run.sh 8077          # foreground
./restart.sh 8077      # background, writes app.pid and server.log
```

Open <http://127.0.0.1:8077>. The server binds to `127.0.0.1` only. To reach
it from your laptop, use an SSH tunnel:

```bash
ssh -L 8077:127.0.0.1:8077 ec2-user@<host>
```

### API

```bash
curl -s -XPOST http://127.0.0.1:8077/api/search -H 'Content-Type: application/json' \
  -d '{"q": "who invented the telephone", "k": 10, "pipeline": "rrf-hybrid-pipeline"}'

# min_max + weighted mean with BM25 0.3 / k-NN 0.7 (bm25_weight is 0-1; k-NN gets the rest)
curl -s -XPOST http://127.0.0.1:8077/api/search -H 'Content-Type: application/json' \
  -d '{"q": "who invented the telephone", "k": 10, "pipeline": "nlp-search-pipeline", "bm25_weight": 0.3}'

curl -s http://127.0.0.1:8077/api/health
```

## Loading the data

The `ingestion/` directory contains the EMR Serverless jobs that load the
English subset of the
[CohereLabs/wikipedia-2023-11-embed-multilingual-v3](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)
dataset (about 41.5 million passages, 1024-dim vectors) into both backends.
Copy the dataset's Parquet files into your own S3 bucket first; the guide
shows how. Both jobs use the source `_id`
as the record key, so a re-run overwrites records instead of duplicating them.

| Target | Code | What it does |
|---|---|---|
| OpenSearch | `ingestion/opensearch/` | `create_cohere_vector_index.py` creates a disk-mode k-NN index (cosine, 32x compression) with BM25 `title`/`text` fields. `emr_serverless_cohere_to_opensearch.py` writes the documents with the OpenSearch Spark connector, using basic auth with the password read from Secrets Manager. |
| S3 Vectors | `ingestion/s3vectors/` | `create_hybrid_index.py` creates a cosine index with `passage_body` as the searchable text field. `ingest_cohere_to_s3vectors.py` writes the vectors with `PutVectors` in batches of up to 500, retrying throttled requests. `package_beta_sdk.sh` packages the beta SDK for Spark, and `query_hybrid_index.py` runs a test query. |

Step-by-step instructions, including IAM permissions, job parameters and
troubleshooting, are in
[docs/ingestion-emr-serverless.md](docs/ingestion-emr-serverless.md).

## Load test

Open <http://127.0.0.1:8077/bench> (the **Load test** link in the header) to
measure query throughput and latency. Only search queries are sent; nothing is
written to either backend.

Settings: concurrency (1–64 Python threads), duration per backend (5–600 s),
serial rounds, topK, OpenSearch fusion (with the BM25/k-NN slider for min_max),
and the query list, which defaults to the 10 sample queries below and is used
round-robin.

The backends are tested one after the other, OpenSearch first, then S3
Vectors. For each backend:

1. **Warmup**: every query once, not measured (optional).
2. **Serial**: one thread runs every query *serial rounds* times. Reported as
   the average latency.
3. **Concurrent**: N threads send queries back to back for the set duration.
   Reported as QPS (successful requests per second), p50/p90/p95/p99 latency,
   and counts of successful, failed and throttled requests.

How it is measured:

- Each query is embedded once before the test starts, so embedding time is
  not part of any number.
- Latency is client-side wall-clock time per search call, including the
  network round trip, measured the same way for both backends. OpenSearch's
  server-side `took` is also shown, for reference.
- Each OpenSearch thread uses its own HTTP keep-alive session. S3 Vectors
  threads share one boto3 client sized for the thread count.
- S3 Vectors calls are made without SDK retries, so throttling
  (`TooManyRequestsException`) is reported as failed requests instead of
  appearing as extra latency. The beta has a per-index request-rate limit, so
  throttling at higher concurrency is expected.
- Only one load test runs at a time; starting a second returns HTTP 409.

When a test finishes or is stopped, **Export to Excel** downloads an `.xlsx`
report with four sheets: *Summary* (settings and the results table),
*Latency details* (count, avg, min, p50 to p99 and max for each phase, plus
OpenSearch `took`), *Errors* (error codes and counts) and *Queries*.

The same test is available over HTTP:

```bash
curl -s -XPOST http://127.0.0.1:8077/api/bench/start -H 'Content-Type: application/json' \
  -d '{"concurrency": 8, "duration_s": 60, "serial_rounds": 3, "k": 10, "pipeline": "rrf-hybrid-pipeline"}'
# -> {"id": "<run-id>"}
curl -s http://127.0.0.1:8077/api/bench/<run-id>          # progress and results
curl -s -XPOST http://127.0.0.1:8077/api/bench/<run-id>/stop
curl -s -o report.xlsx http://127.0.0.1:8077/api/bench/<run-id>/export.xlsx   # after it finishes
```

Run the load test from a host close to both backends. The OpenSearch domain
and the S3 Vectors index in the demo are in different regions, so network
distance affects the numbers; the gap between client latency and OpenSearch
`took` shows how much.

## Sample queries

The demo dataset is English Wikipedia. These queries are grouped by which half
of hybrid search should matter most, which makes ranking differences between
the backends and fusion methods easier to see.

**Semantic** (vector search should lead; the query words may not appear in the text)

1. `why do leaves change color in autumn`
2. `how do vaccines train the immune system`
3. `the economic causes of the Great Depression`
4. `animals that can survive extreme cold`

**Exact terms** (BM25 should lead; proper nouns, codes and acronyms)

5. `TCP three-way handshake SYN ACK`
6. `Apollo 13 oxygen tank explosion`
7. `CRISPR Cas9 gene editing`
8. `Treaty of Westphalia 1648`

**Mixed** (both meaning and keywords matter)

9. `who painted the ceiling of the Sistine Chapel`
10. `difference between nuclear fission and fusion`

Run each query with both OpenSearch fusion options. Queries 5 to 8 show the
clearest difference between RRF, which weights BM25 and vector results equally
by rank, and min_max weighting, which depends on the slider.

## Comparing fairly

- Use `rrf-hybrid-pipeline` on OpenSearch so both sides use the same fusion
  method. With `rank_constant` 60, the top S3 Vectors fused score is 1/60
  (0.0167), which suggests it uses the same constant.
- The full-text halves search different fields. OpenSearch matches on `title`
  and `text`; S3 Vectors searches only `passage_body`, which holds the text
  without the title. This affects BM25 ranking.
- The demo OpenSearch index stores vectors with `mode: on_disk` and 32x
  compression, so the first queries can be slow while data is read from disk.

## Project layout

```
app.py                       Flask backend: embedding, both searches, API
bench.py                     Load test engine (threads, percentiles)
export_xlsx.py               Excel report for a load test run
templates/index.html         Search comparison page
templates/bench.html         Load test page
opensearch/*.json            Search pipeline definitions
run.sh / restart.sh          Start scripts (load .env)
.env.example                 Configuration template
ingestion/opensearch/        EMR Serverless job: Cohere Parquet -> OpenSearch
ingestion/s3vectors/         EMR Serverless job: Cohere Parquet -> S3 Vectors
docs/                        Ingestion guide
```

## License

[MIT](LICENSE)
