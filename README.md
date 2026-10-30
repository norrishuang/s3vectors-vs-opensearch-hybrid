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
templates/index.html         Single-page UI
opensearch/*.json            Search pipeline definitions
run.sh / restart.sh          Start scripts (load .env)
.env.example                 Configuration template
```

## License

[MIT](LICENSE)
