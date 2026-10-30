# Loading the Cohere Wikipedia embeddings with EMR Serverless

This guide loads the same Cohere Wikipedia embedding dataset into both backends
used by the comparison UI:

- an **Amazon OpenSearch Service** k-NN index, and
- an **Amazon S3 Vectors** hybrid-search index (beta, `us-east-2`).

Both jobs are PySpark scripts run on **EMR Serverless**. The code is in
`ingestion/opensearch/` and `ingestion/s3vectors/`.

## Contents

1. [Dataset](#1-dataset)
2. [Prerequisites](#2-prerequisites)
3. [Load into OpenSearch](#3-load-into-opensearch)
4. [Load into S3 Vectors](#4-load-into-s3-vectors)
5. [Querying the loaded data](#5-querying-the-loaded-data)
6. [Troubleshooting](#6-troubleshooting)

## 1. Dataset

The jobs load the English subset of
[CohereLabs/wikipedia-2023-11-embed-multilingual-v3](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)
on Hugging Face: Wikipedia (November 2023) split into passages, each with a
precomputed Cohere embedding. The English subset is in the dataset's `en/`
folder: 415 Parquet files, about 90 GB, 41,488,110 rows. The dataset is not
gated, so no Hugging Face token is needed.

### 1.1 Copy the dataset into your own S3 bucket

The jobs read from an S3 prefix you own, which you pass as `S3_INPUT`. Copy
the files once, preferably from an EC2 instance in the same region as the
bucket.

**Option A: copy one file at a time (needs under 1 GB of local disk).**

```bash
set -euo pipefail
DEST=s3://<your-bucket>/cohere-wikipedia/en
REPO=CohereLabs/wikipedia-2023-11-embed-multilingual-v3
TMP="$(mktemp -d)"

curl -sf "https://huggingface.co/api/datasets/$REPO/tree/main/en" \
  | python3 -c 'import sys,json; [print(f["path"]) for f in json.load(sys.stdin) if f["path"].endswith(".parquet")]' \
  > files.txt
wc -l files.txt          # expect 415

while read -r path; do
  name="$(basename "$path")"
  if aws s3 ls "$DEST/$name" >/dev/null 2>&1; then continue; fi   # already copied
  curl -sfL --retry 5 -o "$TMP/$name" "https://huggingface.co/datasets/$REPO/resolve/main/$path"
  aws s3 cp --only-show-errors "$TMP/$name" "$DEST/$name"
  rm -f "$TMP/$name"
done < files.txt
```

Each file is fully downloaded before it is uploaded, so a failed download
never leaves a partial object in S3. If the loop stops, run it again; it skips
files that are already in the bucket.

**Option B: download with the Hugging Face CLI, then sync.** This needs about
100 GB of free disk.

```bash
pip install -U "huggingface_hub[cli]"
hf download CohereLabs/wikipedia-2023-11-embed-multilingual-v3 \
  --repo-type dataset --include "en/*" --local-dir ./cohere-wikipedia
aws s3 sync ./cohere-wikipedia/en/ s3://<your-bucket>/cohere-wikipedia/en/
```

Check the copy:

```bash
aws s3 ls s3://<your-bucket>/cohere-wikipedia/en/ --summarize | tail -2   # Total Objects: 415
```

Then use `S3_INPUT=s3://<your-bucket>/cohere-wikipedia/en/` in the steps
below. To load another language, copy that language's folder instead; the
jobs only require the columns below with 1024-dimensional vectors.

### 1.2 Schema

| Column | Type | Used as |
|---|---|---|
| `_id` | string | Document ID in OpenSearch, vector key in S3 Vectors |
| `url` | string | Stored, not searched |
| `title` | string | Searched by BM25 in OpenSearch |
| `text` | string | Passage text, searched by BM25 on both sides |
| `emb` | array of 1024 floats | The vector |

Both jobs drop rows with a null `_id` or `emb`, and rows whose `emb` does not
have 1024 values. Because each record is keyed by `_id`, running a job again
overwrites records instead of duplicating them.

**Which model produced `emb`.** The stored vectors match
`cohere.embed-multilingual-v3` applied to `"<title> <text>"` (cosine similarity
1.0 when re-embedded). At query time you must embed queries with
`cohere.embed-multilingual-v3` and `input_type=search_query`. Querying with
`cohere.embed-english-v3` returns near-random vector results.

## 2. Prerequisites

### EMR Serverless applications

| Target | Region | Release |
|---|---|---|
| OpenSearch | The OpenSearch domain's region (default `us-east-1`) | A Spark 4.x / Scala 2.13 release. The scripts were run on EMR 8.1. |
| S3 Vectors | `us-east-2` (the only beta region) | Any Spark release with Python 3.8+ |

Each application needs network access to its target. For an OpenSearch domain
inside a VPC, create the application with subnets and security groups that
can reach the domain endpoint on port 443.

The OpenSearch job downloads the OpenSearch Spark connector from Maven when it
starts. In a private subnet without internet egress, add a NAT gateway, point
`OPENSEARCH_SPARK_PACKAGES` at an internal repository, or stage the JARs in S3
and replace `--packages` with `--jars` in the submit script.

### Execution role

The EMR Serverless job role needs:

- `s3:GetObject` and `s3:ListBucket` on your copy of the dataset (section 1.1).
- `s3:GetObject` on the uploaded job script, and on the beta SDK archive for
  the S3 Vectors job.
- `s3:PutObject` on the log prefix.
- **OpenSearch job:** `secretsmanager:GetSecretValue` on the secret that holds
  the OpenSearch password (see 3.1).
- **S3 Vectors job:** `s3vectors:PutVectors` on the target index. The beta
  documentation's example policy grants `s3vectors:*`.

### Local tools

On the machine you submit from: the AWS CLI with credentials that can call
`emr-serverless:StartJobRun`, `iam:PassRole` on the job role, and `s3:PutObject`
on the script prefix. The scripts create the indexes from this machine, so it
also needs Python 3 with `boto3`, and network access to the OpenSearch
endpoint.

## 3. Load into OpenSearch

Files in `ingestion/opensearch/`:

| File | Purpose |
|---|---|
| `create_cohere_vector_index.py` | Creates the target k-NN index (run locally) |
| `emr_serverless_cohere_to_opensearch.py` | The Spark job |
| `submit_cohere_to_opensearch.sh` | Uploads the job script and starts the job run |

### 3.1 Store the OpenSearch password in Secrets Manager

The job authenticates with basic auth (fine-grained access control). The
submit script passes the password to Spark as
`EMR.secret@<OPENSEARCH_SECRET_NAME>`, so EMR Serverless reads it from Secrets
Manager at run time and it never appears in the job definition.

```bash
aws secretsmanager create-secret --region us-east-1 \
  --name opensearch/cohere-ingest \
  --secret-string '<opensearch-password>'
```

The OpenSearch user (default `admin`, set with `OPENSEARCH_USERNAME`) needs
permission to create and write the index.

### 3.2 Create the index

```bash
python3 ingestion/opensearch/create_cohere_vector_index.py \
  --endpoint "https://<domain-endpoint>" \
  --index cohere-wikipedia-en-cosine-v2 \
  --region us-east-1 \
  --shards 12
```

The script signs the request with SigV4 using your local AWS credentials. On a
domain with fine-grained access control, map that IAM identity to an
OpenSearch role that can create indexes.

`--shards` is required on purpose, so that a 41.5M-vector index is not created
with one shard by accident. Choose it from your data-node count and vector
capacity. The index is created with:

| Setting | Value |
|---|---|
| `embedding` | `knn_vector`, dimension 1024, `space_type: cosinesimil`, `mode: on_disk`, `compression_level: 32x` |
| `document_id` | `keyword` |
| `title`, `text` | `text` (BM25) |
| `url` | `keyword`, not indexed, no doc values |
| Replicas | 0 (`--replicas` to change) |
| `refresh_interval` | `-1` during the load |

### 3.3 Submit the job

```bash
export AWS_REGION=us-east-1
export EMR_APPLICATION_ID=<application-id>
export EMR_EXECUTION_ROLE_ARN=arn:aws:iam::<account-id>:role/<emr-job-role>
export JOB_SCRIPT_S3_URI=s3://<artifact-bucket>/scripts/emr_serverless_cohere_to_opensearch.py
export LOG_URI=s3://<artifact-bucket>/logs/cohere-to-opensearch/
export OPENSEARCH_ENDPOINT=https://<domain-endpoint>
export OPENSEARCH_INDEX=cohere-wikipedia-en-cosine-v2
export OPENSEARCH_SECRET_NAME=opensearch/cohere-ingest
export S3_INPUT=s3://<your-bucket>/cohere-wikipedia/en/   # your copy from section 1.1

bash ingestion/opensearch/submit_cohere_to_opensearch.sh
```

Optional variables and their defaults:

| Variable | Default | Meaning |
|---|---|---|
| `OPENSEARCH_USERNAME` | `admin` | Basic-auth user |
| `OPENSEARCH_TARGET` | `aos` | `aoss` for OpenSearch Serverless |
| `OUTPUT_PARTITIONS` | `4` | Maximum concurrent writers (input partitions are coalesced down to this) |
| `EXECUTOR_CORES` / `EXECUTOR_MEMORY` | `1` / `6G` | Executor size |
| `INITIAL_EXECUTORS` / `MIN_EXECUTORS` / `MAX_EXECUTORS` | `2` / `1` / `4` | Dynamic allocation |
| `BATCH_SIZE_ENTRIES` / `BATCH_SIZE_BYTES` | `100` / `1mb` | Bulk request size |
| `WRITE_RETRY_COUNT` / `WRITE_RETRY_WAIT` | `20` / `60s` | Retries for rejected bulk requests |
| `NETWORK_TIMEOUT` | `300s` | Connector request timeout |

The defaults are sized for a small domain (they were tuned against a single
data node). On a larger domain, raise `OUTPUT_PARTITIONS`, `MAX_EXECUTORS` and
the bulk size together, and watch the domain's write thread-pool rejections.

### 3.4 Monitor

```bash
aws emr-serverless get-job-run --region "$AWS_REGION" \
  --application-id "$EMR_APPLICATION_ID" --job-run-id <job-run-id> \
  --query 'jobRun.state'
```

Driver and executor logs are written under `LOG_URI`.

### 3.5 After the load

The job sets `refresh_interval` back to `30s` when it finishes. Then, in
OpenSearch Dashboards Dev Tools:

```
GET cohere-wikipedia-en-cosine-v2/_count

PUT cohere-wikipedia-en-cosine-v2/_settings
{ "index": { "number_of_replicas": 1 } }
```

Optionally force-merge to fewer segments before benchmarking query latency.

## 4. Load into S3 Vectors

Files in `ingestion/s3vectors/`:

| File | Purpose |
|---|---|
| `package_beta_sdk.sh` | Builds a Spark `--archives` zip containing the beta boto3/botocore |
| `create_hybrid_index.py` | Creates the vector bucket (optional) and the hybrid index (run locally) |
| `ingest_cohere_to_s3vectors.py` | The Spark job |
| `submit_emr_serverless.sh` | Uploads the job script and starts the job run |
| `query_hybrid_index.py` | Runs one hybrid query to check the load |

Hybrid search needs the beta SDK. The public boto3 has no
`textSearchConfigurations` on `CreateIndex` and no `queryTexts` on
`QueryVectors`. Beta participants get the SDK from their AWS account team; it
is not included in this repository.

### 4.1 Package the beta SDK for Spark

```bash
export BETA_SDK_ARCHIVE_LOCAL_PATH=/path/to/hybrid-pythonsdk.zip   # or leave unset to download from BETA_SDK_ARCHIVE_S3_URI
export OUTPUT_S3_URI=s3://<artifact-bucket>/artifacts/s3vectors-beta-sdk.zip

bash ingestion/s3vectors/package_beta_sdk.sh
```

The script unpacks the beta wheels, downloads the pure-Python dependencies
from PyPI (jmespath, python-dateutil, urllib3, six, s3transfer), and uploads
one flat zip. The machine that runs it needs PyPI access. The job later adds
the zip's contents to `PYTHONPATH` on the driver and executors.

### 4.2 Create the bucket and the hybrid index

Run this with the beta SDK installed locally (for example in this project's
`.venv`):

```bash
python3 ingestion/s3vectors/create_hybrid_index.py \
  --vector-bucket <vector-bucket> \
  --index cohere-wikipedia-en-hybrid-v1 \
  --region us-east-2 \
  --create-vector-bucket            # omit if the bucket already exists
```

The index is created with `float32`, dimension 1024 and cosine distance. The
metadata key `passage_body` is listed in both `nonFilterableMetadataKeys` and
`textSearchConfigurations`, so it can hold text longer than the filterable
size limit and is indexed for BM25. Use `--searchable-metadata-key` to change
the name.

Vector bucket encryption cannot be changed after creation. Decide on it before
passing `--create-vector-bucket`.

Each vector is written as:

| Field | Source |
|---|---|
| `key` | `_id` |
| `data.float32` | `emb` |
| `metadata.passage_body` | `text`, truncated to 32 KB (searchable) |
| `metadata.title`, `metadata.url` | truncated to 2 KB (filterable) |
| `metadata.document_id` | `_id` |

### 4.3 Run a small validation load first

```bash
export AWS_REGION=us-east-2
export EMR_APPLICATION_ID=<us-east-2-application-id>
export EMR_EXECUTION_ROLE_ARN=arn:aws:iam::<account-id>:role/<emr-job-role>
export JOB_SCRIPT_S3_URI=s3://<artifact-bucket>/scripts/ingest_cohere_to_s3vectors.py
export BETA_SDK_ARCHIVE_S3_URI=s3://<artifact-bucket>/artifacts/s3vectors-beta-sdk.zip
export LOG_URI=s3://<artifact-bucket>/logs/cohere-to-s3vectors/
export VECTOR_BUCKET=<vector-bucket>
export VECTOR_INDEX=cohere-wikipedia-en-hybrid-v1
export S3_INPUT=s3://<your-bucket>/cohere-wikipedia/en/   # your copy from section 1.1
export LIMIT=1000

bash ingestion/s3vectors/submit_emr_serverless.sh
```

When it succeeds, check the vectors with `query_hybrid_index.py` (4.5), then
run the full load with `LIMIT=0`.

| Variable | Default | Meaning |
|---|---|---|
| `OUTPUT_PARTITIONS` | `4` | Maximum concurrent writers |
| `BATCH_SIZE` | `500` | Vectors per `PutVectors` call (1 to 500) |
| `MAX_RETRIES` | `12` | Retries on throttling and 5xx errors, with exponential backoff capped at 60 s |
| `LIMIT` | `0` | Rows to load; `0` loads everything |

Each `PutVectors` request is capped at 500 vectors and a 19 MB payload, below
the service's request limit. Write throughput is limited by the service, so
more writers mostly produce more throttling; raise `OUTPUT_PARTITIONS` only
gradually.

### 4.4 Wait for the text index

S3 Vectors builds the BM25 index in the background. Full-text and hybrid
queries work once the searchable field is `ACTIVE`:

```python
import boto3
c = boto3.client("s3vectors", region_name="us-east-2")
cfg = c.get_index(vectorBucketName="<vector-bucket>", indexName="cohere-wikipedia-en-hybrid-v1")
print(cfg["index"]["metadataConfiguration"]["textSearchConfigurations"])   # status BUILDING or ACTIVE
```

### 4.5 Check the load with a hybrid query

`query_hybrid_index.py` reads the query vector from a JSON file in S3 (a list
of 1024 floats), so you can test without an embedding model:

```bash
python3 ingestion/s3vectors/query_hybrid_index.py \
  --vector-bucket <vector-bucket> \
  --index cohere-wikipedia-en-hybrid-v1 \
  --query-vector-s3-uri s3://<artifact-bucket>/queries/aurora.json \
  --query-text "aurora borealis" \
  --top-k 5
```

Create the vector with `cohere.embed-multilingual-v3` and
`input_type=search_query` (see section 1).

## 5. Querying the loaded data

Once both loads finish, point the comparison UI at the two indexes in `.env`
(`OPENSEARCH_INDEX`, `S3V_BUCKET`, `S3V_INDEX`) and set `ML_MODEL_ID` to an
OpenSearch ML model that uses `cohere.embed-multilingual-v3`. See the main
[README](../README.md).

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Vector results are unrelated to the query and all scores fall in a narrow band | The query was embedded with a different model than the data. Use `cohere.embed-multilingual-v3` with `input_type=search_query`. |
| OpenSearch job fails at startup resolving `--packages` | No route to Maven Central. See the private-subnet note in section 2. |
| OpenSearch bulk rejections (HTTP 429) or connector timeouts | The domain cannot keep up. Lower `OUTPUT_PARTITIONS`, `MAX_EXECUTORS` or the bulk size, or raise `NETWORK_TIMEOUT`. |
| `OpenSearch returned HTTP 403` from `create_cohere_vector_index.py` | Your IAM identity is not mapped to an OpenSearch role with index-create permission. |
| `create_hybrid_index.py` or a query fails with `Unknown parameter ... textSearchConfigurations` or `queryTexts` | The public boto3 is in use. Install the beta wheels locally. For the Spark job, check the archive path and the `PYTHONPATH` settings in `submit_emr_serverless.sh`. |
| S3 Vectors `TooManyRequestsException` that fails the job | Retries were exhausted. Lower `OUTPUT_PARTITIONS` or raise `MAX_RETRIES`. |
| Full-text or hybrid queries return nothing right after the load | The text index is still `BUILDING` (section 4.4). |
