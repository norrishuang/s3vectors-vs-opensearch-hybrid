#!/usr/bin/env python3
"""Index the Cohere Wikipedia English embedding dataset from S3 into OpenSearch.

Run this file as an EMR Serverless Spark job. It requires the OpenSearch Spark
connector specified in submit_cohere_to_opensearch.sh.
"""

import argparse
import base64
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import urlparse

EXPECTED_COLUMNS = {"_id", "url", "title", "text", "emb"}
VECTOR_DIMENSION = 1024
INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write Cohere Wikipedia Parquet embeddings from S3 to OpenSearch."
    )
    parser.add_argument(
        "--s3-input",
        required=True,
        help="Parquet input prefix, e.g. s3://<your-bucket>/cohere-wikipedia/en/.",
    )
    parser.add_argument(
        "--opensearch-endpoint",
        required=True,
        help="HTTPS endpoint of the OpenSearch domain or Serverless collection.",
    )
    parser.add_argument("--index", required=True, help="Existing OpenSearch index name.")
    parser.add_argument("--region", default="us-east-1", help="AWS Region.")
    parser.add_argument(
        "--target",
        choices=("aos", "aoss"),
        default="aos",
        help="aos for an OpenSearch Service domain, aoss for OpenSearch Serverless.",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=0,
        help="Optional maximum number of Spark output partitions; 0 preserves input partitions.",
    )
    parser.add_argument(
        "--batch-size-entries",
        type=int,
        default=100,
        help="Maximum documents in each OpenSearch bulk request.",
    )
    parser.add_argument(
        "--batch-size-bytes",
        default="1mb",
        help="Maximum payload size of each OpenSearch bulk request.",
    )
    parser.add_argument(
        "--write-retry-count",
        type=int,
        default=20,
        help="Number of retries for retryable OpenSearch bulk-write failures.",
    )
    parser.add_argument(
        "--write-retry-wait",
        default="60s",
        help="Delay between OpenSearch bulk-write retries.",
    )
    parser.add_argument(
        "--network-timeout",
        default="300s",
        help="OpenSearch connector request timeout.",
    )
    parser.add_argument("--auth-mode", choices=("basic", "sigv4"), default="sigv4")
    parser.add_argument("--create-index", action="store_true")
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=0)
    return parser.parse_args()


def endpoint_host(endpoint):
    normalized = endpoint if "://" in endpoint else f"https://{endpoint}"
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ValueError("opensearch-endpoint must be an HTTPS host without a path")
    return parsed.hostname


def endpoint_url(endpoint):
    return f"https://{endpoint_host(endpoint)}"


def validate_args(args):
    if not args.s3_input.startswith("s3://"):
        raise ValueError("s3-input must use an s3:// URI")
    if not INDEX_NAME.fullmatch(args.index) or args.index.startswith("_"):
        raise ValueError("index must be lowercase and must not start with an underscore")
    if args.partitions < 0:
        raise ValueError("partitions must be zero or greater")
    if args.batch_size_entries < 1:
        raise ValueError("batch-size-entries must be at least 1")
    if args.write_retry_count < 0:
        raise ValueError("write-retry-count cannot be negative")
    if args.shards < 1 or args.replicas < 0:
        raise ValueError("shards must be at least 1 and replicas cannot be negative")
    if args.skip_write and not args.create_index:
        raise ValueError("skip-write requires create-index")


def validate_source(source, functions, array_type):
    missing = EXPECTED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Parquet source is missing columns: {', '.join(sorted(missing))}")

    embedding_type = source.schema["emb"].dataType
    if not isinstance(embedding_type, array_type):
        raise ValueError(f"emb must be an array, found {embedding_type.simpleString()}")

    sample = (
        source.where(functions.col("emb").isNotNull())
        .select(functions.size("emb").alias("dimension"))
        .limit(1)
        .collect()
    )
    if not sample:
        raise ValueError("Parquet source contains no non-null embeddings")
    if sample[0]["dimension"] != VECTOR_DIMENSION:
        raise ValueError(
            f"expected {VECTOR_DIMENSION}-dimension embeddings, found {sample[0]['dimension']}"
        )


def basic_auth_headers(spark):
    configuration = spark.sparkContext._jsc.hadoopConfiguration()
    username = configuration.get("opensearch.net.http.auth.user")
    password = configuration.get("opensearch.net.http.auth.pass")
    if not username or not password:
        raise ValueError("Missing OpenSearch basic-auth Spark configuration")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def request_index(endpoint, path, method, payload, headers):
    request = urllib.request.Request(
        f"{endpoint}/{path}",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenSearch returned HTTP {error.code}: {detail}") from error


def index_definition(args):
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": args.shards,
                "number_of_replicas": args.replicas,
                "refresh_interval": "-1",
            }
        },
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "url": {"type": "keyword", "index": False, "doc_values": False},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": VECTOR_DIMENSION,
                    "space_type": "cosinesimil",
                    "mode": "on_disk",
                    "compression_level": "32x",
                },
            }
        },
    }


def write_options(args, host, spark):
    options = {
        "opensearch.nodes": host,
        "opensearch.port": "443",
        "opensearch.net.ssl": "true",
        "opensearch.nodes.wan.only": "true",
        "opensearch.mapping.id": "document_id",
        "opensearch.batch.size.entries": str(args.batch_size_entries),
        "opensearch.batch.size.bytes": args.batch_size_bytes,
        "opensearch.batch.write.retry.count": str(args.write_retry_count),
        "opensearch.batch.write.retry.wait": args.write_retry_wait,
        "opensearch.net.timeout": args.network_timeout,
        # "index" replaces the document with the same _id, making retries safe.
        "opensearch.write.operation": "index",
    }
    if args.auth_mode == "basic":
        configuration = spark.sparkContext._jsc.hadoopConfiguration()
        username = configuration.get("opensearch.net.http.auth.user")
        password = configuration.get("opensearch.net.http.auth.pass")
        if not username or not password:
            raise ValueError("Missing OpenSearch basic-auth Spark configuration")
        options.update(
            {
                "opensearch.net.http.auth.user": username,
                "opensearch.net.http.auth.pass": password,
            }
        )
    else:
        options.update(
            {
                "opensearch.aws.sigv4.enabled": "true",
                "opensearch.aws.sigv4.region": args.region,
            }
        )
    if args.auth_mode == "sigv4" and args.target == "aoss":
        options.update(
            {
                "opensearch.aws.sigv4.service.name": "aoss",
                "opensearch.serverless": "true",
            }
        )
    return options


def main():
    args = parse_args()
    validate_args(args)
    host = endpoint_host(args.opensearch_endpoint)
    endpoint = endpoint_url(args.opensearch_endpoint)
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import ArrayType

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("cohere_to_opensearch")
    spark = SparkSession.builder.appName("cohere-wikipedia-to-opensearch").getOrCreate()

    try:
        if args.create_index:
            if args.auth_mode != "basic":
                raise ValueError("create-index requires --auth-mode basic")
            logger.info("Creating OpenSearch index %s", args.index)
            request_index(
                endpoint, args.index, "PUT", index_definition(args), basic_auth_headers(spark)
            )
            logger.info("OpenSearch index %s created", args.index)
        if args.skip_write:
            return

        logger.info("Reading Cohere Parquet input from %s", args.s3_input)
        source = spark.read.parquet(args.s3_input)
        validate_source(source, F, ArrayType)

        documents = (
            source.select(
                F.col("_id").alias("document_id"),
                F.col("url"),
                F.col("title"),
                F.col("text"),
                F.col("emb").alias("embedding"),
            )
            .where(
                F.col("_id").isNotNull()
                & F.col("emb").isNotNull()
                & (F.size("emb") == VECTOR_DIMENSION)
            )
        )

        input_partitions = documents.rdd.getNumPartitions()
        if 0 < args.partitions < input_partitions:
            documents = documents.coalesce(args.partitions)
            logger.info("Coalesced output from %d to %d partitions", input_partitions, args.partitions)
        else:
            logger.info("Using %d output partitions", input_partitions)

        logger.info("Starting write to OpenSearch index %s", args.index)
        (
            documents.write.format("opensearch")
            .options(**write_options(args, host, spark))
            .mode("append")
            .save(args.index)
        )
        if args.auth_mode == "basic":
            request_index(
                endpoint,
                f"{args.index}/_settings",
                "PUT",
                {"index": {"refresh_interval": "30s"}},
                basic_auth_headers(spark),
            )
        logger.info("OpenSearch write completed for index %s", args.index)
    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.exception("Cohere-to-OpenSearch job failed: %s", error)
        sys.exit(1)
