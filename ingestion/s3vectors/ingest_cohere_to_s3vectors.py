#!/usr/bin/env python3
"""Write Cohere Wikipedia Parquet embeddings to an S3 Vectors hybrid index."""

import argparse
import json
import time


VECTOR_DIMENSION = 1024
MAX_VECTORS_PER_REQUEST = 500
MAX_PUT_BYTES = 19 * 1024 * 1024
MAX_SEARCHABLE_TEXT_BYTES = 32 * 1024
MAX_FILTERABLE_VALUE_BYTES = 2 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3-input",
        default="s3://myemr-bucket-01/data/cohere-embedding-dataset/en/",
    )
    parser.add_argument("--vector-bucket", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional document limit for a small validation import; 0 imports all rows.",
    )
    return parser.parse_args()


def truncate_utf8(value, maximum_bytes):
    encoded = str(value or "").encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return encoded.decode("utf-8")
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def to_s3vector(row):
    return {
        "key": str(row.document_id),
        "data": {"float32": [float(item) for item in row.embedding]},
        "metadata": {
            "document_id": truncate_utf8(row.document_id, 1024),
            "url": truncate_utf8(row.url, MAX_FILTERABLE_VALUE_BYTES),
            "title": truncate_utf8(row.title, MAX_FILTERABLE_VALUE_BYTES),
            "passage_body": truncate_utf8(row.text, MAX_SEARCHABLE_TEXT_BYTES),
        },
    }


def encoded_size(vector):
    return len(json.dumps(vector, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def put_with_retry(client, args, vectors):
    from botocore.exceptions import ClientError

    retryable_codes = {
        "TooManyRequestsException",
        "RequestTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
    }
    for attempt in range(args.max_retries + 1):
        try:
            client.put_vectors(
                vectorBucketName=args.vector_bucket,
                indexName=args.index,
                vectors=vectors,
            )
            return
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in retryable_codes or attempt == args.max_retries:
                raise
            time.sleep(min(60, 2**attempt))


def upload_partition(rows, args):
    from botocore.session import Session

    client = Session().create_client("s3vectors", region_name=args.region)
    batch = []
    batch_bytes = 0
    maximum_vectors = min(args.batch_size, MAX_VECTORS_PER_REQUEST)

    for row in rows:
        vector = to_s3vector(row)
        vector_size = encoded_size(vector)
        if vector_size > MAX_PUT_BYTES:
            raise ValueError(f"Vector {vector['key']} exceeds the PutVectors request limit")
        if batch and (
            len(batch) >= maximum_vectors
            or batch_bytes + vector_size > MAX_PUT_BYTES
        ):
            put_with_retry(client, args, batch)
            batch = []
            batch_bytes = 0
        batch.append(vector)
        batch_bytes += vector_size

    if batch:
        put_with_retry(client, args, batch)


def main():
    args = parse_args()
    if args.region != "us-east-2":
        raise ValueError("The S3 Vectors Hybrid Search beta is available only in us-east-2")
    if not 1 <= args.batch_size <= MAX_VECTORS_PER_REQUEST:
        raise ValueError(f"batch-size must be between 1 and {MAX_VECTORS_PER_REQUEST}")
    if args.partitions < 1 or args.max_retries < 0 or args.limit < 0:
        raise ValueError("partitions must be positive and retry/limit values cannot be negative")

    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.appName("cohere-to-s3vectors-hybrid").getOrCreate()
    try:
        source = spark.read.parquet(args.s3_input)
        required = {"_id", "url", "title", "text", "emb"}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")

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
        if args.limit:
            documents = documents.limit(args.limit)
        input_partitions = documents.rdd.getNumPartitions()
        if args.partitions < input_partitions:
            documents = documents.coalesce(args.partitions)
        documents.foreachPartition(lambda rows: upload_partition(rows, args))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
