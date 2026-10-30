#!/usr/bin/env python3
"""Run a hybrid query against the Cohere S3 Vectors Hybrid Search Beta index."""

import argparse
import json
from urllib.parse import urlparse


VECTOR_DIMENSION = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-bucket", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--query-vector-s3-uri", required=True)
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--metadata-key", default="passage_body")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--region", default="us-east-2")
    return parser.parse_args()


def parse_s3_uri(uri):
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError("query-vector-s3-uri must be an s3://bucket/key URI")
    return parsed.netloc, parsed.path.lstrip("/")


def main():
    args = parse_args()
    if args.region != "us-east-2":
        raise ValueError("The S3 Vectors Hybrid Search beta is available only in us-east-2")
    if not 1 <= args.top_k <= 100:
        raise ValueError("top-k must be between 1 and 100")

    from botocore.session import Session

    session = Session()
    vector_bucket, vector_key = parse_s3_uri(args.query_vector_s3_uri)
    vector_body = session.create_client("s3").get_object(
        Bucket=vector_bucket, Key=vector_key
    )["Body"].read()
    vector = json.loads(vector_body)
    if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
        raise ValueError(f"query vector must contain {VECTOR_DIMENSION} values")

    client = session.create_client("s3vectors", region_name=args.region)
    response = client.query_vectors(
        vectorBucketName=args.vector_bucket,
        indexName=args.index,
        queryVector={"float32": [float(item) for item in vector]},
        queryTexts=[{"metadataKey": args.metadata_key, "text": args.query_text}],
        topK=args.top_k,
        returnDistance=True,
        returnScores=True,
        returnMetadata=True,
    )
    print(json.dumps(response, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
