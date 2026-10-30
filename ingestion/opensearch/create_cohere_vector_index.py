#!/usr/bin/env python3
"""Create the target k-NN index before launching the EMR Serverless ingest job."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from urllib.parse import quote, urlparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a 1024-dimension Cohere vector index in OpenSearch."
    )
    parser.add_argument("--endpoint", required=True, help="OpenSearch HTTPS endpoint.")
    parser.add_argument("--index", required=True, help="New index name.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--shards", required=True, type=int, help="Primary shard count.")
    parser.add_argument("--replicas", default=0, type=int, help="Replica count during ingest.")
    parser.add_argument(
        "--target", choices=("aos", "aoss"), default="aos", help="OpenSearch deployment type."
    )
    return parser.parse_args()


def request_body(args):
    index_settings = {"knn": True}
    if args.target == "aos":
        index_settings.update(
            {
                "number_of_shards": args.shards,
                "number_of_replicas": args.replicas,
                # Restore a normal refresh interval after the bulk load.
                "refresh_interval": "-1",
            }
        )

    embedding = {
        "type": "knn_vector",
        "dimension": 1024,
        # Disk mode uses an HNSW index with automatic rescoring. Explicitly
        # retain its binary 32x compression default for this large corpus.
        "space_type": "cosinesimil",
        "mode": "on_disk",
        "compression_level": "32x",
    }
    return {
        "settings": {"index": index_settings},
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "url": {"type": "keyword", "index": False, "doc_values": False},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embedding": embedding,
            }
        },
    }


def normalized_endpoint(value):
    parsed = urlparse(value if "://" in value else f"https://{value}")
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ValueError("endpoint must be an HTTPS host without a path")
    return f"https://{parsed.hostname}"


def create_index(args):
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    endpoint = normalized_endpoint(args.endpoint)
    body = json.dumps(request_body(args), separators=(",", ":")).encode("utf-8")
    url = f"{endpoint}/{quote(args.index, safe='')}"
    service = "aoss" if args.target == "aoss" else "es"

    session = boto3.Session(region_name=args.region)
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials were found")
    signed = AWSRequest(
        method="PUT",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(credentials.get_frozen_credentials(), service, args.region).add_auth(signed)

    request = urllib.request.Request(
        url, data=body, method="PUT", headers=dict(signed.headers.items())
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            print(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenSearch returned HTTP {error.code}: {detail}") from error


def main():
    args = parse_args()
    if args.shards < 1 or args.replicas < 0:
        raise ValueError("shards must be at least 1 and replicas cannot be negative")
    create_index(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Index creation failed: {error}", file=sys.stderr)
        sys.exit(1)
