#!/usr/bin/env python3
"""Create an Amazon S3 Vectors hybrid-search bucket and index in us-east-2."""

import argparse
import sys


VECTOR_DIMENSION = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-bucket", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--searchable-metadata-key", default="passage_body")
    parser.add_argument("--create-vector-bucket", action="store_true")
    return parser.parse_args()


def already_exists(error):
    return error.response.get("Error", {}).get("Code") in {
        "ConflictException",
        "ResourceAlreadyExistsException",
    }


def create_bucket(client, bucket_name):
    from botocore.exceptions import ClientError

    try:
        return client.create_vector_bucket(vectorBucketName=bucket_name)
    except ClientError as error:
        if already_exists(error):
            return {"alreadyExists": True}
        raise


def create_index(client, args):
    from botocore.exceptions import ClientError

    metadata_configuration = {
        "nonFilterableMetadataKeys": [args.searchable_metadata_key],
        "textSearchConfigurations": [
            {"metadataKey": args.searchable_metadata_key}
        ],
    }
    try:
        return client.create_index(
            vectorBucketName=args.vector_bucket,
            indexName=args.index,
            dataType="float32",
            dimension=VECTOR_DIMENSION,
            distanceMetric="cosine",
            metadataConfiguration=metadata_configuration,
        )
    except ClientError as error:
        if already_exists(error):
            return client.get_index(
                vectorBucketName=args.vector_bucket,
                indexName=args.index,
            )
        raise


def main():
    args = parse_args()
    if args.region != "us-east-2":
        raise ValueError("The S3 Vectors Hybrid Search beta is available only in us-east-2")

    from botocore.session import Session

    client = Session().create_client("s3vectors", region_name=args.region)
    bucket = (
        create_bucket(client, args.vector_bucket)
        if args.create_vector_bucket
        else client.get_vector_bucket(vectorBucketName=args.vector_bucket)
    )
    index = create_index(client, args)
    print({"vectorBucket": bucket, "index": index})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Failed to create S3 Vectors hybrid index: {error}", file=sys.stderr)
        sys.exit(1)
