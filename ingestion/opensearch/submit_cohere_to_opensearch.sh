#!/usr/bin/env bash
set -euo pipefail

# Set these before running this script.
: "${EMR_APPLICATION_ID:?Set EMR_APPLICATION_ID}"
: "${EMR_EXECUTION_ROLE_ARN:?Set EMR_EXECUTION_ROLE_ARN}"
: "${JOB_SCRIPT_S3_URI:?Set JOB_SCRIPT_S3_URI, for example s3://<bucket>/scripts/emr_serverless_cohere_to_opensearch.py}"
: "${LOG_URI:?Set LOG_URI, for example s3://<bucket>/logs/cohere-to-opensearch/}"
: "${OPENSEARCH_ENDPOINT:?Set OPENSEARCH_ENDPOINT}"
: "${OPENSEARCH_INDEX:?Set OPENSEARCH_INDEX}"
: "${OPENSEARCH_SECRET_NAME:?Set OPENSEARCH_SECRET_NAME}"

AWS_REGION="${AWS_REGION:-us-east-1}"
S3_INPUT="${S3_INPUT:-s3://myemr-bucket-01/data/cohere-embedding-dataset/en/}"
OPENSEARCH_TARGET="${OPENSEARCH_TARGET:-aos}"
OPENSEARCH_USERNAME="${OPENSEARCH_USERNAME:-admin}"
OUTPUT_PARTITIONS="${OUTPUT_PARTITIONS:-4}"
EXECUTOR_CORES="${EXECUTOR_CORES:-1}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-6G}"
INITIAL_EXECUTORS="${INITIAL_EXECUTORS:-2}"
MIN_EXECUTORS="${MIN_EXECUTORS:-1}"
MAX_EXECUTORS="${MAX_EXECUTORS:-4}"
BATCH_SIZE_ENTRIES="${BATCH_SIZE_ENTRIES:-100}"
BATCH_SIZE_BYTES="${BATCH_SIZE_BYTES:-1mb}"
WRITE_RETRY_COUNT="${WRITE_RETRY_COUNT:-20}"
WRITE_RETRY_WAIT="${WRITE_RETRY_WAIT:-60s}"
NETWORK_TIMEOUT="${NETWORK_TIMEOUT:-300s}"
LOCAL_JOB_SCRIPT="${LOCAL_JOB_SCRIPT:-$(dirname "$0")/emr_serverless_cohere_to_opensearch.py}"

# This application uses EMR Spark 8.1: Spark 4.x and Scala 2.13. The AWS SDK v2 dependencies
# are required by the connector when it signs requests with the job role.
OPENSEARCH_SPARK_PACKAGES="${OPENSEARCH_SPARK_PACKAGES:-org.opensearch.client:opensearch-spark-40_2.13:2.0.0,software.amazon.awssdk:auth:2.31.59,software.amazon.awssdk:regions:2.31.59,software.amazon.awssdk:http-client-spi:2.31.59,software.amazon.awssdk:identity-spi:2.31.59,software.amazon.awssdk:sdk-core:2.31.59,software.amazon.awssdk:utils:2.31.59}"

aws s3 cp "$LOCAL_JOB_SCRIPT" "$JOB_SCRIPT_S3_URI"

aws emr-serverless start-job-run \
  --application-id "$EMR_APPLICATION_ID" \
  --execution-role-arn "$EMR_EXECUTION_ROLE_ARN" \
  --name "cohere-wikipedia-to-opensearch" \
  --job-driver '{
    "sparkSubmit": {
      "entryPoint": "'"$JOB_SCRIPT_S3_URI"'",
      "entryPointArguments": [
        "--s3-input", "'"$S3_INPUT"'",
        "--opensearch-endpoint", "'"$OPENSEARCH_ENDPOINT"'",
        "--index", "'"$OPENSEARCH_INDEX"'",
        "--region", "'"$AWS_REGION"'",
        "--target", "'"$OPENSEARCH_TARGET"'",
        "--partitions", "'"$OUTPUT_PARTITIONS"'",
        "--batch-size-entries", "'"$BATCH_SIZE_ENTRIES"'",
        "--batch-size-bytes", "'"$BATCH_SIZE_BYTES"'",
        "--write-retry-count", "'"$WRITE_RETRY_COUNT"'",
        "--write-retry-wait", "'"$WRITE_RETRY_WAIT"'",
        "--network-timeout", "'"$NETWORK_TIMEOUT"'",
        "--auth-mode", "basic"
      ],
      "sparkSubmitParameters": "--packages '"$OPENSEARCH_SPARK_PACKAGES"' --conf spark.hadoop.opensearch.net.http.auth.user='"$OPENSEARCH_USERNAME"' --conf spark.hadoop.opensearch.net.http.auth.pass=EMR.secret@'"$OPENSEARCH_SECRET_NAME"' --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.initialExecutors='"$INITIAL_EXECUTORS"' --conf spark.dynamicAllocation.minExecutors='"$MIN_EXECUTORS"' --conf spark.dynamicAllocation.maxExecutors='"$MAX_EXECUTORS"' --conf spark.executor.cores='"$EXECUTOR_CORES"' --conf spark.executor.memory='"$EXECUTOR_MEMORY"' --conf spark.driver.cores=1 --conf spark.driver.memory=4G"
    }
  }' \
  --configuration-overrides '{
    "monitoringConfiguration": {
      "s3MonitoringConfiguration": {
        "logUri": "'"$LOG_URI"'"
      }
    }
  }'
