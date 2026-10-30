#!/usr/bin/env bash
set -euo pipefail

: "${EMR_APPLICATION_ID:?Set EMR_APPLICATION_ID for an EMR Serverless application in us-east-2}"
: "${EMR_EXECUTION_ROLE_ARN:?Set EMR_EXECUTION_ROLE_ARN}"
: "${JOB_SCRIPT_S3_URI:?Set JOB_SCRIPT_S3_URI}"
: "${BETA_SDK_ARCHIVE_S3_URI:?Set BETA_SDK_ARCHIVE_S3_URI}"
: "${LOG_URI:?Set LOG_URI}"
: "${VECTOR_BUCKET:?Set VECTOR_BUCKET}"
: "${VECTOR_INDEX:?Set VECTOR_INDEX}"

AWS_REGION="${AWS_REGION:-us-east-2}"
: "${S3_INPUT:?Set S3_INPUT to your copy of the dataset, for example s3://<bucket>/cohere-wikipedia/en/}"
OUTPUT_PARTITIONS="${OUTPUT_PARTITIONS:-4}"
BATCH_SIZE="${BATCH_SIZE:-500}"
MAX_RETRIES="${MAX_RETRIES:-12}"
LIMIT="${LIMIT:-0}"
LOCAL_JOB_SCRIPT="${LOCAL_JOB_SCRIPT:-$(dirname "$0")/ingest_cohere_to_s3vectors.py}"

aws s3 cp "$LOCAL_JOB_SCRIPT" "$JOB_SCRIPT_S3_URI"

aws emr-serverless start-job-run \
  --region "$AWS_REGION" \
  --application-id "$EMR_APPLICATION_ID" \
  --execution-role-arn "$EMR_EXECUTION_ROLE_ARN" \
  --name "cohere-wikipedia-to-s3vectors-hybrid" \
  --job-driver '{
    "sparkSubmit": {
      "entryPoint": "'"$JOB_SCRIPT_S3_URI"'",
      "entryPointArguments": [
        "--s3-input", "'"$S3_INPUT"'",
        "--vector-bucket", "'"$VECTOR_BUCKET"'",
        "--index", "'"$VECTOR_INDEX"'",
        "--region", "'"$AWS_REGION"'",
        "--partitions", "'"$OUTPUT_PARTITIONS"'",
        "--batch-size", "'"$BATCH_SIZE"'",
        "--max-retries", "'"$MAX_RETRIES"'",
        "--limit", "'"$LIMIT"'"
      ],
      "sparkSubmitParameters": "--archives '"$BETA_SDK_ARCHIVE_S3_URI#s3vectors-beta-sdk"' --conf spark.emr-serverless.driverEnv.PYTHONPATH=./s3vectors-beta-sdk --conf spark.executorEnv.PYTHONPATH=./s3vectors-beta-sdk --conf spark.dynamicAllocation.enabled=true --conf spark.dynamicAllocation.initialExecutors=2 --conf spark.dynamicAllocation.minExecutors=1 --conf spark.dynamicAllocation.maxExecutors=4 --conf spark.executor.cores=1 --conf spark.executor.memory=6G --conf spark.driver.cores=1 --conf spark.driver.memory=4G"
    }
  }' \
  --configuration-overrides '{
    "monitoringConfiguration": {
      "s3MonitoringConfiguration": {
        "logUri": "'"$LOG_URI"'"
      }
    }
  }'
