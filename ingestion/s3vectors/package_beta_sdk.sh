#!/usr/bin/env bash
set -euo pipefail

: "${BETA_SDK_ARCHIVE_S3_URI:=s3://s3vectors-beta-sdk/hybrid-pythonsdk.zip}"
: "${BETA_SDK_ARCHIVE_LOCAL_PATH:=}"
: "${OUTPUT_S3_URI:?Set OUTPUT_S3_URI, for example s3://myemr-bucket-01/artifacts/s3vectors-beta-sdk.zip}"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

if [[ -n "$BETA_SDK_ARCHIVE_LOCAL_PATH" ]]; then
  cp "$BETA_SDK_ARCHIVE_LOCAL_PATH" "$work_dir/hybrid-pythonsdk.zip"
else
  aws s3 cp "$BETA_SDK_ARCHIVE_S3_URI" "$work_dir/hybrid-pythonsdk.zip"
fi
unzip -q "$work_dir/hybrid-pythonsdk.zip" -d "$work_dir/wheels"
mkdir "$work_dir/python" "$work_dir/dependencies"
python3 -m pip download --quiet --only-binary=:all: --no-deps \
  --dest "$work_dir/dependencies" \
  "jmespath==1.0.1" \
  "python-dateutil==2.9.0.post0" \
  "urllib3==2.5.0" \
  "six==1.17.0" \
  "s3transfer==0.19.0"
find "$work_dir/wheels" "$work_dir/dependencies" -type f -name "*.whl" -print0 | \
  while IFS= read -r -d "" wheel; do
    unzip -q "$wheel" -d "$work_dir/python"
  done

(
  cd "$work_dir/python"
  zip -qr "$work_dir/s3vectors-beta-sdk.zip" .
)
aws s3 cp "$work_dir/s3vectors-beta-sdk.zip" "$OUTPUT_S3_URI"
