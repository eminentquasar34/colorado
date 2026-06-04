#!/usr/bin/env bash
set -euo pipefail

# Run training, upload results using gcloud storage, then shut down the VM.
# Edit BUCKET, OUTPUT_DIR, TRAIN_CMD and ZONE as needed before running.

BUCKET=gs://educaiton-colorado-cse493g1
OUTPUT_DIR=models/ebert_generator_flan_t5_small_full_gcp
TRAIN_CMD=(python3 src/train_generator.py
  --output-dir "${OUTPUT_DIR}"
  --model-name google/flan-t5-small
  --max-source-length 768
  --max-target-length 512
  --batch-size 2
  --gradient-accumulation-steps 4
  --epochs 3
  --fp16)

mkdir -p "${OUTPUT_DIR}"

echo "Starting training..."
"${TRAIN_CMD[@]}" 2>&1 | tee train.log

echo "Uploading model and logs to ${BUCKET}..."
# Use gcloud storage cp --recursive to copy directories
gcloud storage cp --recursive "${OUTPUT_DIR}" "${BUCKET}/models/"
gcloud storage cp train.log "${BUCKET}/models/$(basename ${OUTPUT_DIR})/train.log"

echo "Upload complete — shutting down VM."
sudo shutdown -h now
