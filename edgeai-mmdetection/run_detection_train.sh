#!/usr/bin/env bash

# tools is not installed - make it visible as a local module
export PYTHONPATH=.:$PYTHONPATH

python ./scripts/train_detection_main.py

echo "Done."
