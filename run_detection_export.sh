#!/usr/bin/env bash

# xmmdet is not installed - make it visible as a local module
export PYTHONPATH=.:$PYTHONPATH

python ./scripts/export_pytorch2onnx.py

echo "Done."