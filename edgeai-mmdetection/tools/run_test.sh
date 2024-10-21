#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
# TODO enable multi-gpu test
GPUS=$3


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    ${@:4}
