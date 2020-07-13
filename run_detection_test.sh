#!/usr/bin/env bash

while [ 1 ]; do
  python ./scripts/test_detection_main.py
  echo "Sleeping...."
  sleep 7200
done
