#!/usr/bin/env bash

echo "Please download PandaSet Dataset first from https://pandaset.org/ before running the script '"$0"'."

DOWNLOAD_DIR=$1  # The directory where the downloaded data set is stored
# DATA_ROOT=${2:-$DOWNLOAD_DIR}  # The root directory of the converted dataset

echo "If you have already downloaded it at "$1", please ignore above message."

echo "Unzipping all .gz file in directory "$1" to load them separately." 

# Find and process all .gz files in the directory and its subdirectories
find "$DOWNLOAD_DIR" -type f -name "*.gz" | while read -r file; do
  echo "Unzipping file: $file"
  gunzip --keep "$file"
  echo "Unzipped file: $file "
done

echo "All .gz files have been unzipped."
