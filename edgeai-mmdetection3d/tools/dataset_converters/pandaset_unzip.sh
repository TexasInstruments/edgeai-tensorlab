#!/usr/bin/env bash

DOWNLOAD_DIR=$1  # The directory where the downloaded data set is stored
# DATA_ROOT=${2:-$DOWNLOAD_DIR}  # The root directory of the converted dataset


# Find and process all .gz files in the directory and its subdirectories
find "$DOWNLOAD_DIR" -type f -name "*.gz" | while read -r file; do
  echo "Unzipping file: $file"
  gunzip --keep "$file"
  echo "Unzipped file: $file "
done

echo "All .gz files have been unzipped."


# for zip_file in $DOWNLOAD_DIR/KITTI_Object/raw/*.zip; do
#     echo "Unzipping $zip_file to $DATA_ROOT ......"
# 	unzip -oq $zip_file -d $DATA_ROOT
#     echo "[Done] Unzip $zip_file to $DATA_ROOT"
#     # delete the original files
# 	rm -f $zip_file
# done
