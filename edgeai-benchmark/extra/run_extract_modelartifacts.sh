#!/usr/bin/env bash

directory_prefix=./work_dirs/modelartifacts/8bits

cd ${directory_prefix}

find . -name "*.tar.gz" -exec tar --one-top-level -zxvf "{}" \;
