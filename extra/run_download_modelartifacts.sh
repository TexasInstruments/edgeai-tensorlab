#!/usr/bin/env bash

version=$(python3 version.py --delimiter=_)

url=http://software-dl.ti.com/jacinto7/esd/modelzoo/$version/modelartifacts/8bits
directory_prefix=./work_dirs/modelartifacts/8bits

for artifact in $(ls -1 ./work_dirs/modelartifacts/8bits |grep .tar.gz.link)
do
artifact=$(basename $artifact .link)
wget ${url}/${artifact} --directory-prefix=$directory_prefix
done
