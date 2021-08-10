#!/usr/bin/env bash

version=$(python version.py --delimiter=_)

url=http://software-dl.ti.com/jacinto7/esd/modelzoo/$version/modelartifacts/8bits

for artifact in $(ls -1 ./modelartifacts/8bits |grep .tar.gz)
do
echo ${url}/${artifact} > ./modelartifacts/8bits/${artifact}.link
done
