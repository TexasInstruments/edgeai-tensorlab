#!/usr/bin/env bash

url=http://software-dl.ti.com/jacinto7/esd/modelzoo/08_00_00_05/modelartifacts/8bits

for artifact in $(ls -1 ./work_dirs/modelartifacts/8bits |grep .tar.gz)
do
echo ${url}/${artifact} > ./work_dirs/modelartifacts/8bits/${artifact}.link
done
