#!/bin/bash
set -e
echo  -e "process ID $$"

#####################################################################################
EAI_BENCHMARK_MOUNT_PATH=${1:-"172.24.236.164:/home/abhay/edgeai-benchmark"}
DATASET_MOUNT_PATH=${2:-"tensorlabdata2.dhcp.ti.com:/data/shared/pub/projects/edgeai-algo/data/datasets/benchmark/benchmark_datasets"}

mkdir -p ~/edgeai-benchmark
cd ~

counter=1
ip_fetched=0
while [ $counter -le 100 ];
do
    ip_addr=`ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p'`
    if [ "$ip_addr" != "" ]; then
        echo "IP ADDRESS OF EVM: $ip_addr"
        ip_fetched=1
        break
    fi
    sleep 5
    echo "Trying to get IP. Attempt ${counter}/100"
    counter=$((counter+1))
done

if [ $ip_fetched -ne 1 ]; then
    echo "[ Error ] could not get IP"
    exit 1
fi

mount ${EAI_BENCHMARK_MOUNT_PATH} edgeai-benchmark
if [ $? -ne 0 ]; then
    echo "[ Error ] could not mount to ${EAI_BENCHMARK_MOUNT_PATH}"
    exit 1
fi
echo "[ Info ] ${EAI_BENCHMARK_MOUNT_PATH} mount successful"


## add check whether dataset is mounted or not
cd ~/edgeai-benchmark/dependencies
if [ -d "datasets" ]; then
  if mountpoint -q datasets; then
    umount datasets 
  fi
  rm -r datasets
fi

mkdir -p datasets
mount -r -t nfs ${DATASET_MOUNT_PATH} datasets
if [ $? -ne 0 ]; then
    echo "[ Error ] could not mount to ${DATASET_MOUNT_PATH}"
    exit 1
fi
echo "[ Info ] ${DATASET_MOUNT_PATH} mount successful"

export HTTPS_PROXY=http://webproxy.ext.ti.com:80
export https_proxy=http://webproxy.ext.ti.com:80
export HTTP_PROXY=http://webproxy.ext.ti.com:80
export http_proxy=http://webproxy.ext.ti.com:80
export ftp_proxy=http://webproxy.ext.ti.com:80
export FTP_PROXY=http://webproxy.ext.ti.com:80
export no_proxy=ti.com,localhost 

cd ~/edgeai-benchmark
###########################################################
if [ -d "evm_wheels" ]; then 
  pip3 install --no-index --find-links evm_wheels/ -r requirements/requirements_evm.txt
else
  pip3 install --no-input -r requirements/requirements_evm.txt
fi

###########################################################

echo "SCRIPT_EXECUTED_SUCCESSFULLY"