#!/bin/bash
set -e
echo  -e "process ID $$"

#####################################################################################
EAI_BENCHMARK_MOUNT_PATH=${EAI_BENCHMARK_MOUNT_PATH:-""}
TEST_SUITE=${TEST_SUITE:-"BENCHMARK"}
LOCAL_IP=${LOCAL_IP:-""}

if [ "$EAI_BENCHMARK_MOUNT_PATH" == "" ]; then
  echo "[ Error ] EAI_BENCHMARK_MOUNT_PATH not set"
  exit 1
fi

mkdir -p ~/edgeai-benchmark
cd ~

if [ "$LOCAL_IP" != "" ]; then
  ifconfig eth0 $LOCAL_IP
fi

counter=1
ip_fetched=0
while [ $counter -le 50 ];
do
    ip_addr=`ifconfig | sed -En 's/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p'`
    if [ "$ip_addr" != "" ]; then
        echo "IP ADDRESS OF EVM: $ip_addr"
        ip_fetched=1
        break
    fi
    sleep 5
    echo "Trying to get IP. Attempt ${counter}/50"
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

if [ "$TEST_SUITE" == "BENCHMARK" ]; then
  BENCHMARK_DATASET_MOUNT_PATH=${BENCHMARK_DATASET_MOUNT_PATH:-""}
  if [ "$BENCHMARK_DATASET_MOUNT_PATH" == "" ]; then
    echo "[ Error ] BENCHMARK_DATASET_MOUNT_PATH not set"
    exit 1
  fi

  ## add check whether dataset is mounted or not
  cd ~/edgeai-benchmark/dependencies
  if [ -d "datasets" ]; then
    if mountpoint -q datasets; then
      umount datasets 
    fi
    rm -r datasets
  fi

  mkdir -p datasets
  mount -r -t nfs ${BENCHMARK_DATASET_MOUNT_PATH} datasets
  if [ $? -ne 0 ]; then
      echo "[ Error ] could not mount to ${BENCHMARK_DATASET_MOUNT_PATH}"
      exit 1
  fi
  echo "[ Info ] ${BENCHMARK_DATASET_MOUNT_PATH} mount successful"

  export HTTPS_PROXY=http://webproxy.ext.ti.com:80
  export https_proxy=http://webproxy.ext.ti.com:80
  export HTTP_PROXY=http://webproxy.ext.ti.com:80
  export http_proxy=http://webproxy.ext.ti.com:80
  export ftp_proxy=http://webproxy.ext.ti.com:80
  export FTP_PROXY=http://webproxy.ext.ti.com:80
  export no_proxy=ti.com,localhost

  cd ~/edgeai-benchmark
  if [ "$LOCAL_IP" == "" ]; then
    if [ -d "evm_wheels" ]; then 
      pip3 install --no-index --find-links evm_wheels/ -r requirements/requirements_evm.txt
    else
      pip3 install --no-input -r requirements/requirements_evm.txt
    fi
  fi



elif [ "$TEST_SUITE" == "TIDL_UNIT_TEST" ]; then
  TIDL_UNIT_TEST_DATASET_MOUNT_PATH=${TIDL_UNIT_TEST_DATASET_MOUNT_PATH:-""}
  if [ "$TIDL_UNIT_TEST_DATASET_MOUNT_PATH" == "" ]; then
    echo "[ Error ] TIDL_UNIT_TEST_DATASET_MOUNT_PATH not set"
    exit 1
  fi
  TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH=${TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH:-""}
  if [ "$TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH" == "" ]; then
    echo "[ Warning ] TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH not set, using default artifacts"
  fi

  cd ~/edgeai-benchmark/tests/tidl_unit/
  if [ -d "tidl_unit_test_data" ]; then
    if mountpoint -q tidl_unit_test_data; then
      umount -l tidl_unit_test_data &> /dev/null
    fi
  fi

  mkdir -p tidl_unit_test_data
  cd tidl_unit_test_data
  mount -r -t nfs ${TIDL_UNIT_TEST_DATASET_MOUNT_PATH} ./
  if [ $? -ne 0 ]; then
      echo "[ Error ] could not mount to ${TIDL_UNIT_TEST_DATASET_MOUNT_PATH}"
      exit 1
  fi
  echo "[ Info ] ${TIDL_UNIT_TEST_DATASET_MOUNT_PATH} mount successful"
  cd ../

  if [ "$TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH" != "" ]; then
    if [ -d "work_dirs/modelartifacts/8bits" ]; then
      if mountpoint -q work_dirs/modelartifacts/8bits; then
        umount -l work_dirs/modelartifacts/8bits &> /dev/null
      fi
    fi

    mkdir -p work_dirs/modelartifacts/8bits
    cd work_dirs/modelartifacts/8bits
    mount -r -t nfs ${TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH} ./
    if [ $? -ne 0 ]; then
        echo "[ Error ] could not mount to ${TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH}"
        exit 1
    fi
    echo "[ Info ] ${TIDL_UNIT_TEST_MODEL_ARTIFACTS_MOUNT_PATH} mount successful"
  fi

  export HTTPS_PROXY=http://webproxy.ext.ti.com:80
  export https_proxy=http://webproxy.ext.ti.com:80
  export HTTP_PROXY=http://webproxy.ext.ti.com:80
  export http_proxy=http://webproxy.ext.ti.com:80
  export ftp_proxy=http://webproxy.ext.ti.com:80
  export FTP_PROXY=http://webproxy.ext.ti.com:80
  export no_proxy=ti.com,localhost

  cd ~/edgeai-benchmark

  if [ "$LOCAL_IP" == "" ]; then
    if [ -d "evm_wheels" ]; then 
      pip3 install --no-index --find-links evm_wheels/ -r requirements/requirements_evm.txt
    else
      pip3 install --no-input -r requirements/requirements_evm.txt
    fi
    git config --global --add safe.directory /root/edgeai-benchmark
    ./setup_evm.sh
    pip3 install --no-input onnx
    pip3 install --no-input -r tests/tidl_unit/requirements.txt
    pip3 install --no-input numpy==1.26.4
  fi
fi

cd ~/edgeai-benchmark
echo "SCRIPT_EXECUTED_SUCCESSFULLY"