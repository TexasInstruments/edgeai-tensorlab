#!/usr/bin/env bash

# torch.fx based model surgery and training

# PYTHONPATH must start with a : to be able to load local modules
export PYTHONPATH=:$PYTHONPATH

# Date/time in YYYYMMDD-HHmmSS format
DATE_TIME=`date +'%Y%m%d-%H%M%S'`
#=========================================================================================
# sample models that can be used
#model=resnet50
#model=mobilenet_v2
#model=mobilenet_v2
#model=resnet18
#model=regnetx200mf
#model=regnetx400mf
#model=regnetx400mf
#model=regnetx800mf
#model=regnetx1p6gf

# these lite models are created using model surgery from models in torchvision
# these lite models will be available only if --model-surgery <argument> argument is set to one of these
# --model-surgery 1: legacy module based surgery
# --model-surgery 2: advanced model surgery with torch.fx (to be released)
#model=mobilenet_v3_large_lite
#model=mobilenet_v3_small_lite
#=========================================================================================
# set the appropriate pretrained weights for the above model
#model_weights="ResNet50_Weights.IMAGENET1K_V1"
#model_weights="MobileNet_V2_Weights.IMAGENET1K_V1"
# model=mobilenet_v2_lite
dataset_path=/home/a0491009/datasets/imagenet
models=( \
"mobilenet_v3_large_lite" \
"mobilenet_v3_small_lite" \
"mobilenet_v2_lite"
)

ORIGINAL_ACCURACY=(\
"78" \
"78" \
"78" \
)

#=========================================================================================
# set the appropriate pretrained weights for the above model
#model_weights="ResNet50_Weights.IMAGENET1K_V1"
#model_weights="MobileNet_V2_Weights.IMAGENET1K_V1"
model_weights=(\
"/home/a0507161/Kunal/edgeai-modelforest/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_20210507_checkpoint.pth" \
"/home/a0507161/Kunal/edgeai-modelforest/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_small_20210429_checkpoint.pth" \
"/home/a0507161/Kunal/edgeai-modelforest/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_20191224_checkpoint.pth" \
)


val_resize_size=232 #256 #232
val_crop_size=224
gpus=4
batch_sizes=(2 2 2)
out_dir="./data/checkpoints/torchvision/imagenet_classification/${DATE_TIME}"
result_path="${out_dir}/result.csv"
# for val in {1..20..2}
# do
#   echo "printing ${val}"
#   if [ $val == 9 ]
#   then
#      break
#   fi
# done

#=========================================================================================
#Run Evaluation and report generaing script for all models
for i in ${!models[@]}; do
    model=${models[$i]}
    command="./scripts/test_surgery.py --data-path=${dataset_path} --gpus=$gpus \
    --batch-size=${batch_sizes[$i]} --model=${model} \
    --original-accuracy=${ORIGINAL_ACCURACY[$i]} \
    --opset-version=18 --val-resize-size=$val_resize_size --val-crop-size=$val_crop_size"
    output_dir="${out_dir}/${model}"
    echo $model
    python3 $command --weights=${model_weights[$i]} --output-dir=${output_dir} --result-path=${result_path}
  if [ $i == 0 ]
  then
     break
  fi
    # training: single GPU (--device=cuda:0)or CPU (--device=cpu) run
    # python3 ${command} --weights=${model_weights} --output-dir=${output_dir}

    # training: multi-gpu distributed data parallel
    # torchrun --nproc_per_node 4 ${command} --weights=${model_weights} --output-dir=${output_dir}
done