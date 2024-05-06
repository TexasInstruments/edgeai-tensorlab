# Image classification reference training scripts

This folder contains reference training scripts for image classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs with 
the following parameters:

| Parameter                | value  |
| ------------------------ | ------ |
| `--batch_size`           | `32`   |
| `--epochs`               | `90`   |
| `--lr`                   | `0.1`  |
| `--momentum`             | `0.9`  |
| `--wd`, `--weight-decay` | `1e-4` |
| `--lr-step-size`         | `30`   |
| `--lr-gamma`             | `0.1`  |

### AlexNet and VGG

Since `AlexNet` and the original `VGG` architectures do not include batch 
normalization, the default initial learning rate `--lr 0.1` is to high.

```
torchrun --nproc_per_node=8 train.py\
    --model $MODEL --lr 1e-2
```

Here `$MODEL` is one of `alexnet`, `vgg11`, `vgg13`, `vgg16` or `vgg19`. Note
that `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, and `vgg19_bn` include batch
normalization and thus are trained with the default parameters.

### ResNext-50 32x4d
```
torchrun --nproc_per_node=8 train.py\
    --model resnext50_32x4d --epochs 100
```


### ResNext-101 32x8d

```
torchrun --nproc_per_node=8 train.py\
    --model resnext101_32x8d --epochs 100
```

Note that the above command corresponds to a single node with 8 GPUs. If you use
a different number of GPUs and/or a different batch size, then the learning rate
should be scaled accordingly. For example, the pretrained model provided by
`torchvision` was trained on 8 nodes, each with 8 GPUs (for a total of 64 GPUs),
with `--batch_size 16` and `--lr 0.4`, instead of the current defaults
which are respectively batch_size=32 and lr=0.1

### MobileNetV2
```
torchrun --nproc_per_node=8 train.py\
     --model mobilenet_v2 --epochs 300 --lr 0.045 --wd 0.00004\
     --lr-step-size 1 --lr-gamma 0.98
```


### MobileNetV3 Large & Small
```
torchrun --nproc_per_node=8 train.py\
     --model $MODEL --epochs 600 --opt rmsprop --batch-size 128 --lr 0.064\ 
     --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2
```

Here `$MODEL` is one of `mobilenet_v3_large` or `mobilenet_v3_small`.

Then we averaged the parameters of the last 3 checkpoints that improved the Acc@1. See [#3182](https://github.com/pytorch/vision/pull/3182) 
and [#3354](https://github.com/pytorch/vision/pull/3354) for details.


### EfficientNet

The weights of the B0-B4 variants are ported from Ross Wightman's [timm repo](https://github.com/rwightman/pytorch-image-models/blob/01cb46a9a50e3ba4be167965b5764e9702f09b30/timm/models/efficientnet.py#L95-L108).

The weights of the B5-B7 variants are ported from Luke Melas' [EfficientNet-PyTorch repo](https://github.com/lukemelas/EfficientNet-PyTorch/blob/1039e009545d9329ea026c9f7541341439712b96/efficientnet_pytorch/utils.py#L562-L564).


### RegNet

#### Small models
```
torchrun --nproc_per_node=8 train.py\
     --model $MODEL --epochs 100 --batch-size 128 --wd 0.00005 --lr=0.8\
     --lr-scheduler=cosineannealinglr --lr-warmup-method=linear\
     --lr-warmup-epochs=5 --lr-warmup-decay=0.1
```
Here `$MODEL` is one of `regnet_x_400mf`, `regnet_x_800mf`, `regnet_x_1_6gf`, `regnet_y_400mf`, `regnet_y_800mf` and `regnet_y_1_6gf`. Please note we used learning rate 0.4 for `regent_y_400mf` to get the same Acc@1 as [the paper)(https://arxiv.org/abs/2003.13678).

### Medium models
```
torchrun --nproc_per_node=8 train.py\
     --model $MODEL --epochs 100 --batch-size 64 --wd 0.00005 --lr=0.4\
     --lr-scheduler=cosineannealinglr --lr-warmup-method=linear\
     --lr-warmup-epochs=5 --lr-warmup-decay=0.1
```
Here `$MODEL` is one of `regnet_x_3_2gf`, `regnet_x_8gf`, `regnet_x_16gf`, `regnet_y_3_2gf` and `regnet_y_8gf`.

### Large models
```
torchrun --nproc_per_node=8 train.py\
     --model $MODEL --epochs 100 --batch-size 32 --wd 0.00005 --lr=0.2\
     --lr-scheduler=cosineannealinglr --lr-warmup-method=linear\
     --lr-warmup-epochs=5 --lr-warmup-decay=0.1
```
Here `$MODEL` is one of `regnet_x_32gf`, `regnet_y_16gf` and `regnet_y_32gf`.

## Mixed precision training
Automatic Mixed Precision (AMP) training on GPU for Pytorch can be enabled with the [NVIDIA Apex extension](https://github.com/NVIDIA/apex).

Mixed precision training makes use of both FP32 and FP16 precisions where appropriate. FP16 operations can leverage the Tensor cores on NVIDIA GPUs (Volta, Turing or newer architectures) for improved throughput, generally without loss in model accuracy. Mixed precision training also often allows larger batch sizes. GPU automatic mixed precision training for Pytorch Vision can be enabled via the flag value `--apex=True`.

```
torchrun --nproc_per_node=8 train.py\
    --model resnext50_32x4d --epochs 100 --apex
```

## Quantized

### Parameters used for generating quantized models:

For all post training quantized models (All quantized models except mobilenet-v2), the settings are:

1. num_calibration_batches: 32
2. num_workers: 16
3. batch_size: 32
4. eval_batch_size: 128
5. backend: 'fbgemm'

```
python train_quantization.py --device='cpu' --post-training-quantize --backend='fbgemm' --model='<model_name>'
```

For Mobilenet-v2, the model was trained with quantization aware training, the settings used are:
1. num_workers: 16
2. batch_size: 32
3. eval_batch_size: 128
4. backend: 'qnnpack'
5. learning-rate: 0.0001
6. num_epochs: 90
7. num_observer_update_epochs:4
8. num_batch_norm_update_epochs:3
9. momentum: 0.9
10. lr_step_size:30
11. lr_gamma: 0.1
12. weight-decay: 0.0001

```
torchrun --nproc_per_node=8 train_quantization.py --model='mobilenet_v2'
```

Training converges at about 10 epochs.

For Mobilenet-v3 Large, the model was trained with quantization aware training, the settings used are:
1. num_workers: 16
2. batch_size: 32
3. eval_batch_size: 128
4. backend: 'qnnpack'
5. learning-rate: 0.001
6. num_epochs: 90
7. num_observer_update_epochs:4
8. num_batch_norm_update_epochs:3
9. momentum: 0.9
10. lr_step_size:30
11. lr_gamma: 0.1
12. weight-decay: 0.00001

```
torchrun --nproc_per_node=8 train_quantization.py --model='mobilenet_v3_large' \
    --wd 0.00001 --lr 0.001
```

For post training quant, device is set to CPU. For training, the device is set to CUDA.

### Command to evaluate quantized models using the pre-trained weights:

```
python train_quantization.py --device='cpu' --test-only --backend='<backend>' --model='<model_name>'
```
