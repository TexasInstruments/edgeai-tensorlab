
# Introduction

Deep Neural Networks (DNN) are used for a variety of Computer Vision tasks such as object detection, image segmentation, Image classification, etc. Other sensors such as LiDAR, Radar etc. are also being used with Neural Networks for applications such as Machine Vision, Industrial Inspection, Advanced Driver Assistance, Autonomous Driving etc. 

Embedded SoCs are designed to operate under constrained environments and typically they have limited amount of memory and computation throughput compared to Cloud based inference platforms. 

The DNNs usually have redundant weights and making those redundant weights, 0, is known as pruning which help reduce the complexity. Pruning could be structured as well as unstructured which is basically is whether is a particular pattern to prune the weights or not. 

Unstructured pruning is thus, we are making the weights 0 without any predefined pattern, whereas in structured pruning, we have a specific pattern in mind, on the basis os which pruning is carried out. We will be discussing two structured pruning approaches over here.

1. N:M Structured Pruning

    ![N:M Pruning](n2m.png)

    This approach would reduce the number of parameters and further the latency. Both the streaming ]engine as well as the inference engine can take advantage of this pruning approach. The example is for 3:6 pruning, it can be easily extended to any N:M pruning. 

2. Channel Structured Pruning

    ![Channel Pruning](channel.png)

    This effectively reduces the computations as those channels could be removed from the layer. With these considerations, we can get a smaller network, which is parametrically as well as computationally efficient.


# APIs

## Basic Usage

Pruner wrapper can be directly wrapped around your model while training, which allows you to introduce pruning.

    from edgeai_torchtoolkit import xmodelopt
    model = xmodelopt.pruning.PrunerModule(model, pruning_ratio=args.pruning_ratio, total_epochs=args.epochs, 
                            init_train_ep = args.init_train_ep, pruning_class=args.pruning_class, 
                            pruning_type=args.pruning_type, global_pruning=args.global_pruning)

Here, we need to specify : 

1.  pruning ratio - the amount of pruning we need in the network by the end of training process
2.  total_epochs - total number of training epochs

We can also specify the following, depepnding on the use case :

1. init_train_ep - the number of epochs that need to be trained before weights start to get pruned (Default: 5)
2. pruning_class - the pruning class to be used (Options : 'blend' (default), 'sigmoid', 'incremental'). However, only blend class has been tested. The user can make their own pruning class as well. Refer to Section : Advanced Usage
3. pruning_type - the type of pruning that we want to incorporate in the network (Options: 'channel' (default), 'n2m', 'prunechannelunstructured', 'unstructured')
4. global_pruning - whether we want to prune each layer with a different pruning ratio, depending on the spread of weights (Default: False)


> This could be incorporated in the training script itself, and model thereafter could be trained as it was getting trained before.


<!-- ## Advanced Usage 

### Declaring own Parametrization / Pruning Class 

We can make our own parametrization class, which lets one to directly use the toolkit for own pruning algorithm. Here, we will guide through our parametrization class 


### Specifying own pruning type -->



