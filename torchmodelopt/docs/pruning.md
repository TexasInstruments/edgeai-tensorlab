
- Prune/sparsify a model: [edgeai_torchmodelopt.xmodelopt.pruning](../edgeai_torchmodelopt/xmodelopt/pruning) - Both structured and unstructured pruning is supported. Structured pruning includes N:M pruning and channel pruning. Here, we provide with a few parametrization techniques, and the user can bring their own technique as well and implement as a part of the toolkit.<br>

- Channel Pruning/Sparsity uses torch.fx to find the connections and obtain a smaller network after the training with induced sparsity finishes. 

The detailed usage for the API is documented in [Model Sparsity](../edgeai_torchmodelopt/xmodelopt/pruning/README.md).

## Results

Here are the results on pruning the network with n:m pruning and channel pruning using our blending based pruning algorithm.

Below are the results with networks having 30% channel sparsity. These networks could give upto 50% FLOP reduction and double the speedup.
After obtaining 30% channel sparsity, only 70% of the channels remain and the operations (MACs) are dependant on the square of the parameters. Thus, 49% of the operations remain and thus would lead to 51% FLOP reduction.

| Models        |  Accuracy          | Pruned Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| ResNet50     | 76.13         |   74.07              |

Here are the results on 41:64 (n:m) pruning, that comes up to 0.640625 pruning ratio.

| Models        |  Accuracy          | Pruned Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2  | 71.88 | 70.37           |
| ResNet50     | 76.13         |   76.02               |


> <h3> The results obtained are preliminary results, and have scope for further optimizations. </h3>
