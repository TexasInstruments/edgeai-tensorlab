
### Guidelines For Training To Get Best Accuracy With Quantization

**These are important** - we are listing these guidelines upfront because it is important to follow these.

We recommend that the training uses **sufficient amount of regularization (weight decay) for all parameters**. Regularization / weight decay ensures that the weights, biases and other parameters (if any) are small and compact - this is good for quantization. These features are supported in most of the popular training framework.<br>

We have noticed that some training code bases do not use weight decay for biases. Some other code bases do not use weight decay for the parameters in Depthwise convolution layers. All these are bad strategies for quantization. These poor choices done (probably to get a 0.1% accuracy lift with floating point) will result in a huge degradation in fixed point - sometimes several percentage points. The weight decay factor should not be too small. We have used a weight decay factor of 1e-4 for training several networks and we highly recommend a similar value. Please do no use small values such as 1e-5.<br>

We also highly recommend to use **Batch Normalization immediately after every Convolution layer**. This helps the feature map to be properly regularized/normalized. If this is not done, there can be accuracy degradation with quantization. This especially true for Depthwise Convolution layers. However applying Batch Normalization to the Convolution layer that does the prediction (for example, the very last layer in segmentation/object detection network) may hurt accuracy and can be avoided.<br>

There are several model types out there including MobileNets, ResNets, DenseNets, EfficientNets(Lite), RegNetX [9] etc. that are popular in the embedded community. The models using Depthwise convolutions (such as MobileNets) are more difficult to quantize - expect higher accuracy drop with quantization when using such models. We would like to **recommend RegNetX [9] models** as the most embedded friendly as they balance accuracy for a given complexity and the ease of quantization due to their use of grouped convolutions with carefully selected group sizes.<br>

To get the best accuracy at the quantization stage, it is important that the model is trained carefully, following the guidelines (even during the floating point training). Having spent a lot of time solving quantization issues, we would like to highlight that following these guidelines are of at most importance. Otherwise, there is a high risk that the tools and techniques described here may not be completely effective in solving the accuracy drop due to quantization. To summarize, if you are getting poor accuracy with quantization, please check the following:<br>
- Weight decay is applied to all layers / parameters and that weight decay factor is good.<br>
- Ensure that the Convolution layers in the network have Batch Normalization layers immediately after that. The only exception allowed to this rule is for the very last Convolution layer in the network (for example the prediction layer in a segmentation network or detection network, where adding Batch normalization might hurt the floating point accuracy).<br>

It is important to refactor your code such that it is symbolically traceable and properly quantizable (observers are inserted at proper locations). More information about symbolic tracing support can be found [here](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing). Few common suggestions could be :
1. Removing the assert statements in the code because the symbolic tracing step does not have shape inference and it could cause some issues.
2. Using torch functions for arithmetics, for example, using torch.add(a,b) instead of a + b.
3. If some module is not supposted to be quantized, it can be wrapped with @torch.fx.wrap

