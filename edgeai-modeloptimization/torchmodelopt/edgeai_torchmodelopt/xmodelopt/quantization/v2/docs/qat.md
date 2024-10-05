## Quantization Aware Training (QAT)
QAT is needed only if the accuracy obtained with PTQ / PTC is not satisfactory.

QAT is easy to incorporate into an existing PyTorch training code. We provide a wrapper module called QATFxModule to automate all the tasks required for QAT. The user simply needs to wrap their model in QATFxModule and do the training.

The overall flow of training is as follows:<br>
- Step 1: Train your model in floating point as usual.<br>
- Step 2: Starting from the floating point model as pretrained weights, do Quantization Aware Training. In order to do this wrap your model in the wrapper module called edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule and perform training with a small learning rate. About 25 to 50 epochs of training may be required to get the best accuracy.<br>

QATFxModule does the following operations to the model. Note that QATFxModule will handle these tasks - the only thing that is required is to wrap the user's module in QATFxModule as explained in the section "How to use  QATFxModule".<br>
- Replace layers in the model by their Fake Quantized versions - including merging Conv+BN+Activation layers & range collection.<br>
- Quantize weghts & activations during the forward pass.<br>
- Other modifications to help the learning process.<br>

Please see the documentation of [Pytorch native QAT](https://pytorch.org/docs/stable/quantization.html) for mode details.


#### How to use QATFxModule

Example ipython notebook to use the API is available [over here](../../../../../example_notebooks/quantization_qat.ipynb).

The following is a brief description of how to use this wrapper module:
```
import edgeai_torchmodelopt

# create your model here:
model = ...

# load your pretrained checkpoint/weights here to do QAT
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# wrap your model in xnn.quantization.QATFxModule. 
# once it is wrapped, the actual model is in model.module
model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(model, total_epochs=epochs)

## Note: if you want to test your model after QAT, loading of the QAT checkpoint/weights should be here into model.module
## pretrained_qat_data = torch.load(pretrained_qat_path)
## model.module.load_state_dict(pretrained_qat_data)

# your training loop here with with loss, backward, optimizer and scheduler. 
# this is the usual training loop - but use a lower learning rate such as 1e-5
model.train()
for images, target in my_dataset_train:
    output = model(images)
    # loss, backward(), optimizer step etc comes here as usual in training

model.eval()

# save the checkpoint/weights - the trained module is in model.module
torch.save(model.module.state_dict(), os.path.join(save_path,'model.pth'))

# convert the model to operate with integer operations (instead of QDQ FakeQuantize operations)
model = model.convert()

# create a dummy input - this is required for onnx export.
dummy_input = torch.rand((1,3,384,768))

# export the model to onnx format - the trained module is in model.module
torch.onnx.export(model.module, dummy_input, os.path.join(save_path,'model.onnx'), export_params=True, verbose=False, do_constant_folding=True, opset_version=18)
```

Optional: Careful attention needs to be given to how the parameters of the pretrained model is loaded and trained model is saved as shown in the above code snippet. We have provided a utility function called edgeai_torchmodelopt.xnn.utils.load_weights() that prints which parameters are loaded correctly and which are not - you can use this load function if needed to ensure that your parameters are loaded correctly.


### Compilation of QAT Models in TIDL

#### Instructions for compiling QAT models in TIDL 9.1 onwards:

If you are using TIDL to infer a model trained using QAT tools provided in this repository, please set the following in the import config file of TIDL to use the provided calibration parameters: <br>
```
'advanced_options:prequantized_model' : 1 # Bypasses TIDL quantization
'accuracy_level': 0 
```

