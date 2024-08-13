### Post Training Calibration (PTC) in PyTorch

It can be useful to do PTC in torch framework as it can ensure that the model is getting properly quantized and the expected accuracy can be evaluated. It is easy to incorporate into an existing PyTorch training code. We provide a wrapper module called PTCFxModule to automate all the tasks required for PTC. The user simply needs to wrap his model in PTCFxModule and do the training.

The overall flow of training is as follows:<br>
- Step 1: Train your model in floating point as usual.<br>
- Step 2: Starting from the floating point model as pretrained weights, do Post Training Quantization. In order to do this wrap your model in the wrapper module called  edgeai_torchmodelopt.xmodelopt.quantization.v2.PTCFxModule and perform calibration for 100 examples from your dataset.<br>

PTCFxModule does the following operations to the model. Note that PTCFxModule will handle these tasks - Refer to the section "How to use  PTCFxModule" to know how to use the module.<br>
- Symbolically Trace the model
- Insert Quantization Stubs to capture the data distribution
- Calibrate the network (can be done separately or training script could be utilised)
- Convert the network to a quantized version
- Optionally export the onnx graph of the model


Please see the documentation of [Pytorch fx-based PTQ](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html) for mode details.


#### How to use  PTCFxModule
The following is a brief description of how to use this wrapper module:
```
import edgeai_torchmodelopt

# create your model here:
model = ...

# load your pretrained checkpoint/weights here to do PTC
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# wrap your model in xnn.quantization.PTCFxModule. 
# once it is wrapped, the actual model is in model.module
model = edgeai_torchmodelopt.xmodelopt.quantization.v2.PTCFxModule(model, backend='qnnpack', bias_calibration_factor=0.01, num_batch_norm_update_epochs=1, num_observer_update_epochs=2)

## Note: if you want to test your model after PTC, loading of the PTC checkpoint/weights should be here into model.module
## pretrained_ptc_data = torch.load(pretrained_ptc_path)
## model.module.load_state_dict(pretrained_ptc_data)

# your training loop here with with loss, backward, optimizer and scheduler. 
# this is the usual training loop - but use a lower learning rate such as 1e-5
model.eval()
for images, target in my_dataset_calibration:
    output = model(images)

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


### Compilation of PTC Models in TIDL

#### Instructions for compiling PTC models in TIDL 9.1 onwards:

If you are using TIDL to infer a model trained using QAT tools provided in this repository, please set the following in the import config file of TIDL to use the provided calibration parameters: <br>
```
'advanced_options:prequantized_model' : 1 # Bypasses TIDL quantization
'accuracy_level': 0 
```

