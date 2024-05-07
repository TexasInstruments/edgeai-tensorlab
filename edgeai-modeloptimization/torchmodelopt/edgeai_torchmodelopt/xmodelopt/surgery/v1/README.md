## Easy to use model surgery utilities

This can be used to replace modules in a model that are derived from torch.nn.Modules. Most of the common classes used a model such as torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU are derived from torch.nn.Modules, so they can be easily replaced. 

But if a model has used torch.relu operator directory or a x.relu (x being a tensor), those cannot be replaced by this surgery utility. However, those also can be replaced by our latest torch.fx based surgery tools - please consult the documentation of (v2) tools.

## APIs

### convert_to_lite_model
This is the main model surgery function.

#### Usage: 
```
## insert this line into your model training script
model = edgeai_torchmodelopt.xmodelopt.surgery.v1.convert_to_lite_model(model)
```

#### Example:
```
model = torchvision.models.mobilenet_v3_large()
model = edgeai_torchmodelopt.xmodelopt.surgery.v1.convert_to_lite_model(model)
```

### get_replacement_dict_default
This can be used to get the default replacement dict. This replacement dict can then be modified and passed to the convert api. 

#### Usage:
```
replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v1.get_replacement_dict_default())
replacement_dict.update({torch.nn.GELU: torch.nn.ReLU})
```

Now apply the conversion using the updated replacement_dict
```
model = edgeai_torchmodelopt.xmodelopt.surgery.v1.convert_to_lite_model(model, replacement_dict=replacement_dict)
```

It is possible to pass additional arguments to the replacement entries
```
replacement_dict.update({torch.nn.GELU: [torch.nn.ReLU, dict(inplace=True)]})
```

To take the value from source modules being replaced to the new modules that are replacing them, just provide the attribute names as strings.
```
replacement_dict.update({torch.nn.LeakyReLU: [torch.nn.ReLU, 'inplace']})
```

