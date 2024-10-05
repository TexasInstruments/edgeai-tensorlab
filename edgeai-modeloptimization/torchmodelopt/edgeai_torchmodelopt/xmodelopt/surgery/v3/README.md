## Easy to use model surgery utilities using torch.fx

This can be used to replace modules in a model that are derived from torch.nn.Modules. Most of the common classes used a model such as torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU are derived from torch.nn.Modules, so they can be easily replaced. If a model has used torch.relu operator directory or a x.relu (x being a tensor), those can also be replaced by this surgery utility. 

## APIs

### convert_to_lite_fx
This is the main model surgery function.

#### Usage: 
```py
## insert this line into your model training script
model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model, example_args,example_kwargs)
```

#### Example:
```py
model = torchvision.models.mobilenet_v3_large()
example_args = [torch.rand(128, 3, 224, 224)] # inputs passed in one forward pass to model (must match value or shape in case of tensors)
example_kwargs = {} # keyword arguments (if any) (must match value or shape in case of tensors)
model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model, example_args, example_kwargs)
```

### get_replacement_dict_default
This can be used to get the default replacement dict. This replacement dict can then be modified and passed to the convert api. 

#### Usage:
```
replacement_dict = copy.deepcopy(edgeai_torchmodelopt.xmodelopt.surgery.v2.get_replacement_dict_default())
replacement_dict.update({torch.nn.GELU: torch.nn.ReLU})
```

Now apply the conversion using the updated replacement_dict
```
model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model, example_args,example_kwargs, replacement_dict=replacement_dict)
```
| key             |   value|
|----|----|
| `module`/`type`        |   `module` (if same module is applicable all instances of former)
|    `module`/`type`     |   `type` (if the `__init__` doesn't require any positional arguement and same  module with default keyword arguments (if any) is applicable all instances of former )     |                     
|    `module`/`type`     |   a `replacement module generator function` (generates a module based on partition and main model or returns None if no replacement required) |                       
|    `module`/`type`     |   `tuple` of two elements - first   : `replacement module generator function` (sane as prev) and - second  : a `input adjustment function` based on the partition and inputs given to it (`default`: pass them as they appear in partition.input_nodes) |
<!-- TODO write about func here and in source code-->
The Utility function  [utils.get_source_partition](utils.py#L348) is used to find partition of nodes from the graph based on the class of the module which can be replaced afterwards with appropiate module generated for each partition in their place. 

## Note:
See more [detailed documentation](docs/details.md) of the model surgery implementation.