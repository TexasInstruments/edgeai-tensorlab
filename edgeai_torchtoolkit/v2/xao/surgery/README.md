># PyTorch Model Surgery
---
This repository is a model surgery API that will make the models TIDL friendly by changing inner structure module of using [torch.fx](https://pytorch.org/docs/stable/fx.html) package.
 
<center>
<br><img src="Images/Intro_diagram.png"  width = "75%" alt="" /><br>
</center>

---

## Table of Content
---
- [Table of Content](#table-of-content)
- [Main API (Usages)](#main-api-usages)
  - [get\_replacement\_dict\_default](#get_replacement_dict_default)
  - [replace\_unsupported\_layers function](#replace_unsupported_layers-function)
  - [SurgeryModule Class](#surgerymodule-class)
- [Basics](#basics)
  - [Symbolic Trace](#symbolic-trace)
  - [Intermediate Representation (Graph)](#intermediate-representation-graph)
  - [Components of Graph](#components-of-graph)
  - [Node Objects](#node-objects)
  - [Code Generation](#code-generation)
- [Single Node Replacement Process](#single-node-replacement-process)
- [Main API (Explained)](#main-api-explained)
  - [Getting Default Replacement Dict](#getting-default-replacement-dict)
  - [Replacing Pattern with Replacement:](#replacing-pattern-with-replacement)
  - [About SurgeryModule Class](#about-surgerymodule-class)
- [Different Types of Possible Replacement Rules](#different-types-of-possible-replacement-rules)
  - [Replacement Type 1 (Module to Module change)](#replacement-type-1-module-to-module-change)
  - [Replacement Type 2 (Function to Function change)](#replacement-type-2-function-to-function-change)
  - [Replacement Type 3 (Function to Type/Module change)](#replacement-type-3-function-to-typemodule-change)
  - [Replacement Type 4 (Type to Type/Module change)](#replacement-type-4-type-to-typemodule-change)
  - [Replacement Type 5 (Custom Surgery Function)](#replacement-type-5-custom-surgery-function)
- [Other Module Components](#other-module-components)
- [Module Tested on till Date](#module-tested-on-till-date)
- [Results](#results)
---

## Main API (Usages)
[<p align = 'right'>Go To Top</p>](#table-of-content)

---
### [get_replacement_dict_default]()
```python
default_dict = surgery.get_replacement_dict_default() # a dict
```
- returns the default replacement dictionary defined in the module
- user can update the dictiony but the mapping should follow for the patterns and their replacements

### [replace_unsupported_layers function]()
```python
changed_model = surgery.replace_unsupported_layers(model:nn.Module,replacement_dict=default_dict) # a torch.fx.GraphModule
```
- returns the module after all changes are done on the model, which are defined in the replacement_dict
- args:
  - model: the target model we have to surgery on
  - replacement_dict: the replacement rules in form of pattern(keys) and replacements(vals) 
    - if repalcement_dict == None: it will take default replacement dict

### [SurgeryModule Class]()
```python
changed_model=surgery.SurgeryModule(model, replacement_dict=default_dict) # a nn.Module
```
- creates a wrapper module over the changed model after going through function [replace_unsupported_layer](#replace_unsupported_layers-function) along with the replacement_dict
- args:
  - model: the target model we have to surgery on
  - replacement_dict: the replacement rules in form of pattern(keys) and replacements(vals)
    - if repalcement_dict == None: it will take default replacement dict
- user can also get the replacement_dict used for surgery by using **get_replacement_dict** method on the changed model object
```python
replacement_dict=changed_model.get_replacement_dict() # a dict
```
---

## Basics 
[<p align = 'right'>Go To Top</p>](#table-of-content)

---
This API uses functionalities from the package [torch.fx](https://pytorch.org/docs/stable/fx.html) package to change the model. 

This package says to change the model, we have to go through 3 steps.
> 1. [Symbolic Trace](#symbolic-trace)
> 1. Modification in the [Intermediate Representation](#intermediate-representation) (Graph)
> 1. [Code Generation]()


<center>
<br><img src="Images/basic_diagram.png"  width = "75%" alt="" /><br>
</center>

### [Symbolic Trace](https://pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace)
It is the process of performing **“symbolic execution”** of the Python code. It feeds fake values, called **Proxies**, through the code. Operations on theses Proxies are recorded.

Code:
```python
class ExampleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.rand(1,3,224,224)
        self.act= nn.ReLU()
    def forward(self,x):
        return self.act(x.permute(0,1,2,3)+self.param)

model= ExampleModule()
from torch.fx import symbolic_trace
traced_model=symbolic_trace(model)
print(traced_model)
```

Output:
```
ExampleModule(
  (act): ReLU()
)


the container for the operations that were recorded during symbolic tracing. It consists of a list of Nodes that represent function inputs, call-sites (to functions, methods, or torch.nn.Module instances), and return values
def forward(self, x):
    permute = x.permute(0, 1, 2, 3);  x = None
    param = self.param
    add = permute + param;  permute = param = None
    act = self.act(add);  add = None
    return act
```

After symbolic trace, [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) object will be converted [torch.fx.GraphModule](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule) object with its own graph and code. Due to this, Inner composite modules (i.e. which are made up of more sub module) will lose their forward functions which can be seen in the **code** attribute of main model's GraphModule along with the the function calls and method calls of different intermediate objects.

### Intermediate Representation ([Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph))
It is the container for the operations that were recorded during symbolic tracing. It consists of a list of Nodes that represent function inputs, call-sites (to functions, methods, or [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) instances), and return values.

Code:
```python
print(traced_model.graph)
```

Output:
```
graph():
    %x : [#users=1] = placeholder[target=x]
    %permute : [#users=1] = call_method[target=permute](args = (%x, 0, 1, 2, 3), kwargs = {})
    %param : [#users=1] = get_attr[target=param]
    %add : [#users=1] = call_function[target=operator.add](args = (%permute, %param), kwargs = {})
    %act : [#users=1] = call_module[target=act](args = (%add,), kwargs = {})
```
Graph also can be printed in tabular way using its [print_tabular](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.print_tabular). For that we have to install package called '[tabulate](https://pypi.org/project/tabulate/)' using following command:

``` shell
pip install tabulate
```

Code:
```python
traced_model.graph.print_tabular()
```

Output:
```
opcode         name     target                   args              kwargs
-------------  -------  -----------------------  ----------------  --------
placeholder    x        x                        ()                {}
call_method    permute  permute                  (x, 0, 1, 2, 3)   {}
get_attr       param    param                    ()                {}
call_function  add      <built-in function add>  (permute, param)  {}
call_module    act      act                      (add,)            {}
output         output   output                   (act,)            {}
```

<center>
<br><img src="Images/basic_model_graph.png"  width = "75%" alt="" /><br>
</center>

Graph records the operation in 6 categories:
1. **placeholder** -> inputs
    - like x, y, inp
2. **output** -> final output of the model (tuple of all output provided by the model)
    - only a single node of this type
3. **get_attr** -> to get any attribute of intermediate objects
    - also for parameters stored as attributes of modules
4. **call_method**  -> to access any method of different objects used in forward passing
   - for Tensor: permute,add,sub,mul,etc 
5. **call_function** -> to access any function that will work on differnt objects and can be used in forward function
   - for torch: torch.permute, torch.add,etc
   - for +, -, etc: operator.add, operator.sub, etc
6. **call_module** -> to accessing primary module like instances of class in (nn.Conv2d,nn.BatchNorm2d,nn.ReLU,etc) in side the module structure

> **Note:** All Model will generate a directed acyclic graph as once an operation is done in the input in forward passing it will add a new node for it, hence preventing any cycle of nodes and direction will be the flow of the data between the nodes.


### Components of Graph
Graph has the following attributes: 
- **nodes:** of node_list type which has all the nodes in a topological order
<center>
<br><img src="Images/topological_example.png"  width = "75%" alt="" /><br>
</center>
For example, in the above graph the node_list will be:
<br><b><center>[x,y,add,sub,mul]</center></b>

- various functions for different purposes:
  - inserting nodes:
    - [insert_after](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph)(Node n)
    - [insert_before](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph)(Node n)
  - deleting nodes:
    - [erase_node](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph)(Node n)
  - creating nodes:
    - [call_function](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_function)(*args,**kwargs)
    - [call_method](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_method)(*args,**kwargs)
    - [call_module](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.call_module)(*args,**kwargs)
  - cheking if graph is correctly designed or not
    - [lint](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint)()
  - printing the graph:
    - [print_tabular](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.print_tabular)()
  
### Node Objects
Node generally contains the following attributes:
- **Name**: its name
- **Op**: its operation (call_function, call_method, call_module, placeholder, output)
- **Target**: where does it stores data intermediately for the forwarding the data for call_module and call_method or instace for function incase of it is a call_function node
- **Next and Prev**: Next and previous nodes of it in the node list of graph
- **Users**: a dictionary of nodes those use the output of this node 
- **Args**: argument passed to it. It also contains the node address it takes input from
- **Kwargs**: kwyword argument passed to it. It also contains the node address it takes input from
<center>
<br><img src="Images/node_table.png"  width = "75%" alt="" /><br>
</center>
### Code Generation
After changing the nodes in the graph, we can generate a code for the graph that will be the forward function for the module. After that, the module structure, graph and the forward code for the model in wrapper object of class [torch.fx.GraphModule](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule). So,

<b><p align = 'center'>Graph Module = Module Structure + Graph + Code</p></b> 

Code:
```python
print(traced_model.code)
```
Output:
```
def forward(self, x):
    permute = x.permute(0, 1, 2, 3);  x = None
    param = self.param
    add = permute + param;  permute = param = None
    act = self.act(add);  add = None
    return act
```

While forward passing an input through the GraphModule, this code is iterated over as it is actually stored as a string in the data structure.
 
---

## Single Node Replacement Process
For replacing single node, **node** in Graph Module **traced_model**:
1. with **call_method**
```py
with traced_model.graph.insert_after(node):
    new_node= traced_model.graph.call_method(replacement_method,args,kwargs):
    node.replace_all_uses_with(new_node)
traced_model.graph.erase_node(node)
```
Here replcement is also a method, so, **call_method** is used. Args must contain the previous nodes and any required arguement and kwargs must contain all requiered keyword arguements for the new method.

2. with **call_function**
```py
with traced_model.graph.insert_after(node):
    new_node= traced_model.graph.call_function(replacement_func,args,kwargs):
    node.replace_all_uses_with(new_node)
traced_model.graph.erase_node(node)
```
Here replcement is also a function, so, **call_function** is used. Args must contain the previous nodes and any required arguement and kwargs must contain all requiered keyword arguements for the new function.

3. with **call_module**
```python
traced_model.add_submodule(new_node_name,replacement_module)
with traced_model.graph.insert_before(node):
    new_node= traced_model.graph.call_module(new_node_name,args,{}):
    node.replace_all_uses_with(new_node)
traced_model.graph.erase_node(node)
```
Here replcement is also a module, so,it is first added to the structure and using the target name, **call_module** is called to create a new_node. Args must contain the previous nodes and kwargs must be empty as modules don't need keyword arguments.

If target node is **call_module** node we can directly replace the module from the structure by replacing the module in the parent module of the target with the replacement module.

Parent Module can be reached from the dictionary returned by **name_modules()** of the Graph Module and target of the node.

<center>
<br><img src="Images/gelu2relu.png"  width = "75%" alt="" /><br>
<br>
<b>GeLU to ReLU</b>
<br><img src="Images/onnx_gelu2relu.png"  width = "75%" alt="" /><br>
<b>Change in ONNX export of the model</b>
</center>

Here, we are changing GeLU with ReLU which can be done with pair pattern(GeLU) present in the model and any of call_fucntion or call_module of ReLU

## Main API (Explained)
[<p align = 'right'>Go To Top</p>](#table-of-content)

---
As previously said in section [Main API (Usages)](#main-api-usages), this API provides user with following functionalities in in [surgery.py](https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeltoolkit/browse/edgeai_torchtoolkit/v2/xao/surgery/surgery.py):

- [get_replacement_dict_default](#get_replacement_dict_default)
- [replace_unsupported_layer function](#replace_unsupported_layers-function)
- [SurgeryModule Class](#surgerymodule-class) 

### Getting Default Replacement Dict
Definition:
```
def get_replacement_dict_default():
```
- returns the default replacement dict that is used module
- it contains all the rules that are used for replacement
> **Note**: For efficient replacement rules should be organized in such a way that they will fulfil the following requirements:
> - replacements for pattern with larger number of nodes should come at the top in all type of replacements
>   - e.g.: Squeeze and Excite Layer before any Activation layer used in it.
> - if one layer is replaced with another instance of same type and another layer of a different type is to be replaced with same type, then former type of replacement should happen first then later should come.
>   - e.g. ReLU(inplace=True) to ReLU() before ReLU6() to ReLU()'

### Replacing Pattern with Replacement:
```python
def replace_unsupported_layer(model:nn.Module,replacement_dict=None):
```
- returns the module after all changes are done on the model, which are defined in the replacement_dict
- args:
  - model: the target model we have to surgery on
  - replacement_dict: the replacement rules in form of pattern(keys) and replacements(vals) 
    - if repalcement_dict == None: it will take default replacement dict
  
This function iterates through the replacement_dict and performs surgery according to the pair of pattern(keys) and replacement(values). 
<center>
<br><img src="Images/main_api_l1.png"  width = "75%" alt="" /><br>
</center>

### About SurgeryModule Class
This is actually wrapper class inherited attriutes from torch.nn.Modules defined as:
```py
class SurgeryModule(torch.nn.Module):
    
    def __init__(self, model, replacement_dict=None) -> None:
        super().__init__()
        self.replacement_dict=replacement_dict or get_replacement_dict_default()
        self.module = replace_unsuppoted_layers(model, self.replacement_dict)

    def forward(self,x,*args,**kwargs):
        return self.module(x,*args,**kwargs)

    def get_replacement_dict(self):
        return self.replacement_dict
```
- creates a wrapper module over the changed model after going through function [replace_unsupported_layer](#replace_unsupported_layers-function) along with the replacement_dict
- args:
  - model: the target model we have to surgery on
  - replacement_dict: the replacement rules in form of pattern(keys) and replacements(vals)
    - if repalcement_dict == None: it will take default replacement dict
- user can also get the replacement_dict used for surgery by using **get_replacement_dict** method on the changed model object
```python
replacement_dict=changed_model.get_replacement_dict() # a dict
```
---

## Different Types of Possible Replacement Rules
[<p align = 'right'>Go To Top</p>](#table-of-content)

---
There are 5 type of replacement rules implemented in the API, which are as following:
1. [Module to Module change](#replacement-type-1-module-to-module-change)
2. [Fucntion to Function change](#replacement-type-2-function-to-function-change)
3. [Function to Type/Module change](#replacement-type-3-function-to-typemodule-change)
4. [Type to Type/Module change](#replacement-type-4-type-to-typemodule-change)
5. [Custom Surgery Function](#replacement-type-5-custom-surgery-function)

### Replacement Type 1 (Module to Module change)
This approach is used when we don’t have to care about parameter (args and kwargs except nodes) passed to the pattern module.
It consists of two main process:
- symbolic trace both main module and pattern for graph
- searching of provided pattern in the node_list of the main model’s graph
- replacing them with the replacement

> Note: if modules in torch.nn directory are given for symbolic trace without any wrapper module, they will be converted to their respective functional counter parts.
> 
>  e.g.: nn.ReLU() -> nn.functional.relu()

**Searching pattern in the graph**

For this we use node_lists of both main model’s graph and pattern’s graph.
- After that this will be a simple linear pattern search in a linear list after discarding input nodes and output node.
- For comparing nodes in both graph, we first match their operation (op) and then follow according to the following table:
<center>
<br><img src ="Images/node_match.png">
</center>
Note: Till Date, Pattern with either single node or single input and single output is supported for replacement	

**Replacing the matches**

When we find the start and end of the matches of pattern in main model’s graph, replacement will be done as following cases:
1. If each of pattern and replacement is consists of single call_function or call_method node:
   - corresponding node for replacement’s node will be created and replaced with pattern’s node.
2. If start is a call_module node: 
    - corresponding module for that node will be changed from the structure
3.  else:
    - module will be added to the module structure and a new call_module node will be created for it for replacement in the graph.

Possible candidate for change:<br>
|Pattern|Replacement|
|------|------|
|SqueezeAndExcite 	| nn.Identity()|
|nn.Dropout(…)	    | nn.Dropout()|	
|nn.Hardswish()	    | nn.ReLU()|
|nn.GeLU()		    | nn.ReLU()|


All this functionalities are availabe in [replacer.py](https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeltoolkit/browse/edgeai_torchtoolkit/v2/xao/surgery/replacer.py) file which will be discussed later.

For patterns and Replacements which are composite models, some of them are defiend in [custom_modules.py](https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeltoolkit/browse/edgeai_torchtoolkit/v2/xao/surgery/custom_.py) file.

<center>
<br><img src="Images/SE2I.png"  width = "75%" alt="" /><br>
<br>
<b>Squeeze And Excite to Identity</b>
<br><img src="Images/onnx_SE2I.png"  width = "75%" alt="" /><br>
<b>Change in ONNX export of the model</b>
<br><img src="Images/gelu2relu.png"  width = "75%" alt="" /><br>
<br>
<b>GeLU to ReLU</b>
<br><img src="Images/onnx_gelu2relu.png"  width = "75%" alt="" /><br>
<b>Change in ONNX export of the model</b>
</center>

### Replacement Type 2 (Function to Function change)
This will call replace_function_node function that will create new node with replacement function as target with same args and kwargs.
Then it shifts the incoming and outgoing edges from the pattern to the new node.
<center>
<br><img src="Images/relu62relu.png"  width = "75%" alt="" /><br>
<br>
<b>ReLU6 to ReLU</b>
<br><img src="Images/onnx_relu62relu.png"  width = "75%" alt="" /><br>
<b>Change in ONNX export of the model</b>
</center>

### Replacement Type 3 (Function to Type/Module change)
This will call replace_function_node function that will create new node with replacement module after adding it to module structure as target with same args.
  - as module don’t take kwargs in their nodes.
  
Then it shifts the incoming and outgoing edges from the pattern to the new node

<center>
<br><img src="Images/silu2relu.png"  width = "75%" alt="" /><br>
<br>
<b>SiLU to ReLU</b>
<br><img src="Images/onnx_silu2relu.png"  width = "75%" alt="" /><br>
<b>Change in ONNX export of the model</b>
</center>

### Replacement Type 4 (Type to Type/Module change)
This replacement calls a function called replace_module_nodes which uses traditional approach of changing module from attribute of their parents
- so this useful if we know class of repetitive blocks in the model that doesn’t take parameters
  - e.g.: SEModule in timm models to Identity

It is better to be used before any symbolic trace is done on the model as after that all user defined models will lose their forward function and become instance Module class.
- but the (most of) module in torch.nn directory will retain their forward function even after symbolic trace until they are inside a wrapper module

<center>
<br><img src="Images/SE2I.png"  width = "75%" alt="" /><br>
<br>
<b>Squeeze and Excite to Identity</b>
<br><img src="Images/onnx_SE2I1.png"  width = "75%" alt="" /><br>
<b>Change in ONNX export of the model</b>
</center>

### Replacement Type 5 (Custom Surgery Function)
Users also can add their own surgery functions that will return the changed model paired their callable with any value other than callable.
- These function must be made where we have change parameters (args and kwargs)
- These must return the changed graph module after surgery.
    - e.g.:	
        1. LayerNorm2D (accepting input dimension of 4) to 1BatchNorm2D,
        <center>
        <br><img src="Images/layernorm2batchnorm2d.png"  width = "75%" alt="" /><br>
        <br>
        <b>LayerNorm to BatchNorm2D</b>
        <br><img src="Images/onnx_layernorm2batchnorm2d.png"  width = "75%" alt="" /><br>
        <b>Change in ONNX export of the model</b>
        </center>
		2. Conv. with kernel size greater than five : Equivalent series of Conv. of kernel 	  size 3 and a Conv. of kernel 5, etc
        <center>
        <br><img src="Images/conv_replacement.png"  width = "75%" alt="" /><br>
        <br>
        <b>Conv2D with kernel_size = 7  to equivalent series of Conv2D with kernel_size = 3 and a Conv2D with kernel_size = 5  </b>
        <br><img src="Images/onnx_conv_replacement.png"  width = "75%" alt="" /><br>
        <b>Change in ONNX export of the model</b>
        </center>
Example for these custom functions are available in [custom_surgery_functions.py](https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-modeltoolkit/browse/edgeai_torchtoolkit/v2/xao/surgery/custom_surgery_functions.py)

---
## Other Module Components
[<p align = 'right'>Go To Top</p>](#table-of-content)

---

---
## Module Tested on till Date
[<p align = 'right'>Go To Top</p>](#table-of-content)

---
The following models have been modified using the API and default replacement dictionary and also tested for whether an expected can forward pass through them or not.

**TorchVision Models:**
|Model|Rules|
|-----|-----|
|MobileNet V2|ReLU6 -> ReLU|
|MobileNet V3|SE -> identity<br>Hardswish -> ReLU|
|ConvNeXt|LayerNorm -> BatchNorm2D, <br>CNBlock of ConvNeXt to  its equivalent<br> 1. (Conv2D with kernel greater than 5<br> 2. LayerNorm -> BatchNorm2D<br>3. GeLU -> ReLU)|  
|EfficientNet|SiLU -> ReLU<br>SE -> Identity<br>|
|DeepLab V3|resize with size -> resize with scale factor<br>(any other depending on the version)|

**TIMM Models:**
|Model|Rules|
|-----|-----|
|MobileNet V3|SE -> identity<br>Hardswish -> ReLU|
|ConvNeXt|LayerNorm -> BatchNorm2D, <br>CNBlock of ConvNeXt to  its equivalent<br> 1. (Conv2D with kernel greater than 5<br> 2. LayerNorm -> BatchNorm2D<br>3. GeLU -> ReLU)|
|EfficientNet|SiLU -> ReLU<br>SE -> Identity<br>|
|ResNet|(no change just tried)|
|RegNet|SiLU -> ReLU<br>SE -> Identity<br>|

For **swin transformer model**, We encountered an issue associated with LayerNorms while replacing them with BatchNorm2d as it contains LayerNorm handling input of dimension 3 which can't be handled by BatchNorm2d.


**mmYolo Models:**

For [mmYolo](https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-mmyolo) Repository, surgery is given as a option to select whether to do or not for the user with default replacement dictionry in [export.py](https://bitbucket.itg.ti.com/projects/EDGEAI-ALGO/repos/edgeai-mmyolo/browse/dir)
|Model|Rules|
|-----|-----|
|yolox|\<all\>|
|yolo v5|\<all\>|
|yolo v6|\<all\>|
|yolo v7|\<all\>|
|yolo v8|\<all\>|

---
## Results
[<p align = 'right'>Go To Top</p>](#table-of-content)

---
This API can change Unsupported activation layers, different variation Squeeze And Excite Layer, Layer Norm 2D, redundant Identities, Conv2D, MaxPool2D, AvgPool2D module with kernel size greater than supported one. 
This API also provides user the feature for selective replacement of layers
- after they satisfy the requirement for the pattern or a custom surgery function.

Here are some of the models tested and trained along with their accuracies before and after change:

|Model|Accuracy before |Accuracy After Change|
|-----|----|----|
|MobileNet V3 (large) (torchvision)|69.7|68.4|
|EfficientNet B0 (torchvision)|70.3|67.4|
||||
> **Note**: 
> - Here changes are the replacement specified in default replacement dictionary.
> - Each Model is only trained for 100 epochs

--- 

<p align ='right'>
Developed and Documented by:<br>
<b>Kunal Ranjan Patel</b><br>
Embedded Software Intern Under Mentor:<br> <b>Manu Mathew</b><br>
Texas Instrument, Bangalore, India
</p>