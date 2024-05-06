from collections import OrderedDict
from typing import Dict, Optional
import copy
from torch import nn


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class IntermediateModuleGetter(nn.ModuleDict):
    """
    feature_extraction.create_feature_extractor provides more granularity compared to IntermediateLayerGetter,
      but it uses torch.fx and it doesn't keep the original class names.
    IntermediateModuleGetter is our modification to IntermediateLayerGetter - this provides more granulatrity,
      but it still has one of the original limitations - forward fuction of the model is not preserved -
      so the forward function of the model passed should not use functionals.
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_modules()]):
            raise ValueError("return_layers are not present in model")

        # logic to find only upto the required laeyrs and remove the remaining
        # otherwise DistributedDataParallel will complain that certain parameters were not used
        self.return_layers = return_layers
        return_layers_copy = copy.deepcopy(return_layers)
        layers = OrderedDict()
        for iter in range(len(return_layers)):
            for name, child in model.named_children():
                if name not in layers:
                    layers[name] = child
                #
                layer_found = None
                for cn, cm in child.named_modules():
                    cn_full = name + '.' + cn if cn else name
                    if cn_full in return_layers_copy:
                        layer_found = cn_full
                    #
                #
                if layer_found:
                    del return_layers_copy[layer_found]
                    break
                #
            #
        #
        assert len(return_layers_copy) == 0, f'some intermediate layers could be found in the model: {return_layers_copy.keys()}'
        super().__init__(layers)

        for name, module in self.named_modules():
            module._module_name = name
            module.register_forward_hook(self._forward_hook_fn)
        #

    def _forward_hook_fn(self, m, input, output):
        if m._module_name in self.return_layers:
            return_layer_name = self.return_layers[m._module_name]
            self._return_outputs[return_layer_name] = output

    def forward(self, x):
        self._return_outputs = OrderedDict()
        for name, module in self.items():
            x = module(x)
        #
        assert len(self._return_outputs) > 0, 'Could not find intermediate outputs to return'
        return self._return_outputs


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
