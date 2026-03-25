import sys
import importlib
import os
import torch
from torch import nn
from torch.nn.modules import Module
from torchinfo.formatting import FormattingOptions
from torchinfo.layer_info import *
from torchinfo.layer_info import LayerInfo
from torchinfo.torchinfo import *
from torchinfo.model_statistics import *
_cached_forward_pass: dict[str, list[LayerInfo]] = {}


def import_file_or_folder(folder_or_file_name, package_name=None, force_import=False):
    if folder_or_file_name.endswith(os.sep):
        folder_or_file_name = folder_or_file_name[:-1]
    #
    if folder_or_file_name.endswith('.py'):
        folder_or_file_name = folder_or_file_name[:-3]
    #
    parent_folder = os.path.dirname(folder_or_file_name)
    basename = os.path.basename(folder_or_file_name)
    if force_import:
        sys.modules.pop(basename, None)
    #
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, package_name or __name__)
    sys.path.pop(0)
    return imported_module

class CustomLayerInfo(LayerInfo):
    def __init__(self, var_name: str, module: Module, depth: int, parent_info: LayerInfo | None = None) -> None:
        super().__init__(var_name, module, depth, parent_info)
        # print(self.calculate_macs())
        
    @staticmethod
    def get_param_count(
        module: nn.Module, name: str, param: torch.Tensor
    ) -> tuple[int, str]:
        """
        Get count of number of params, accounting for mask.

        Masked models save parameters with the suffix "_orig" added.
        They have a buffer ending with "_mask" which has only 0s and 1s.
        If a mask exists, the sum of 1s in mask is number of params.
        """
        if name.endswith("_orig"):
            without_suffix = name[:-5]
            pruned_weights = rgetattr(module, f"{without_suffix}_mask")
            if pruned_weights is not None:
                parameter_count = int(torch.sum(pruned_weights))
                return parameter_count, without_suffix
        # print(type(module).__name__,prod(param.shape))
        param= param[param.nonzero(as_tuple=True)]
        # print(prod(param.shape))
        return (param).nelement(), name
    
    def calculate_macs(self) -> None:
        """
        Set MACs using the module's parameters and layer's output size, which is
        used for computing number of operations for Conv layers.

        Please note: Returned MACs is the number of MACs for the full tensor,
        i.e., taking the batch-dimension into account.
        """
        for name, param in self.module.named_parameters():
            cur_params, name = self.get_param_count(self.module, name, param)
            if name in ("weight", "bias"):
                # ignore C when calculating Mult-Adds in ConvNd
                if "Conv" in self.class_name:
                    curr_macs = cur_params * prod(self.output_size[:1] + self.output_size[2:])
                    # print(f'{self.module}_{name}:{cur_params}({list(param.shape)}*{self.output_size[:1]}*{self.output_size[2:]}={curr_macs})')
                    self.macs += int(curr_macs)
                else:
                    self.macs += self.output_size[0] * cur_params
            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name or "bias" in name:
                self.macs += prod(self.output_size[:2]) * cur_params
        pass



class CustomModelStatistics(ModelStatistics):
    def __init__(self, summary_list: list[LayerInfo], input_size, total_input_size: int, formatting: FormattingOptions) -> None:
        super().__init__(summary_list, input_size, total_input_size, formatting)
    
    def __repr__(self) -> str:
        """Print results of the summary."""
        divider = "=" * self.formatting.get_total_width()
        total_params = ModelStatistics.format_output_num(
            self.total_params, self.formatting.params_units
        )
        trainable_params = ModelStatistics.format_output_num(
            self.trainable_params, self.formatting.params_units
        )
        non_trainable_params = ModelStatistics.format_output_num(
            self.total_params - self.trainable_params, self.formatting.params_units
        )
        all_layers = self.formatting.layers_to_str(self.summary_list, self.total_params)
        summary_str = (
            f"{divider}\n"
            f"{self.formatting.header_row()}{divider}\n"
            f"{all_layers}{divider}\n"
            f"Total params{total_params}\n"
            f"Trainable params{trainable_params}\n"
            f"Non-trainable params{non_trainable_params}\n"
        )
        if self.input_size:
            macs = ModelStatistics.format_output_num(
                self.total_mult_adds, self.formatting.macs_units
            )
            input_size = self.to_megabytes(self.total_input)
            output_bytes = self.to_megabytes(self.total_output_bytes)
            param_bytes = self.to_megabytes(self.total_param_bytes)
            total_bytes = self.to_megabytes(
                self.total_input + self.total_output_bytes + self.total_param_bytes
            )
            summary_str += (
                f"Total mult-adds{macs}\n{divider}\n"
                f"Input size (MB): {input_size:0.2f}\n"
                f"Forward/backward pass size (MB): {output_bytes:0.2f}\n"
                f"Params size (MB): {param_bytes:0.2f}\n"
                f"Estimated Total Size (MB): {total_bytes:0.2f}\n"
            )
        summary_str += divider
        return summary_str


def construct_pre_hook(
    global_layer_info: dict[int, LayerInfo],
    summary_list: list[LayerInfo],
    layer_ids: set[int],
    var_name: str,
    curr_depth: int,
    parent_info: LayerInfo | None,
) -> Callable[[nn.Module, Any], None]:
    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate layer information."""
        del inputs
        info = CustomLayerInfo(var_name, module, curr_depth, parent_info)
        info.calculate_num_params()
        info.check_recursive(layer_ids)
        summary_list.append(info)
        layer_ids.add(info.layer_id)
        global_layer_info[info.layer_id] = info

    return pre_hook
    
def apply_hooks(
    model_name: str,
    module: nn.Module,
    input_data: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
) -> tuple[
    list[LayerInfo],
    dict[int, LayerInfo],
    dict[int, tuple[RemovableHandle, RemovableHandle]],
]:
    """
    If input_data is provided, recursively adds hooks to all layers of the model.
    Else, fills summary_list with layer info without computing a
    forward pass through the network.
    """
    summary_list: list[LayerInfo] = []
    layer_ids: set[int] = set()  # Used to optimize is_recursive()
    global_layer_info: dict[int, LayerInfo] = {}
    hooks: dict[int, tuple[RemovableHandle, RemovableHandle]] = {}
    stack: list[tuple[str, nn.Module, int, LayerInfo | None]] = [
        (model_name, module, 0, None)
    ]
    while stack:
        var_name, module, curr_depth, parent_info = stack.pop()
        module_id = id(module)

        # Fallback is used if the layer's pre-hook is never called, for example in
        # ModuleLists or Sequentials.
        global_layer_info[module_id] = CustomLayerInfo(
            var_name, module, curr_depth, parent_info
        )
        pre_hook = construct_pre_hook(
            global_layer_info,
            summary_list,
            layer_ids,
            var_name,
            curr_depth,
            parent_info,
        )
        if input_data is None or isinstance(module, WRAPPER_MODULES):
            pre_hook(module, None)
        else:
            # Register the hook using the last layer that uses this module.
            if module_id in hooks:
                for hook in hooks[module_id]:
                    hook.remove()
            hooks[module_id] = (
                module.register_forward_pre_hook(pre_hook),
                module.register_forward_hook(
                    construct_hook(global_layer_info, batch_dim)
                ),
            )

        # Replaces the equivalent recursive call by appending all of the
        # subsequent the module children stack calls in the encountered order.
        # Note: module.named_modules(remove_duplicate=False) doesn't work for
        # some unknown reason (infinite recursion)
        stack += [
            (name, mod, curr_depth + 1, global_layer_info[module_id])
            # pylint: disable=protected-access
            for name, mod in reversed(module._modules.items())
            if mod is not None
        ]
    return summary_list, global_layer_info, hooks

def forward_pass(
    model: nn.Module,
    x: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
    cache_forward_pass: bool,
    device: torch.device | None,
    mode: Mode,
    **kwargs: Any,
) -> list[LayerInfo]:
    """Perform a forward pass on the model using forward hooks."""
    global _cached_forward_pass  # pylint: disable=global-variable-not-assigned
    model_name = model.__class__.__name__
    if cache_forward_pass and model_name in _cached_forward_pass:
        return _cached_forward_pass[model_name]

    summary_list, _, hooks = apply_hooks(model_name, model, x, batch_dim)
    if x is None:
        set_children_layers(summary_list)
        return summary_list

    kwargs = set_device(kwargs, device)
    saved_model_mode = model.training
    try:
        if mode == Mode.TRAIN:
            model.train()
        elif mode == Mode.EVAL:
            model.eval()
        else:
            raise RuntimeError(
                f"Specified model mode ({list(Mode)}) not recognized: {mode}"
            )

        with torch.no_grad():
            model = model if device is None else model.to(device)
            if isinstance(x, (list, tuple)):
                _ = model(*x, **kwargs)
            elif isinstance(x, dict):
                _ = model(**x, **kwargs)
            else:
                # Should not reach this point, since process_input_data ensures
                # x is either a list, tuple, or dict
                raise ValueError("Unknown input type")
    except Exception as e:
        executed_layers = [layer for layer in summary_list if layer.executed]
        raise RuntimeError(
            "Failed to run torchinfo. See above stack traces for more details. "
            f"Executed layers up to: {executed_layers}"
        ) from e
    finally:
        if hooks:
            for pre_hook, hook in hooks.values():
                pre_hook.remove()
                hook.remove()
        model.train(saved_model_mode)

    add_missing_container_layers(summary_list)
    set_children_layers(summary_list)

    _cached_forward_pass[model_name] = summary_list
    return summary_list


def summary(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE | None = None,
    input_data: INPUT_DATA_TYPE | None = None,
    batch_dim: int | None = None,
    cache_forward_pass: bool | None = None,
    col_names: Iterable[str] | None = None,
    col_width: int = 25,
    depth: int = 3,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str | None = None,
    row_settings: Iterable[str] | None = None,
    verbose: int | None = None,
    **kwargs: Any,
) -> ModelStatistics:
    """
    Summarize the given PyTorch model. Summarized information includes:
        1) Layer names,
        2) input/output shapes,
        3) kernel shape,
        4) # of parameters,
        5) # of operations (Mult-Adds),
        6) whether layer is trainable

    NOTE: If neither input_data or input_size are provided, no forward pass through the
    network is performed, and the provided model information is limited to layer names.

    Args:
        model (nn.Module):
                PyTorch model to summarize. The model should be fully in either train()
                or eval() mode. If layers are not all in the same mode, running summary
                may have side effects on batchnorm or dropout statistics. If you
                encounter an issue with this, please open a GitHub issue.

        input_size (Sequence of Sizes):
                Shape of input data as a List/Tuple/torch.Size
                (dtypes must match model input, default is FloatTensors).
                You should include batch size in the tuple.
                Default: None

        input_data (Sequence of Tensors):
                Arguments for the model's forward pass (dtypes inferred).
                If the forward() function takes several parameters, pass in a list of
                args or a dict of kwargs (if your forward() function takes in a dict
                as its only argument, wrap it in a list).
                Default: None

        batch_dim (int):
                Batch_dimension of input data. If batch_dim is None, assume
                input_data / input_size contains the batch dimension, which is used
                in all calculations. Else, expand all tensors to contain the batch_dim.
                Specifying batch_dim can be an runtime optimization, since if batch_dim
                is specified, torchinfo uses a batch size of 1 for the forward pass.
                Default: None

        cache_forward_pass (bool):
                If True, cache the run of the forward() function using the model
                class name as the key. If the forward pass is an expensive operation,
                this can make it easier to modify the formatting of your model
                summary, e.g. changing the depth or enabled column types, especially
                in Jupyter Notebooks.
                WARNING: Modifying the model architecture or input data/input size when
                this feature is enabled does not invalidate the cache or re-run the
                forward pass, and can cause incorrect summaries as a result.
                Default: False

        col_names (Iterable[str]):
                Specify which columns to show in the output. Currently supported: (
                    "input_size",
                    "output_size",
                    "num_params",
                    "params_percent",
                    "kernel_size",
                    "mult_adds",
                    "trainable",
                )
                Default: ("output_size", "num_params")
                If input_data / input_size are not provided, only "num_params" is used.

        col_width (int):
                Width of each column.
                Default: 25

        depth (int):
                Depth of nested layers to display (e.g. Sequentials).
                Nested layers below this depth will not be displayed in the summary.
                Default: 3

        device (torch.Device):
                Uses this torch device for model and input_data.
                If not specified, uses the dtype of input_data if given, or the
                parameters of the model. Otherwise, uses the result of
                torch.cuda.is_available().
                Default: None

        dtypes (List[torch.dtype]):
                If you use input_size, torchinfo assumes your input uses FloatTensors.
                If your model use a different data type, specify that dtype.
                For multiple inputs, specify the size of both inputs, and
                also specify the types of each parameter here.
                Default: None

        mode (str)
                Either "train" or "eval", which determines whether we call
                model.train() or model.eval() before calling summary().
                Default: "eval".

        row_settings (Iterable[str]):
                Specify which features to show in a row. Currently supported: (
                    "ascii_only",
                    "depth",
                    "var_names",
                )
                Default: ("depth",)

        verbose (int):
                0 (quiet): No output
                1 (default): Print model summary
                2 (verbose): Show weight and bias layers in full detail
                Default: 1
                If using a Juypter Notebook or Google Colab, the default is 0.

        **kwargs:
                Other arguments used in `model.forward` function. Passing *args is no
                longer supported.

    Return:
        ModelStatistics object
                See torchinfo/model_statistics.py for more information.
    """
    input_data_specified = input_data is not None or input_size is not None
    if col_names is None:
        columns = (
            DEFAULT_COLUMN_NAMES
            if input_data_specified
            else (ColumnSettings.NUM_PARAMS,)
        )
    else:
        columns = tuple(ColumnSettings(name) for name in col_names)

    if row_settings is None:
        rows = DEFAULT_ROW_SETTINGS
    else:
        rows = {RowSettings(name) for name in row_settings}

    if mode is None:
        model_mode = Mode.EVAL
    else:
        model_mode = Mode(mode)

    if verbose is None:
        # pylint: disable=no-member
        verbose = 0 if hasattr(sys, "ps1") and sys.ps1 else 1

    if cache_forward_pass is None:
        # In the future, this may be enabled by default in Jupyter Notebooks
        cache_forward_pass = False

    if device is None:
        device = get_device(model, input_data)
    elif isinstance(device, str):
        device = torch.device(device)

    validate_user_params(
        input_data, input_size, columns, col_width, device, dtypes, verbose
    )

    x, correct_input_size = process_input(
        input_data, input_size, batch_dim, device, dtypes
    )
    summary_list = forward_pass(
        model, x, batch_dim, cache_forward_pass, device, model_mode, **kwargs
    )
    formatting = FormattingOptions(depth, verbose, columns, col_width, rows)
    results = CustomModelStatistics(
        summary_list, correct_input_size, get_total_memory_used(x), formatting
    )
    if verbose > Verbosity.QUIET:
        print(results)
    return results

