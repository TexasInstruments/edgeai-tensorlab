# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from .constants import IR, SDK_TASK_MAP, Backend, Codebase, Task
from .device import parse_cuda_device_id, parse_device_id, parse_device_type
from .env import get_backend_version, get_codebase_version, get_library_version
from .utils import get_file_path, get_root_logger, target_wrapper, build_model_from_cfg
from .save_model import save_model_proto

__all__ = [
    'SDK_TASK_MAP', 'IR', 'Backend', 'Codebase', 'Task',
    'parse_cuda_device_id', 'get_library_version', 'get_codebase_version',
    'get_backend_version', 'parse_device_id', 'get_file_path', 'build_model_from_cfg',
    'get_root_logger', 'target_wrapper', 'parse_device_type','save_model_proto'
]

if importlib.util.find_spec('mmcv') is not None:
    # yapf: disable
    from .config_utils import (cfg_apply_marks, get_backend,
                               get_backend_config, get_calib_config,
                               get_calib_filename, get_codebase,
                               get_codebase_config, get_common_config,
                               get_dynamic_axes, get_input_shape,
                               get_ir_config, get_model_inputs,
                               get_normalization, get_onnx_config,
                               get_partition_config, get_precision,
                               get_quantization_config, get_rknn_quantization,
                               get_task_type, is_dynamic_batch,
                               is_dynamic_shape, load_config)

    # yapf: enable

    __all__ += [
        'cfg_apply_marks', 'get_backend', 'get_backend_config',
        'get_calib_config', 'get_calib_filename', 'get_codebase',
        'get_codebase_config', 'get_common_config', 'get_dynamic_axes',
        'get_input_shape', 'get_ir_config', 'get_model_inputs',
        'get_onnx_config', 'get_partition_config', 'get_quantization_config',
        'get_precision', 'get_task_type', 'is_dynamic_batch',
        'is_dynamic_shape', 'load_config', 'get_rknn_quantization',
        'get_normalization'
    ]
