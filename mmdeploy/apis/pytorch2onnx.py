# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Union

import torch
import mmengine
from mmengine.runner import load_checkpoint
from mmdeploy.utils import build_model_from_cfg
from mmdet.apis import init_detector

from .core import PIPELINE_MANAGER

def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


@PIPELINE_MANAGER.register_pipeline()
def torch2onnx(img: Any,
               work_dir: str,
               save_file: str,
               deploy_cfg: Union[str, mmengine.Config],
               model_cfg: Union[str, mmengine.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               torch_model = None):
    """Convert PyTorch model to ONNX model.

    Examples:
        >>> from mmdeploy.apis import torch2onnx
        >>> img = 'demo.jpg'
        >>> work_dir = 'work_dir'
        >>> save_file = 'fcos.onnx'
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_onnxruntime_dynamic.py')
        >>> model_cfg = ('mmdetection/configs/fcos/'
                         'fcos_r50_caffe_fpn_gn-head_1x_coco.py')
        >>> model_checkpoint = ('checkpoints/'
                                'fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth')
        >>> device = 'cpu'
        >>> torch2onnx(img, work_dir, save_file, deploy_cfg, \
            model_cfg, model_checkpoint, device)

    Args:
        img (str | np.ndarray | torch.Tensor): Input image used to assist
            converting model.
        work_dir (str): A working directory to save files.
        save_file (str): Filename to save onnx model.
        deploy_cfg (str | mmengine.Config): Deployment config file or
            Config object.
        model_cfg (str | mmengine.Config): Model config file or Config object.
        model_checkpoint (str): A checkpoint path of PyTorch model,
            defaults to `None`.
        device (str): A string specifying device type, defaults to 'cuda:0'.
    """

    from mmdeploy.apis.core.pipeline_manager import no_mp
    from mmdeploy.utils import (Backend, get_backend, get_dynamic_axes,
                                get_input_shape, get_onnx_config, load_config)
    from .onnx import export

    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmengine.mkdir_or_exist(osp.abspath(work_dir))

    input_shape = get_input_shape(deploy_cfg)

    # create model an inputs
    from mmdeploy.apis import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    if torch_model is None:
        torch_model = task_processor.build_pytorch_model(model_checkpoint)

    input_metas=None
    if not isinstance(img, str):
        model_inputs = img
    elif hasattr(model_cfg, 'input_size'):
        if isinstance(model_cfg.input_size,tuple):
            img_size = model_cfg.input_size
        else:
            img_size = (model_cfg.input_size,model_cfg.input_size)
        model_inputs = torch.randn(1, 3,
                             *img_size).to(device)
    else:
        data, model_inputs = task_processor.create_input(
            img,
            input_shape,
            data_preprocessor=getattr(torch_model, 'data_preprocessor', None))

        if isinstance(model_inputs, list) and len(model_inputs) == 1:
            model_inputs = model_inputs[0]
        data_samples = data['data_samples']
        input_metas = {'data_samples': data_samples, 'mode': 'predict'}

    # export to onnx
    context_info = dict()
    context_info['deploy_cfg'] = deploy_cfg
    # output_prefix = osp.join(work_dir,
    #                          osp.splitext(osp.basename(save_file))[0])
    backend = get_backend(deploy_cfg).value

    onnx_cfg = get_onnx_config(deploy_cfg)
    opset_version = onnx_cfg.get('opset_version', 17)

    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get(
        'verbose', False)
    keep_initializers_as_inputs = onnx_cfg.get('keep_initializers_as_inputs',
                                               True)
    optimize = onnx_cfg.get('optimize', False)

    if backend == Backend.NCNN.value:
        """NCNN backend needs a precise blob counts, while using onnx optimizer
        will merge duplicate initilizers without reference count."""
        optimize = False
    with no_mp():
        export(
            torch_model,
            model_inputs,
            input_metas=input_metas,
            save_file=save_file,
            backend=backend,
            input_names=input_names,
            output_names=output_names,
            context_info=context_info,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            optimize=optimize)


@PIPELINE_MANAGER.register_pipeline()
def model2onnx(img: Any,
               work_dir: str,
               save_file: str,
               model_cfg: Union[str, mmengine.Config],
               deploy_cfg: Union[str, mmengine.Config],
               model: Any = None,
               device: str = 'cpu',
               simplify: bool = True):
    """Convert PyTorch model to ONNX model.

    Examples:
        >>> from mmdeploy.apis import torch2onnx
        >>> img = 'demo.jpg'
        >>> work_dir = 'work_dir'
        >>> save_file = 'fcos.onnx'
        >>> deploy_cfg = ('configs/mmdet/detection/'
                          'detection_onnxruntime_dynamic.py')
        >>> model_cfg = ('mmdetection/configs/fcos/'
                         'fcos_r50_caffe_fpn_gn-head_1x_coco.py')
        >>> model_checkpoint = ('checkpoints/'
                                'fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth')
        >>> device = 'cpu'
        >>> torch2onnx(img, work_dir, save_file, deploy_cfg, \
            model_cfg, model_checkpoint, device)

    Args:
        img (str | np.ndarray | torch.Tensor): Input image used to assist
            converting model.
        work_dir (str): A working directory to save files.
        save_file (str): Filename to save onnx model.
        deploy_cfg (str | mmengine.Config): Deployment config file or
            Config object.
        model_cfg (str | mmengine.Config): Model config file or Config object.
        model (Module): PyTorch model instance.
        device (str): A string specifying device type, defaults to 'cuda:0'.
    """

    from mmdeploy.apis.core.pipeline_manager import no_mp
    from mmdeploy.utils import (Backend, get_backend, get_dynamic_axes,
                                get_input_shape, get_onnx_config, load_config)
    from .onnx import export

    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmengine.mkdir_or_exist(osp.abspath(work_dir))

    input_shape = get_input_shape(deploy_cfg)

    # create model an inputs
    from mmdeploy.apis import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    torch_model = model
    from mmengine.model import revert_sync_batchnorm
    torch_model = revert_sync_batchnorm(torch_model)
    data, model_inputs = task_processor.create_input(
        img,
        input_shape,
        data_preprocessor=getattr(torch_model, 'data_preprocessor', None))

    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    data_samples = data['data_samples']
    input_metas = {'data_samples': data_samples, 'mode': 'predict'}

    # export to onnx
    context_info = dict()
    context_info['deploy_cfg'] = deploy_cfg
    output_prefix = osp.join(work_dir,
                             osp.splitext(osp.basename(save_file))[0])
    backend = get_backend(deploy_cfg).value

    onnx_cfg = get_onnx_config(deploy_cfg)
    opset_version = onnx_cfg.get('opset_version', 17)

    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get(
        'verbose', False)
    keep_initializers_as_inputs = onnx_cfg.get('keep_initializers_as_inputs',
                                               True)
    optimize = onnx_cfg.get('optimize', False)
    if backend == Backend.NCNN.value:
        """NCNN backend needs a precise blob counts, while using onnx optimizer
        will merge duplicate initilizers without reference count."""
        optimize = False

    if hasattr(torch_model, 'quant_convert') and hasattr(torch_model.backbone, 'convert'):
        torch_model = torch_model.quant_convert()
    else:
        torch_model.to(device=device)

    print("Model is now converted, attempting to onnx export!")
    with no_mp():
        export(
            torch_model,
            model_inputs,
            input_metas=input_metas,
            output_path_prefix=output_prefix,
            backend=backend,
            input_names=input_names,
            output_names=output_names,
            context_info=context_info,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            optimize=optimize)
    print("Model Export is complete!")

    if simplify:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(output_prefix + '.onnx')
        onnx_model, check = simplify(onnx_model)
        onnx.save(onnx_model, output_prefix + '_simplified.onnx')
