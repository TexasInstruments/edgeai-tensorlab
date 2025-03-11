# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings
from functools import partial

import numpy as np
import onnx
import torch
# from mmcv import Config, DictAction

import mmdet.utils
from mmdet.core.export import build_model_from_cfg, preprocess_example_input
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector

from mmdet.utils.edgeai_utils import XMMDetQuantTestModule, save_model_proto, mmdet_load_checkpoint, is_mmdet_quant_module
from mmdet.utils.edgeai_utils import save_model_proto, patch_onnx_file

import edgeai_torchtoolkit


def pytorch2onnx(args,
                 cfg,
                 model,
                 input_img,
                 input_shape,
                 normalize_cfg,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None,
                 skip_postprocess=False):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    img_list, img_meta_list = [one_img], [[one_meta]]

    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post '
                      'process, especially two stage detectors!')
        model.forward = model.forward_dummy
        torch.onnx.export(
            model,
            one_img,
            output_file,
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=show,
            opset_version=opset_version)

        print(f'Successfully exported ONNX model without '
              f'post process: {output_file}')
        return

    # replace original forward function
    origin_forward = model.forward
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)

    output_names = ['dets', 'labels']
    model_org = model.module if is_mmdet_quant_module(model) else model
    if model_org.with_mask:
        output_names.append('masks')
    input_name = 'input'
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            input_name: {
                0: 'batch',
                2: 'height',
                3: 'width'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        if model_org.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}

    # export the full onnx file
    model.with_intermediate_outputs = True
    torch.onnx.export(
        model,
        img_list,
        output_file,
        input_names=[input_name],
        output_names=None,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
    )

    # Apply path to Reduce OP to make it run in onnxruntime
    patch_onnx_file(output_file, output_file)

    onnx_model = onnx.load(output_file)
    feature_names = [node.name for node in onnx_model.graph.output[2:]]
    for opt in onnx_model.graph.output[2:]:
        onnx_model.graph.output.remove(opt)
    #
    for opt_name, opt in zip(output_names, onnx_model.graph.output[:2]):
        name_changed = False
        for node in onnx_model.graph.node:
            for node_o_idx in range(len(node.output)):
                if node.output[node_o_idx] == opt.name:
                    node.output[node_o_idx] = opt_name
                    opt.name = opt_name
                    name_changed = True
                    break
                #
            #
            if name_changed:
                break
            #
        #
    #
    # make model
    opset = onnx.OperatorSetIdProto()
    opset.version = opset_version
    onnx_model = onnx.helper.make_model(onnx_model.graph, opset_imports=[opset])
    # check model and save
    onnx.checker.check_model(onnx_model)
    # shape inference to make it easy for inference
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_file)
    # write prototxt
    save_model_proto(cfg, model, img_list, output_file, feature_names=feature_names, output_names=output_names)
    model.with_intermediate_outputs = False

    # # export the partial onnx file
    # onnx_proto_file = osp.splitext(output_file)[0] + '-proto.onnx'
    # model.forward = model.forward_dummy
    # torch.onnx.export(
    #     model,
    #     img_list[0],
    #     onnx_proto_file,
    #     input_names=[input_name],
    #     output_names=None,
    #     export_params=True,
    #     keep_initializers_as_inputs=True,
    #     do_constant_folding=True,
    #     verbose=show,
    #     opset_version=opset_version,
    #     dynamic_axes=dynamic_axes)
    # onnx_model = onnx.load(onnx_proto_file)
    # feature_names = [node.name for node in onnx_model.graph.output]
    # # shape inference is required to support onnx+proto detection models in edgeai-tidl-tools
    # onnx.shape_inference.infer_shapes_path(onnx_proto_file, onnx_proto_file)
    # save_model_proto(cfg, model, img_list, onnx_proto_file, feature_names=feature_names, output_names=output_names)

    model.forward = origin_forward

    # get the custom op path
    ort_custom_op_path = ''
    try:
        from mmcv.ops import get_onnxruntime_op_path
        ort_custom_op_path = get_onnxruntime_op_path()
    except (ImportError, ModuleNotFoundError):
        warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')

    if do_simplify:
        import onnxsim

        from mmdet import digit_version

        min_required_version = '0.3.0'
        assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

        input_dic = {'input': img_list[0].detach().cpu().numpy()}
        model_opt, check_ok = onnxsim.simplify(
            output_file,
            input_data=input_dic,
            custom_lib=ort_custom_op_path,
            dynamic_input_shape=dynamic_export)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {output_file}')

    # onnx model does not have the quant hooks, so in quant mode the outputs won't match
    if verify:
        from mmdet.core import get_classes
        from mmdet.apis import show_result_pyplot
        model.CLASSES = get_classes(dataset)
        num_classes = len(model.CLASSES)
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # wrap onnx model
        onnx_model = ONNXRuntimeDetector(output_file, model.CLASSES, 0)
        if dynamic_export:
            # scale up to test dynamic shape
            h, w = [int((_ * 1.5) // 32 * 32) for _ in input_shape[2:]]
            h, w = min(1344, h), min(1344, w)
            input_config['input_shape'] = (1, 3, h, w)

        if test_img is None:
            input_config['input_path'] = input_img

        # prepare input once again
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]

        # get pytorch output
        with torch.no_grad():
            pytorch_results = model(
                img_list,
                img_metas=img_meta_list,
                return_loss=False,
                rescale=True)[0]

        img_list = [_.cuda().contiguous() for _ in img_list]
        if dynamic_export:
            img_list = img_list + [_.flip(-1).contiguous() for _ in img_list]
            img_meta_list = img_meta_list * 2
        # get onnx output
        onnx_results = onnx_model(
            img_list, img_metas=img_meta_list, return_loss=False)[0]
        # visualize predictions
        score_thr = 0.3
        if show:
            out_file_ort, out_file_pt = None, None
        else:
            out_file_ort, out_file_pt = 'show-ort.png', 'show-pt.png'

        show_img = one_meta['show_img']
        model.show_result(
            show_img,
            pytorch_results,
            score_thr=score_thr,
            show=True,
            win_name='PyTorch',
            out_file=out_file_pt)
        onnx_model.show_result(
            show_img,
            onnx_results,
            score_thr=score_thr,
            show=True,
            win_name='ONNXRuntime',
            out_file=out_file_ort)

        # compare a part of result
        if model.with_mask:
            compare_pairs = list(zip(onnx_results, pytorch_results))
        else:
            compare_pairs = [(onnx_results, pytorch_results)]
        err_msg = 'The numerical values are different between Pytorch' + \
                  ' and ONNX, but it does not necessarily mean the' + \
                  ' exported ONNX model is problematic.'
        # check the numerical value
        for onnx_res, pytorch_res in compare_pairs:
            for o_res, p_res in zip(onnx_res, pytorch_res):
                np.testing.assert_allclose(
                    o_res, p_res, rtol=1e-03, atol=1e-05, err_msg=err_msg)
        print('The numerical values are the same between Pytorch and ONNX')


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) <= 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0] if len(norm_config_li)>0 else dict(mean=0.0, std=1.0)
    return norm_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='Dataset name. This argument is deprecated and will be removed \
        in future releases.')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data.This argument \
        is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data. '
        'This argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action='DictAction',
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    parser.add_argument(
        '--skip-postprocess',
        action='store_true',
        help='Whether to export model without post process. Experimental '
        'option. We do not guarantee the correctness of the exported '
        'model.')
    args = parser.parse_args()
    return args


# https://github.com/openai/CLIP/issues/79
# From https://github.com/pytorch/pytorch/blob/2efe4d809fdc94501fc38bf429e9a8d4205b51b6/torch/utils/tensorboard/_pytorch_graph.py#L384
def _node_getitem(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)


def main(args):
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    # 2023/11/13 added to support older mmcv on latest Pytorch
    # TODO: remove this after upgrading mmdet/mmcv
    if not hasattr(torch._C.Node, '__getitem__'):
        torch._C.Node.__getitem__ = _node_getitem
    #

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    # cfg = Config.fromfile(args.config)
    cfg = args.config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    if hasattr(cfg, 'resize_with_scale_factor') and cfg.resize_with_scale_factor:
        torch.nn.functional._interpolate_orig = torch.nn.functional.interpolate
        torch.nn.functional.interpolate = edgeai_torchtoolkit.xnn.layers.resize_with_scale_factor

    # build the model and load checkpoint
    model = build_model_from_cfg(args.config, args.checkpoint,
                                 args.cfg_options)

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)

    # convert model to onnx file
    pytorch2onnx(
        args,
        cfg,
        model,
        args.input_img,
        input_shape,
        normalize_cfg,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export,
        skip_postprocess=args.skip_postprocess)

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)


if __name__ == '__main__':
    args = parse_args()
    main(args)

