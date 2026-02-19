import traceback
from typing import OrderedDict
import torch
from torch import nn

from tv_models import construct_tv_model, TVMODEL_CONFIGS, TVMODEL_NAMES

import os
os.chdir(os.path.dirname(__file__))

from simple_onnx_test import simple_onnx_test # type: ignore

from edgeai_torchmodelopt import xmodelopt

import random
import string
import copy


# def my_onnx_save(onnxprogram, destination):
#     #### VIGNESH: UNder construction TODO
#     from torch.onnx._internal._lazy_import import onnx, onnxscript_apis, onnxscript_ir as ir

#     model = onnxprogram.model
#     original_initializers = copy.copy(model.graph.initializers)
#     original_inputs = copy.copy(model.graph.inputs)
#     model.graph.initializers.clear()
#     model.graph.inputs.extend(original_initializers.values())
    
#     try:
#         res = ir.save(model, destination) 
#     finally:
#         # Revert the changes to the model
#         model.graph.initializers.update(original_initializers)
        
#         model.graph.inputs.clear()
#         model.graph.inputs.extend(original_inputs)
    
#     return res


def generate_random_slug(length=8):
    """
    Generate a random slug consisting of lowercase letters and digits.
    
    Args:
        length (int): Length of the slug (default is 8).
    
    Returns:
        str: Random slug string.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer.")
    
    # Characters allowed in slug
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def near_zero(tensor):
    '''
        Check if tensor is near-0 with tolerance
        Also checks if all inf or nan. Consider changing name

        Assumes shape batch_size x num_classes
    '''
    tensor = torch.softmax(tensor, dim=1)
    if torch.allclose(tensor, torch.full(tensor.shape,1.0/tensor.shape[1], device=tensor.device)):
        return True 
    mask = torch.isnan(tensor) | torch.isinf(tensor) # Check if all items are nan or inf
    if mask.all().item():
        return True 
    return False

def is_close(tensor1, tensor2):
    return tensor1.shape == tensor2.shape and torch.allclose(tensor1, tensor2)

def compute_sparsity(tensor):
    tot = tensor.numel()
    if tot == 0:
        return 1.0
    zeros = (tensor == 0).sum().item()
    return zeros/tot

def test_n_m_sparse(tensor, n=2, m=4):
    # NOTE: this is not great is dimension isnt divisible by 4
    tensor = tensor.flatten()
    if len(tensor) % m != 0:
        tensor = tensor[:m*(len(tensor)//m)]
    tensor = tensor.reshape(-1,4)
    tensor = (tensor != 0.0)
    tensor = tensor.sum(dim=1)
    return torch.all(tensor <= n).item()

def check_sparsity(model, params):
    sparse_params = model.module.__sparse_params__
    verbose = params['sparsity'].get('verbose_mode', False)
    is_sparse = True
    # assume n:m sparsity for now
    print_params = copy.deepcopy(sparse_params)
    del print_params['sparsity_nodes'], print_params['weights'], print_params['parametrized_params']
    if verbose:
        print(f'----\n[TEST]: {print_params}\n')
    n = sparse_params.n
    m = sparse_params.m
    if verbose:
        print('Sparsified layers:')
        for name, node_list in sparse_params['sparsity_nodes'].items():
            print(f'{name}, {len(node_list)}')
    for name, param in model.module.named_parameters():
        # print(name)
        if name in sparse_params['parametrized_params']:
            if not test_n_m_sparse(param,n,m):
                is_sparse = False
    return is_sparse

# def dummy_data_maker(params):
#     image_size = params.get('image_size', (32,32))
#     batch_size = params.get('batch_size', 4)

#     training_data = torch.rand([2*batch_size, 3, image_size[0], image_size[1]], dtype=torch.float32)
#     test_data = torch.rand([batch_size, 3, image_size[0], image_size[1]], dtype=torch.float32)

#     train_dataloader = DataLoader(training_data, batch_size=batch_size,
#                                    shuffle=False, pin_memory=True, drop_last=True)
#     test_dataloader = DataLoader(test_data, batch_size=batch_size,
#                                    shuffle=False, pin_memory=True, drop_last=True)
    
#     return (train_dataloader, test_dataloader)

def default_params(device):
    params = {
        'device': device,
        'image_size': (224,224),
        'batch_size': 4,
        'weights': 'default',
        
        'pre_save': False,
        'name': generate_random_slug(10)
    }
    return params

def dummy_testbench(params, model_maker=None, model=None):
    image_size = params.get('image_size', (32,32))
    device = params['device']
    torch.set_default_device(device)

    params.setdefault('name', generate_random_slug(10))
    # torch.cuda.manual_seed_all(77)

    if params.get('input_type', 'image') == 'video':
        # 16 frames of images (MVit wants min 16)
        inps = [torch.rand([params['batch_size'],3,16,image_size[0], image_size[1]]).to(device)]
    elif params['batch_size'] == 4 and image_size == (224,224):
        print('[TEST]: Loading saved sample inputs\n----')
        inps = [torch.load('sample_inp224.pt').to(device)]
    else:
        inps = [torch.rand([params['batch_size'],3,image_size[0], image_size[1]]).to(device)]
    
    if (model_maker is not None) and (model is None):
        model = model_maker(params)
    
    model = model.to(device)


    if params.get('pre_save', False):
        # model.train()
        model.eval()
        onnx_model = torch.onnx.export(model, tuple(inps))
        onnx_model.save(f'saved_models/{params["name"]}-pre.onnx', include_initializers=False, keep_initializers_as_inputs=True)
        print(f'netron cmd:\nnetron -p 8080 saved_models/{params["name"]}-pre.onnx')

    model.eval()
    # orig_model = copy.deepcopy(model) # for debug 
    orig_outputs = model(inps[0])

    if params.get('surgery', None) is not None:
        model = xmodelopt.surgery.v3.SurgeryModule(model, example_inputs=tuple(inps), **params['surgery'])
        model = model.to(device)
    
    if params.get('sparsity', None) is not None:
        model = xmodelopt.sparsity.v3.SparserModule(model, example_inputs=tuple(inps), **params['sparsity'])
        model = model.to(device)
        if 'total_epochs' in params['sparsity']:
            dummy_labels = torch.randint(0,2,(len(inps[0]),),device=inps[0].device)
            dummy_opt = torch.optim.SGD(model.parameters(), lr=0.1)
            dummy_loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
            for _ in range(params['sparsity']['total_epochs']):
                model.train()
                dummy_opt.zero_grad()

                inputs, labels = inps[0], dummy_labels 

                pred = model(inputs)
                loss = dummy_loss_fn(pred, labels)

                loss.backward()
                dummy_opt.zero_grad()
            model.eval()
            new_outputs = model(inps[0])
            if params['sparsity'].get('verbose_mode', False):
                print(f'----\n[TEST]: Dist after training: {torch.dist(dummy_labels, torch.argmax(new_outputs, dim=1).to(dtype=torch.float32))}\n----')
            is_sparse = check_sparsity(model, params)
            print(f'----\n[TEST]: Sparsity check: {"✅" if is_sparse else "❌"}\n----')


    model.eval()
    new_outputs = model(inps[0])

    # print(new_outputs)
    equal = False
    all_zero = False
    verbose_mode = 'surgery' in params and params['surgery'].get('verbose_mode', False)
    if isinstance(new_outputs, torch.Tensor):
        all_zero = near_zero(orig_outputs) or near_zero(new_outputs)
        equal = torch.allclose(orig_outputs, new_outputs) 
        if verbose_mode:
            print(f'inp: {inps[0][0][0][0][0:10]}')
            print(f'orig: {orig_outputs[0][0:10]}\nnew: {new_outputs[0][0:10]}')
    elif isinstance(new_outputs, list):
        if len(new_outputs) == 0 or  len(orig_outputs) == 0:
            print(f'----\n[TEST]: WARNING: Zero length outputs detected\n----')
        if 'boxes' in new_outputs[0]:
            # object detection?
            all_zero = True 
            if len(new_outputs) == len(orig_outputs):
                equal = True
                for (x,y) in zip(orig_outputs, new_outputs):
                    if len(x['boxes']) > 0 or len(y['boxes']) > 0:
                        # print(x, y)
                        all_zero = False
                    if verbose_mode:
                        print(f'Checking boxes: {x["boxes"].shape}, {y["boxes"].shape}')
                    if (not is_close(x['boxes'],y['boxes'])) or (not is_close(x['labels'],y['labels'])) or (not is_close(x['scores'],y['scores'])):
                        equal = False
                    if ('masks' in x) and (not is_close(x['masks'],y['masks'])):
                        # maskrcnn test
                        equal = False
                    if verbose_mode and ('keypoints' in x):
                        print(f'checking keypoints: {x["keypoints"].shape}')
                    if ('keypoints' in x) and (not is_close(x['keypoints'],y['keypoints'])):
                        # keypointrcnn test
                        equal = False
            # if all_zero:
            #     print(f'----\n[TEST]: \n----')
    elif isinstance(new_outputs, OrderedDict):
        if 'out' in new_outputs:
            # segmentation (?)
            equal = is_close(orig_outputs['out'], new_outputs['out'])
        
    print(f'----\n[TEST]: Equal outputs check:  {"✅" if equal else "❌"} {"WARNING: Zero length outputs detected" if all_zero else ""}\n----')
    # model.eval()
    if 'surgery' in params and params['surgery'] is not None:
        dynamo = params.get('dynamo', True)
        if dynamo:
            torch.set_default_device('cpu')
            model = model.to('cpu')
            inps[0] = inps[0].to('cpu')
            onnx_model = torch.onnx.export(model, tuple(inps), verbose=False)
            onnx_model.save(f'saved_models/{params["name"]}.onnx', include_initializers=False, keep_initializers_as_inputs=True)
        else:
            torch.set_default_device('cpu')
            model = model.to('cpu')
            inps[0] = inps[0].to('cpu')
            onnx_model = torch.onnx.export(model, tuple(inps), verbose=False, dynamo=False, f=f'saved_models/{params["name"]}.onnx')

        print(f'netron cmd:\nnetron -p 8080 saved_models/{params["name"]}.onnx')

    del model
    del inps
    
    
    torch.cuda.empty_cache()
    if 'surgery' in params and params['surgery'] is not None:
        del onnx_model
        try:
            onnx_check = simple_onnx_test(f'saved_models/{params["name"]}.onnx')
        except AssertionError as e:
            onnx_check = False
            print(f'----\n[TEST]: AssertionError: {e}\n----')
            traceback.print_exc()
        print(f'----\n[TEST]: Simple ONNX graph check: {"✅" if onnx_check else "❌"}\n----')
    


def test_tvmodel_all_options(device, base_params, **kwargs):
    model_name = base_params.get('model_name', 'resnet')
    config = TVMODEL_CONFIGS[model_name]

    for i, option in enumerate(config['options']):
        if 'limit' in kwargs and i >= kwargs['limit']:
            break
        params = copy.deepcopy(base_params)

        fetch_weights = kwargs.get('fetch_weights', 
                                   (i==0) or (option == 'mobilenet_v3_large_fpn'))
        print(f'----\n[TEST]: Testing {model_name} {option}\n----')
        if config['type'] == 'classification':
            dummy_testbench(params, model=construct_tv_model(model_name, option=option, verbose=True, fetch_weights=fetch_weights, **kwargs))
        elif config['type'] == 'detection':
            params['surgery']['transformation_dict'] = {
                'backbone': None,
                'head': None
            }
            params['dynamo'] = False
            dummy_testbench(params, model=construct_tv_model(model_name, option=option, verbose=True, fetch_weights=fetch_weights,  **kwargs))
        elif config['type'] == 'segmentation':
            dummy_testbench(params, model=construct_tv_model(model_name, option=option, verbose=True, fetch_weights=fetch_weights,  **kwargs))
        elif config['type'] == 'video':
            params['input_type'] = 'video'
            dummy_testbench(params, model=construct_tv_model(model_name, option=option, verbose=True, fetch_weights=fetch_weights,  **kwargs))
        else:
            raise NotImplementedError(f'{config["type"]}')


def test_surgery_tvmodel_all_options(device, model_name='resnet', **kwargs):
    params = default_params(device)
    params.update({
        'model_name': model_name,
        # 'pre_save': True,
    })
    params['surgery'] = {
        'verbose_mode': False
    }
    params['surgery']['replacement_dict'] = {
        'squeeze_and_excite_to_identity' : True, 
        # 'all_activation_to_relu': False,
        'relu_inplace_to_relu' : True,
        'gelu_to_relu' : True, 
        'relu6_to_relu' : True,
        'silu_to_relu' : True, 
        'hardswish_to_relu' : True,
        'hardsigmoid_to_relu' : True,
        'leakyrelu_to_relu' : True,
        'dropout_inplace_to_dropout':True,
        'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
        'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
        'convert_resize_params_size_to_scale':True,
        'promote_conv2d_with_even_kernel_to_larger_odd_kernel':True,
        'break_conv2d_with_kernel_size_greater_than_7':True,
    }
    test_tvmodel_all_options(device, params, **kwargs)

def test_sparsity_tvmodel_all_options(device, model_name='resnet', **kwargs):
    params = default_params(device)
    params.update({
        'model_name': model_name,
        # 'pre_save': True,
    })
    params['sparsity'] = {
        'verbose_mode': True,
        'sparsity_ratio': 0.5,
        'sparsity_m': 4,
        'total_epochs': 10, 
    }

    test_tvmodel_all_options(device, params, **kwargs)


tvmodel_names = TVMODEL_NAMES
test_tvmodel_names = ['custom_swin', 'custom_resnet']

current = TVMODEL_NAMES[5:10]

if __name__ == "__main__":
    idx = 0
    # torch.accelerator.set_device_index(idx)
    # device = torch.device(f'cuda:{idx}')
    device = torch.device('cpu')

    for name in current:
        # test_surgery_tvmodel_all_options(device, name)
        test_sparsity_tvmodel_all_options(device, name)

    # test_sparsity_tvmodel_all_options(device, 'resnet')