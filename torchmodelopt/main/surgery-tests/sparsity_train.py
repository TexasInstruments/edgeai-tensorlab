import argparse
import copy
import os 
import logging 
from functools import partial 

import torch 
import optuna 
import mlflow
from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, nvmlInit, nvmlShutdown

from edgeai_torchmodelopt import xmodelopt
from train_test import train_epochs, my_train_scheme_maker, imagenet_data_maker, count_params
from tv_models import construct_tv_model

from edgeai_torchmodelopt.xmodelopt.sparsity.v3.utils import register_n2m_filter
from edgeai_torchmodelopt.xmodelopt.utils.helper_functions import nested_getattr
from torch import fx

# from  edgeai_torchmodelopt.xmodelopt.sparsity.v3.sparsity_func import calculate_sparsity

def compute_sparsity(tensor):
    tot = tensor.numel()
    if tot == 0:
        return 1.0
    zeros = (tensor == 0).sum().item()
    return zeros/tot

def test_n_m_sparse(tensor, n=2, m=4):
    # TODO: determine behaviour if num. el. isnt divisible by 4
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
    del print_params['sparsity_nodes'], print_params['weights'], print_params['parametrization_list']
    if verbose:
        print(f'----\n[TEST]:  __sparse_params__ (partial) = {print_params}\n')
    n = 2
    m = 4
    if verbose:
        print('Sparsified layers:')
        for name, node_list in sparse_params['sparsity_nodes'].items():
            print(f'{name}, {len(node_list)}')
    for param_name in sparse_params['parametrized_params']:
        param = nested_getattr(model.module, param_name)
        if verbose:
            print(f'Sparsity of param {param_name}: {compute_sparsity(param)}')
        if not test_n_m_sparse(param,n,m):
            if verbose:
                print(f'param {param_name} is not sparse!')
            is_sparse = False
    if verbose:
        print(f'[TEST]:  Testing {n}:{m} sparsity..\n')
    return is_sparse

def sample_inputs(params):
    batch_size = params.get('batch_size', 128)
    image_size = params.get('image_size', (224,224))
    return (torch.rand((batch_size,3,image_size[0],image_size[1]), device=params['device']),)

def create_custom_filters(n,m,params={}):
    sparse_list = params['sparsity'].get('to_sparsify', [])
    
    @register_n2m_filter('Conv2d', n, m)
    def convs_filter_func(module):
        """Filters Conv2d operations that are compatible with n:m sparsity.
        
        This function examines all Conv2d operations in the module and determines
        which ones are compatible with n:m sparsity based on their dimensions.
        For Conv2d, both input and output channels must be divisible by m.
        
        Args:
            module: A GraphModule to filter nodes from.
            
        Returns:
            list: List of node lists that can be sparsified.
        """
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.conv2d.default:
                continue
            weight = node.args[1]
            if weight.op != 'get_attr':
                # Skipping as weight tensor is variable to conv
                continue
            weight = params.get(weight.target)
            out_channel, in_channel, *kernel_size = weight.shape
            if out_channel % m != 0 or in_channel % m != 0:
                # skipping as conv is not supported for n:m sparsity
                continue
            # print(f'checking conv kernel {kernel_size}')
            allow = False 
            if 'conv1x1' in sparse_list and all(k==1 for k in kernel_size ):
                allow = True
            if 'conv3x3' in sparse_list and all(k==3 for k in kernel_size ):
                allow = True
            if allow:
                ret.append([node])
        return ret
    
    @register_n2m_filter('Linear', n, m)
    def linears_filter_func(module):
        """Filters Linear operations that are compatible with n:m sparsity.
        
        This function examines all Linear operations in the module and determines
        which ones are compatible with n:m sparsity based on their dimensions.
        For Linear layers, both input and output features must be divisible by m.
        
        Args:
            module: A GraphModule to filter nodes from.
            
        Returns:
            list: List of node lists that can be sparsified.
        """
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.linear.default:
                continue
            weight = node.args[1]
            if weight.op != 'get_attr':
                # Skipping as weight tensor is variable to conv
                continue
            weight = params.get(weight.target)
            out_channel, in_channel= weight.shape
            if out_channel % m != 0 or in_channel % m != 0 :
                # skipping as linear is not supported for n:m sparsity
                continue
            allow = False 
            if 'linear' in sparse_list:
                allow = True
            if node.args[1].target == 'head.weight': # exclude swin-t head
                allow = False
            if allow:
                # print(node.name)
                # print(node.args[1].target)
                ret.append([node])
        
        return ret
    
class SparserModuleNM(xmodelopt.sparsity.v3.SparserModule):
    def __init__(self, module, params, transformation_dict=None, **kwargs):
        # total_epochs = params.get('total_epochs', 10)
        # sparsity_epochs = params['sparsity'].get('total_epochs', total_epochs)
        # batch_size = params.get('batch_size', 128)
        
        inps = sample_inputs(params)

        filter_func = None
        if 'to_sparsify' in params['sparsity']:
            filter_func = partial(create_custom_filters, params=params)
        
        super().__init__(module, example_inputs=inps, filter_func_register=filter_func, **params['sparsity']['module_args'])
        # print(self.module.__sparse_params__)
    

def construct_resnet(params):
    return construct_tv_model('resnet', option='50', fetch_weights=True)

def construct_mobilenet(params):
    return construct_tv_model('mobilenet', option='v3_large', fetch_weights=True)

def construct_sparse_resnet(params):
    depth = params.get('resnet_depth', 18)
    
    model = construct_tv_model('resnet', option=depth, fetch_weights=True)
    model = model.to(params['device'])
  
    if params.get('distillation', False):
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False
    if 'sparsity' in params and params['sparsity']:
        model = SparserModuleNM(model, params)
    if params.get('distillation', False):
        return model, teacher
    return model 

def construct_sparse_mobilenet(params):
    fetch_weights = params.get('fetch_weights', True)
    model = construct_tv_model('mobilenet', option='v3_large', fetch_weights=fetch_weights)
    model = model.to(params['device'])
    if 'sparsity' in params and params['sparsity']:
        model = SparserModuleNM(model, params)

    return model 

def construct_sparse_swin(params):
    model = construct_tv_model('swin', option='t', fetch_weights=True)
    model = model.to(params['device'])

    if params.get('distillation', False):
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False
    if 'sparsity' in params and params['sparsity']:
        model = SparserModuleNM(model, params)
    if params.get('distillation', False):
        return model, teacher

    return model 


def generic_main(params, model_maker, data_maker, train_scheme_maker):
    train_dataloader, test_dataloader = data_maker(params)
    params['steps_per_epoch'] = len(train_dataloader)
    model = model_maker(params)
    teacher = None
    if params.get('distillation', False):
        model, teacher = model
        teacher = teacher.to(params['device'])
    model = model.to(params['device'])
    print(params['model_name'], count_params(model))
    optimizer, loss_fn, scheduler = train_scheme_maker(params, model)

    device = params.get('device', 'cuda:0')
    total_epochs = params.get('total_epochs', 2)

    def debug_callback(epoch, model, train_acc, test_acc, params):
        model.step()
        if params['trial']:
            print(f"Trial {params['trial'].number}: Epoch {epoch+1}/{total_epochs}")
        else:
            print(f"Epoch {epoch+1}/{total_epochs}..")
        print(f'Train Acc.: {train_acc*100.0:.2f}%, Test Acc.: {test_acc*100.0:.2f}%')
    

    test_acc = train_epochs(train_dataloader, test_dataloader, model, loss_fn, optimizer, 
                            scheduler, device, params=params, total_epochs=total_epochs, 
                            log_mlflow=True, epoch_callback=debug_callback, teacher=teacher)
    model.finalize()
    
    # params['sparsity'] = {'verbose_mode': True}
    
    if 'sparsity' in params and params['sparsity']:
        is_sparse = check_sparsity(model, params)
        print(f'----\n[TEST]: Sparsity check: {"✅" if is_sparse else "❌"}\n----')

    # inps = sample_inputs(params)
    # onnx_model = torch.onnx.export(model, (inps), verbose=False)
    # onnx_model.save(f'saved_models/{params["model_name"]}-{params["name"]}.onnx', include_initializers=True, keep_initializers_as_inputs=False)
    # print(f'netron cmd:\nnetron -p 8080 saved_models/{params["model_name"]}-{params["name"]}.onnx')
    torch.cuda.empty_cache()

    return test_acc


def get_free_gpu_index(idx=0):
    """
        Utility to automate GPU choice. Picks a gpu with min utilization
        Start with idx and round robin
    """
    nvmlInit()
    num_gpus = nvmlDeviceGetCount()
    min_util = 101
    min_idx = idx
    for i in range(num_gpus):
        gpu_idx = (i+idx)%num_gpus
        handle = nvmlDeviceGetHandleByIndex(gpu_idx)
        util = nvmlDeviceGetUtilizationRates(handle)
        if util.gpu < min_util:
            min_util = util.gpu
            min_idx = gpu_idx
    nvmlShutdown()
    return min_idx

def accuracy_objective(trial, idx=0, study_context=''):
    """
        Set params and call generic_main
    """

    device_idx = get_free_gpu_index(idx)
    if device_idx == -1:
        raise ValueError("No free gpu")
    torch.accelerator.set_device_index(device_idx)
    device = torch.device(f'cuda:{device_idx}')
    
    params = {
        'device': device,
    }

    print(f'Running trial {trial.number=} in PID {os.getpid()}')

    trial.set_user_attr('name', f"{study_context.get('name', '')}-{trial.number}")
    params['name'] =  f"{study_context.get('name', '')}-{trial.number}"

    params['batch_size'] = 128
    # params['batch_size'] = 64
    params['image_size'] = (224,224)
    params['fraction'] = 1
    # params['fraction'] = 0.1

    params['optimizer_type'] = 'SGD'
    params['lr'] = 1e-5
    # params['sr-ste'] = 2e-4
    params['momentum'] =  0.9

    if 'sr-ste' in params:
        xmodelopt.sparsity.v3.parametrization.STE_GAMMA = params['sr-ste']
    else:
        xmodelopt.sparsity.v3.parametrization.STE_GAMMA = 0

    # params['label_smoothing'] = 0.1
    # params.update({
    #     'cutmix_alpha': 1.0,
    #     'mixup_alpha': 1.0,
    #     'randomresizedcrop': True,
    #     # 'randaugment': (1,3),
    #     'randerasing': 0.25,
    #     'autoaugment': "imagenet",
    # })

    # params['lr_schedule'] = {
    #     'type': '1cycle'
    # }
    # params['lr_schedule'] = {
    #     'type': 'step',
    #     'warmup': 5,
    #     'epochs': 15,
    #     'rate': 0.3
    # }
    # params['lr_schedule'] = {
    #     'type': 'cosine',
    #     'warmup': 15,
    # }

    params['loss_fn'] = 'ce'

    params['total_epochs'] = 60
    # params['total_epochs'] = 5
    params['trial'] = trial
    params['trial_no'] = trial.number

    params['model_name'] = 'resnet18' 
    # params['model_name'] = 'resnet50'
    # params['model_name'] = 'mobilenet_v3' 
    # params['model_name'] = 'swin_t'
    # params['model_name'] = 'mobilenet_v3_scratch' 

    n = 2
    m = 4
    params['sparsity'] = {
        'verbose_mode': True,
        'module_args': {
            'sparsity_ratio': n/m,
            'sparsity_m': m,
            'm': m,
            'n': n,
            'sparsity_start_epoch': 0,
            'sparsity_end_epoch': 30,
            'freeze_mask': True,
            'mode': 'topk_blockwise',
            # 'mode': 'topk',
        },
        
        # 
        # 'to_sparsify': ['linear']
        'to_sparsify': ['linear', 'conv1x1', 'conv3x3']
    }
    # params['distillation'] = {
    #     'temp': 1.0,
    #     'teacher_weight': 2.0,
    #     'use_ground_labels': True,
    # }

    print(f'{params=}')
    if params['model_name'] == 'resnet18':
        params['resnet_depth'] = 18
        model_maker = construct_sparse_resnet
    elif params['model_name'] == 'mobilenet_v3':
        model_maker = construct_sparse_mobilenet
    elif params['model_name'] == 'mobilenet_v3_scratch':
        params['fetch_weights'] = False
        model_maker = construct_sparse_mobilenet
    elif params['model_name'] == 'resnet50':
        params['resnet_depth'] = 50
        model_maker = construct_sparse_resnet
    elif params['model_name'] == 'swin_t':
        model_maker = construct_sparse_swin
    else:
        model_maker = construct_sparse_resnet
    test_acc = generic_main(params, model_maker, imagenet_data_maker, my_train_scheme_maker)
    # trial.set_user_attr('params', json.dumps(params)) # For this to work, need to remove device, trial etc.
    return test_acc


def optuna_runner(idx, study_context=None):
    study = optuna.create_study(study_name=study_context['name'], direction='maximize', 
                                    storage=study_context['storage'], load_if_exists=True,
                                    pruner=optuna.pruners.NopPruner())
    study.optimize(partial(accuracy_objective, idx=idx, 
                               study_context=study_context), n_trials=study_context.get('n_trials_per',1))

def single_test():
    torch.cuda.empty_cache()
    study_name = 'sparsity-imagenet-0.1'
    study_context = {
        'name': study_name,
        'storage': f'sqlite:///{study_name}.db',
        'n_trials_per': 1
    } 
    logger = logging.getLogger("mlflow")
    if logger:
        logger.setLevel('ERROR')
    
    mlflow.set_experiment('sparsity-imagenet')
    # mlflow.set_experiment('test')
    study = optuna.create_study(study_name=study_context['name'], direction='maximize', 
                                            storage=study_context['storage'], load_if_exists=True)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--idx",
        type=int,
        default=0,
        help="preferred GPU index"
    )
    args = parser.parse_args()
    optuna_runner(args.idx, study_context)
    print(study.trials[-1].value)


if __name__ == "__main__":
    single_test()