import os 
import json
import datetime
from torch import nn

from demo_utils import  summary
from demo_utils import import_file_or_folder
from edgeai_torchmodelopt.xmodelopt.pruning.v2.utils import create_channel_pruned_model
from edgeai_torchmodelopt.xmodelopt.pruning.v2.pruner_module import PrunerModule

# from edgeai_xvision.references.classification import train as train_module

n_epochs = 80
sp_epochs = 1*1


def _get_model_name(model_name:str):
    return model_name.lower().replace(' ','_')


def backend_task(model_type="Mobilenet V2",n_epochs=n_epochs,surgery_dict=None,sp_ratio=0.0,sp_type='channel',qntzn=False,):
    # please have edgeai-torchvision in same directory as edgeai-modeloptimization
    train_module = import_file_or_folder('../../../edgeai-torchvision/references/classification/train.py')
    
    if surgery_dict is None:
        surgery_dict = {}
    
    model_name = _get_model_name(model_type)
    weights = f"../output/{model_name}/checkpoint.pth"
    dt = datetime.datetime.now()
    dt = f'{dt:%Y_%m_%d_%H_%M_%S}'
    output_dir = f'../output/{model_name}/{dt}'
    
    with open('../surgery_dict.json','w') as json_file:
        json.dump(surgery_dict,json_file,indent=4)
    
    args=[]
    surgery_args = None       
    if len(surgery_dict):
        surgery_args = ['--model-surgery','2','--surgery-json','../surgery_dict.json',]
        args.extend(surgery_args)
    
    sp_args = None
    if sp_ratio >0 :
        n_epochs += sp_epochs
        sp_args = ['--pruning','2','--pruning-ratio',str(sp_ratio),'--pruning-type',sp_type,'--pruning-init-train-ep',str(sp_epochs)]
        args.extend(sp_args)   
    
    # please have imagenette dataset (subset of imagenet with 10 classes) in same directory as edgeai-modeloptimization
    args.extend(['--data-path','../../../imagenette','--epochs',str(n_epochs),'--gpus','4','--parallel','1','--batch-size','128','--wd','4e-5','--lr','0.1','--lr-scheduler','cosineannealinglr','--lr-warmup-epochs','1','--model',model_name,'--opset-version','18','--val-resize-size','232','--val-crop-size','224', '--output-dir',output_dir])
    
    if  (os.path.exists(weights)):
        args.extend(['--weights',weights,])
        # n_epochs =0 if len(surgery_dict) == 0  else n_epochs
    # else:
    #     sp_epochs = 20
        
    args = train_module.get_args_parser().parse_args(args)
    model, best_acc,last_acc= train_module.main(args)
    
    print("best accuracy:",best_acc)

    if ((sp_ratio > 0) and (sp_type=='channel')):
        model.module.module = create_channel_pruned_model(model.module.module) 
    model_without_ddp = model.module if isinstance(model,nn.DataParallel) else model
    model_summary = summary(model_without_ddp.module if isinstance(model_without_ddp,PrunerModule) else model_without_ddp,input_size=(1,3,224,224))
    with open(f'{output_dir}/parameters.json','w') as parameter_file:
        json.dump(
            {
                'surgery dictionary': str(surgery_dict),
                'sparsity ratio': sp_ratio,
                'sparsity type' : sp_type,
                'quantization':qntzn,
                'accuracy':best_acc,
                'Flops':model_summary.total_mult_adds,
                'Params':model_summary.total_params
            },parameter_file,indent=4
        )
    print(type(model))
    return best_acc,model_summary.total_mult_adds, model_summary.total_params #* (1 - ())

if __name__ == '__main__':
    # backend_task(surgery_dict=None,sp_ratio=0.15,sp_type='unstructured')
    backend_task(model_type='resnet50',surgery_dict=None,sp_ratio=0.0,sp_type='unstructured')
    # summary(torchvision.models.resnet50(),(1,3,224,224))
# backend_task(n_epochs=2,surgery_dict=None,sp_ratio=0.15)
