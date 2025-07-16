import os
import sys
import importlib

import pandas as pd


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





model_utils_module = import_file_or_folder('./references/classification/model_utils.py')
MODEL_SURGERY_NAMES_LITE2ORIGINAL:dict = model_utils_module.MODEL_SURGERY_NAMES_LITE2ORIGINAL


def get_args_parser(add_help = True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument('--model',default="resnet18", type=str, help="model name")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument('--gpus', default='1', type=str, help='number of gpus')
    parser.add_argument("-b", "--batch-size", default='128', type=str, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument(
        "--val-resize-size", default='256', type=str, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default='224', type=str, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument('--original-accuracy','--acc',dest='orig_acc', default=78, type=float, help='accuracy of original model')
    parser.add_argument('--result-path',type=str,default=None,help='path to csv file that will store all result (default: result.csv in parent directory of output directory)')
    parser.add_argument("--opset-version", default='17', type=str, help="ONNX Opset version")
    return parser


def main(args):
    train_module = import_file_or_folder('./references/classification/train.py')
    if args.result_path is None:
        args.result_path = str(args.output_dir).rsplit('/',1)[0]+'/result.csv'
        
    train_args = ['--data-path',args.data_path,'--gpus',args.gpus,'--parallel','1','--batch-size',args.batch_size,'--model',args.model,'--opset-version',args.opset_version,'--val-resize-size',args.val_resize_size,'--val-crop-size',args.val_crop_size, '--output-dir',args.output_dir,"--print-freq",'200','--weights',args.weights,'--test-only','--mixup-alpha','0.2','--cutmix-alpha','1.0']
    
    train_args.extend(['--model-surgery','2'])
    
    print(args)
    print()
    train_args = train_module.get_args_parser().parse_args(train_args)
    # return str(train_args)
    acc:int = train_module.main(train_args)
    
    orig_model:str = MODEL_SURGERY_NAMES_LITE2ORIGINAL[args.model][0] if args.model in MODEL_SURGERY_NAMES_LITE2ORIGINAL else args.model
    difference = acc - args.orig_acc
    remark = 'success' if difference > -0.1 else 'failed'
        
    if not os.path.exists(args.result_path):
        columns = ['original_model', 'actual_accuracy', 'lite_model', 'new_accuracy', 'accuracy_difference','remark']
        data= {col:list() for col in columns}
        df = pd.DataFrame(columns=columns)
    else:  
        df = pd.read_csv(args.result_path)
    df.loc[ len(df.index)] = [orig_model,args.orig_acc,args.model,acc,difference,remark]
    df.to_csv(args.result_path,index=False)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

    