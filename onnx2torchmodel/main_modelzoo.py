# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import onnx.shape_inference
import onnxruntime
import numpy as np
import torch
import onnxsim
import onnx
import onnx_graphsurgeon as gs
import os
import yaml

from edgeai_onnx2torchmodel.onnx2pytorch import convert
model_zoo_path = '/data/ssd/files/a0507161/edgeai/edgeai-modelzoo/models/'
exp_path = './workdir/onnx2onnx_test/modelzoo'

def add_all_output(model_path, output_path=None):
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
    output_path = output_path or os.path.dirname(model_path)
    graph = gs.import_onnx(model)
    #%%
    for node in graph.nodes:
        if node.op in ('Constant','ConstantOfShape '):
            continue
        for out in node.outputs:
            if out in graph.outputs:
                continue
            graph.outputs.append(out)
    #%%
    try:
        os.makedirs(output_path, exist_ok=True)
        model = gs.export_onnx(graph)
        output_path = os.path.join(output_path, os.path.basename(model_path).replace('.onnx', '_all.onnx'))
        onnx.save_model(model, output_path)
    except Exception as e:
        print(f"Failed to add all outputs to model's output because of error {e}")
        output_path = model_path
    return output_path
x = None


def main(args=None, inps=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Path to ONNX model')
    parser.add_argument('model_path', type=str, help='Path to ONNX model')
    parser.add_argument('--all_output' ,'-a', action='store_true', help='to export model with outputs of all nodes')
    parser.add_argument('--export_txts' ,'-e', action='store_true', help='to export error txt')
    parser.add_argument('--for-training','-t', action='store_true', help='to use training mode')
    parser.add_argument('--threshold1' ,'-t1', type=float, default=1e-5, help='to export error txt')
    parser.add_argument('--threshold2' ,'-t2', type=float, default=1e-2, help='to export error txt')
    parser.add_argument('--cuda','-c', action='store_true', help='to use cuda')
    parser.add_argument('--simplify','-s', action='store_true', help='to simplify model')
    
    args = parser.parse_args() if args is None else parser.parse_args(args)
    directory = os.path.join(exp_path, args.model_name.replace('-','_'))
    model_path = args.model_path
    os.makedirs(directory, exist_ok=True)
    if args.all_output:
        model_path = add_all_output(model_path, directory)
        # onnx.shape_inference.infer_shapes_path(model_path, model_path)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    torch_model = convert(model_path, args.for_training)
    return torch_model,None
    session1 = onnxruntime.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    torch_model.eval()
    if args.export_txts:
        with open(os.path.join(directory,'graph.txt'), 'w') as f:
            f.write(str(torch_model.graph))
        with open(os.path.join(directory,'code.txt'), 'w') as f:
            f.write(str(torch_model.code))
    if args.cuda:
        torch_model = torch_model.cuda()
    inputs = []
    input_dict = {}
    dtype_mapping = {'tensor(float)' : np.float32, 
                 'tensor(int64)' : np.int64,
                 'tensor(uint8)' : np.uint8,
                 'tensor(int32)' : np.int32}
    if inps:
        inps = [inp.numpy() if isinstance(inp, torch.Tensor) else inp for inp in inps]
    for i, inp in enumerate(session1.get_inputs()):
        input_dict[inp.name] = inps[i] if inps else np.ones(inp.shape, dtype=dtype_mapping[inp.type])
        inputs.append(  torch.from_numpy(input_dict[inp.name]))
        if args.cuda:
            inputs[-1] = inputs[-1].cuda()
    output_names = [o.name for o in session1.get_outputs()]
    output1 = session1.run([], input_dict)
    output2 = torch_model(*inputs)
    if len(output1)==1:
        output2 = [output2]
    output2 = [o.detach() if hasattr(o,'detach') else torch.tensor(o) for o in output2]
    if args.cuda:
        output2 = [o.cpu() for o in output2]
    output1 = [torch.from_numpy(o) for o in output1]
    output1 = [o.float() for o in output1]
    output2 = [o.float() for o in output2]
    # output1 = [torch.round(o.float(), decimals=5) for o in output1]
    # output2 = [torch.round(o.float(), decimals=5) for o in output2]
    if args.cuda:
        torch_model = torch_model.cpu()
        inputs = [inp.cpu() for inp in inputs]
    new_path = os.path.join(directory, os.path.basename(model_path))
    torch.onnx.export(torch_model, tuple(inputs),new_path)
    model = onnx.load(new_path)
    if args.simplify:
        model , _= onnxsim.simplify(model)
    else:
        model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model,new_path)
    session3 = onnxruntime.InferenceSession(new_path, sess_options, providers=['CPUExecutionProvider'])
    input_dict1 = {inp.name : inputs[i].numpy() for i,inp in enumerate(session3.get_inputs())}
    output3 = session3.run([], input_dict1)
    output3 = [torch.from_numpy(o) for o in output3]
    output3 = [o.float() for o in output3]
    output_names1 = [o.name for o in session3.get_outputs()]
    # output3 = [torch.round(o.float(), decimals=5) for o in output3]
    def export_error(output1, output2, name):
        if args.export_txts:
            bv = [o1.shape == o2.shape and torch.all((o1-o2).abs() < (args.threshold1)) for o1, o2 in zip(output1, output2) ]
            with open(os.path.join(directory,f'{name}.txt'),'w') as f:
                s = 'status, (max_abs_error, max_abs_rel_error(non-zero), max_error(zero)),output1_name, output2_name\n'
                for i, b in enumerate(bv):
                    o1 = output1[i]
                    o2 = output2[i]
                    max_error = None
                    diff = (torch.abs(o1-o2)) if isinstance(b, torch.Tensor) and o2.numel() else None
                    if diff is not None :
                        threshold= torch.max(diff) /100
                        condition = torch.logical_or(diff<threshold, o1 == 0)
                        if torch.any(condition):
                            max_error = diff[condition].max()
                        o1 = o1[torch.logical_not(condition)].abs() 
                        diff = diff[torch.logical_not(condition)].abs() 
                    error = (diff.max() if isinstance(diff, torch.Tensor) and diff.numel() else None)
                    error = error, ((diff/(o1)).max() if error or isinstance(error,(torch.Tensor)) else None)
                    b = bv[i] = b or (error[1]  < args.threshold2) if error[1] else b
                    s += (('' if b else 'not ')+f'matched, ({(error[0])}, {error[1]}, {max_error} ) , '+output_names[i]+', '+output_names1[i])
                    s += '\n'
                f.write(s)
    
    export_error(output1, output2, 'error1')
    export_error(output1, output3, 'error2')
    export_error(output2, output3, 'error3')

    return torch_model, output1, output2, output3

if __name__ == '__main__':
    model_names = []
    config_path = os.path.join(model_zoo_path, 'configs.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)['configs']
    status = {}
    failed = ['od-8920']
    failed = [
        # 'cl-6508', # QDQ
        # 'cl-6507', QDQ
        # 'od-8950', # use actual input
        # 'od-8080', # use actual input
    ]
    total_count = 0
    model_names = sorted(config.keys())
    # model_names = failed
    for model_name in model_names:
    # for model_name, path in config.items():
        path = config[model_name]
        path = os.path.join(model_zoo_path, path)
        if not os.path.exists(path):
            print(f"Config {path} of Model {model_name} does not exist")
            status[model_name] = "yaml not found"
            continue
        model_config = yaml.load(open(path, 'r'), Loader=yaml.Loader)
        model_path = model_config['session']['model_path']
        model_path = os.path.join(os.path.dirname(path), model_path)
        if not os.path.exists(model_path):
            print(f"Model {model_path} does not exist")
            status[model_name] = 'onnx not found'
            continue
        if not model_path.endswith('.onnx'):
            continue
        # else:
        try:
            print("#######################################################")
            print(model_name)
            total_count += 1
            torch_model = main([model_name, model_path, '-e','-s', '-t',],)[0]
            example_inputs = []
            for inp, info in torch_model.input_info.items():
                if info['dtype'] not in [ torch.int8, torch.uint8, torch.int16, torch.uint16, torch.int32, torch.uint32, torch.int64, torch.uint64,]:
                    example_inputs.append(torch.rand(info['shape'], dtype=info['dtype']))
                else:
                    example_inputs.append(torch.randint(low=0, high=255, size=info['shape'], dtype=info['dtype']))
            torch_model.eval()
            outputs1 = torch_model(*example_inputs)
            # pt2e_model = torch.export.export(torch_model, tuple(example_inputs)).module()
            # pt2e_model.eval()
            # outputs2 = pt2e_model(*example_inputs)
            print(f"Successfully converted model {model_name}")
            status[model_name] = 'passed'
        except Exception as e:
            print(f"Failed to convert model {model_name} because of error {e}")
            status[model_name] = 'failed'
            continue
        # Step 1. export
        # model_name = model_name[:-5]
        
        # directory = f'./workdir/onnx2onnx_test/{model_name}'
            
        # ep = torch.export.export(torch_model, tuple(example_inputs))
        # pt2e_model = ep.module()
        # # torch.export.save()
        # with open(os.path.join(directory,'pt2e_model_graph.txt'), 'w') as f:
        #     f.write(str(pt2e_model.graph))
        # with open(os.path.join(directory,'pt2e_model_code.txt'), 'w') as f:
        #     f.write(pt2e_model.code)
        # # break
        # # Step 2. quantization
        # from torchao.quantization.pt2e.quantize_pt2e import (prepare_qat_pt2e, convert_pt2e,)
        # # install executorch: `pip install executorch`
        # from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (get_symmetric_quantization_config, XNNPACKQuantizer,)


        # # backend developer will write their own Quantizer and expose methods to allow
        # # users to express how they
        # # want the model to be quantized
        # quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        # student_model = prepare_qat_pt2e(pt2e_model, quantizer)
        
        
        # with open(os.path.join(directory,'student_model_graph.txt'), 'w') as f:
        #     f.write(str(student_model.graph))
        # with open(os.path.join(directory,'student_model_code.txt'), 'w') as f:
        #     f.write(student_model.code)


        # student_model1 = convert_pt2e(student_model)
        # with open(os.path.join(directory,'student_model1_graph.txt'), 'w') as f:
        #     f.write(str(student_model1.graph))
        # with open(os.path.join(directory,'student_model1_code.txt'), 'w') as f:
        #     f.write(student_model1.code)
        # break
    # output1 = new_model(inp)
    pass
    # main(['test.onnx', '-e'])
    count = 0
    failed = []
    for k, v in status.items():
        # print(f'\t\t{k},# {"passed" if v else "failed"}')
        if v != 'failed': 
            continue
        failed.append(k)
        count += 1
    print('failed',count, 'out of', total_count)
    print(failed)