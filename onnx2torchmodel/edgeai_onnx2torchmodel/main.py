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
from torchvision import models
from onnx2pytorch import convert
import edgeai_torchmodelopt
import onnxsim
import onnx
import onnx_graphsurgeon as gs
import os

def add_all_output(model_path):
    model = onnx.load(model_path)
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
        model = gs.export_onnx(graph)
        output_path = model_path.replace('.onnx', '/all.onnx')
        onnx.save_model(model, output_path)
    except Exception as e:
        print(f"Failed to add all outputs to model's output because of error {e}")
        output_path = model_path
    return output_path
x = None


def main(args=None, inps=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to ONNX model')
    parser.add_argument('--all_output' ,'-a', action='store_true', help='to export model with outputs of all nodes')
    parser.add_argument('--export_txts' ,'-e', action='store_true', help='to export error txt')
    parser.add_argument('--threshold1' ,'-t1', type=float, default=1e-5, help='to export error txt')
    parser.add_argument('--threshold2' ,'-t2', type=float, default=1e-2, help='to export error txt')
    parser.add_argument('--cuda','-c', action='store_true', help='to use cuda')
    parser.add_argument('--simplify','-s', action='store_true', help='to simplify model')
    
    args = parser.parse_args() if args is None else parser.parse_args(args)
    directory = os.path.splitext(args.model_path)[0]
    os.makedirs(directory, exist_ok=True)
    if args.all_output:
        args.model_path = add_all_output(args.model_path)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    torch_model = convert(args.model_path)
    session1 = onnxruntime.InferenceSession(args.model_path, sess_options, providers=['CPUExecutionProvider'])
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
    output2 = [o.detach() for o in output2]
    if args.cuda:
        output2 = [o.cpu() for o in output2]
    output1 = [torch.from_numpy(o) for o in output1]
    # output1 = [torch.round(o.float(), decimals=5) for o in output1]
    # output2 = [torch.round(o.float(), decimals=5) for o in output2]
    bv = [o1.shape == o2.shape and torch.all((o1-o2).abs() < (args.threshold1)) for o1, o2 in zip(output1, output2) ]
    if args.cuda:
        torch_model = torch_model.cpu()
        inputs = [inp.cpu() for inp in inputs]
    torch.onnx.export(torch_model, tuple(inputs),args.model_path.replace('.onnx','1.onnx'))
    model = onnx.load(args.model_path.replace('.onnx','1.onnx'))
    if args.simplify:
        model , _= onnxsim.simplify(model)
    else:
        model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model,args.model_path.replace('.onnx','1.onnx'))
    session3 = onnxruntime.InferenceSession(args.model_path.replace('.onnx','1.onnx'), sess_options, providers=['CPUExecutionProvider'])
    input_dict1 = {inp.name : inputs[i].numpy() for i,inp in enumerate(session3.get_inputs())}
    output3 = session3.run([], input_dict1)
    output3 = [torch.from_numpy(o) for o in output3]
    output_names1 = [o.name for o in session3.get_outputs()]
    # output3 = [torch.round(o.float(), decimals=5) for o in output3]
    if args.export_txts:
        with open(os.path.join(directory,'error1.txt'),'w') as f:
            s = 'status, (max_abs_error, max_abs_rel_error(non-zero), max_error(zero)),output1_name, output2_name\n'
            for i, b in enumerate(bv):
                o1 = output1[i]
                o2 = output2[i]
                max_error = None
                diff = (torch.abs(o1-o2)) if isinstance(b, torch.Tensor) and o2.numel() else None
                if torch.any(o1 == 0):
                    max_error = diff[torch.where(o1==0)].max()
                diff = diff[torch.where(o1!=0)] if diff is not None else None
                o1 = o1[torch.where(o1!=0)].abs()
                error = (diff.max() if isinstance(diff, torch.Tensor) and diff.numel() else None)
                error = error, ((diff/(o1)).max() if error or isinstance(error,(torch.Tensor)) else None)
                b = bv[i] = error[1]  < args.threshold2 if error[1] else b
                s += (('' if b else 'not ')+f'matched, ({(error[0])}, {error[1]}, {max_error} ) , '+output_names[i]+', '+output_names1[i])
                s += '\n'
            f.write(s)
    bv1 = [o1.shape == o2.shape and torch.all((o1-o2).abs()<(args.threshold1)) for o1, o2 in zip(output1, output3) ]
    if args.export_txts:
        with open(os.path.join(directory,'error2.txt'),'w') as f:
            s = 'status, (max_abs_error, max_abs_rel_error(non-zero), max_error(zero)),output1_name, output2_name\n'
            for i, b in enumerate(bv1):
                o1 = output1[i]
                o2 = output3[i]
                max_error = None
                diff = (torch.abs(o1-o2)) if isinstance(b, torch.Tensor) and o2.numel() else None
                if torch.any(o1 == 0):
                    max_error = diff[torch.where(o1==0)].max()
                diff = diff[torch.where(o1!=0)] if diff is not None else None
                o1 = o1[torch.where(o1!=0)].abs()
                error = (diff.max() if isinstance(diff, torch.Tensor) and diff.numel() else None)
                error = error, ((diff/(o1)).max() if error or isinstance(error,(torch.Tensor)) else None)
                b = bv1[i] = error[1]  < args.threshold2 if error[1] else b
                s+= (('' if b else 'not ')+f'matched, ({( error[0])}, { error[1]}, {max_error}) , '+output_names[i]+', '+output_names1[i])
                s += '\n'
            f.write(s)
    bv2 = [o1.shape == o2.shape and torch.all((o1-o2).abs()<(args.threshold1)) for o1, o2 in zip(output2, output3) ]
    
    if args.export_txts:
        with open(os.path.join(directory,'error3.txt'),'w') as f:
            s = 'status, (max_abs_error, max_abs_rel_error(non-zero), max_error(zero)),output1_name, output2_name\n'
            for i, b in enumerate(bv2):
                o1 = output2[i]
                o2 = output3[i]
                max_error = None
                diff = (torch.abs(o1-o2)) if isinstance(b, torch.Tensor) and o2.numel() else None
                if torch.any(o1 == 0):
                    max_error = diff[torch.where(o1==0)].max()
                diff = diff[torch.where(o1!=0)] if diff is not None else None
                o1 = o1[torch.where(o1!=0)].abs()
                error = (diff.max() if isinstance(diff, torch.Tensor) and diff.numel() else None)
                error = error, ((diff/(o1)).max() if error or isinstance(error,(torch.Tensor)) else None)
                b = bv1[i] = error[1]  < args.threshold2 if error[1] else b
                s+= (('' if b else 'not ')+f'matched, ({( error[0])}, { error[1]}, {max_error}) , '+output_names[i]+', '+output_names1[i])
                s += '\n'
            f.write(s)
    

    return torch_model, output1, output2, output3

if __name__ == '__main__':
    old_model = torch.nn.Sequential(
        torch.nn.Linear(3,2)
    )
    old_model = models.vit_b_16(pretrained=True)
    inp = torch.rand(1,3,224,224)
    old_model.eval()
    output = old_model(inp)
    torch.onnx.export(old_model, (inp,), '/data/ssd/files/a0507161/exps/onnx_exp/test.onnx', training=torch.onnx.TrainingMode.PRESERVE)
    model = onnx.load('/data/ssd/files/a0507161/exps/onnx_exp/test.onnx')
    # model,_ = onnxsim.simplify(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, '/data/ssd/files/a0507161/exps/onnx_exp/test.onnx')
    new_model, o1, o2, o3 = main(['/data/ssd/files/a0507161/exps/onnx_exp/test.onnx', '-e','-a', '-s'], inps=[inp])
    # output1 = new_model(inp)
    pass
    # main(['test.onnx', '-e'])