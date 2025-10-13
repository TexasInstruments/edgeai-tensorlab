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
import torchvision as tv
import onnxsim
import onnx
import onnx_graphsurgeon as gs
import os

from edgeai_onnx2torchmodel.onnx2pytorch import convert

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--simplify','-s', action='store_true', help='to simplify model')
    parser.add_argument('--cuda','-c', action='store_true', help='to use cuda')
    parser.add_argument('--export_txts' ,'-e', action='store_true', help='to export error txt')
    parser.add_argument('--threshold1' ,'-t1', type=float, default=1e-5, help='to export error txt')
    parser.add_argument('--threshold2' ,'-t2', type=float, default=1e-2, help='to export error txt')
    args = parser.parse_args() if args is None else parser.parse_args(args)
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    model = tv.models.get_model(args.model_name, pretrained=True)
    inp = torch.rand((1, 3, 224, 224))
    model.eval()
    if args.cuda:
        model = model.cuda()
        inp = inp.cuda()
    output1 = model(inp)
    if args.cuda:
        inp = inp.cpu()
        model = model.cpu()
    model_path = os.path.join(output_dir, f'{args.model_name}.onnx')
    torch.onnx.export(model, (inp,), model_path,)
    onnx_model = onnx.load(model_path)
    if args.simplify:
        onnx_model, check= onnxsim.simplify(onnx_model)
        if not check:
            raise ValueError('Simpplification Failed')
    else:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.save_model(onnx_model, model_path)
    
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session1 = onnxruntime.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    input_feed = {session1.get_inputs()[0].name: inp.numpy()}
    output2 = session1.run(None, input_feed)
    output2 = [torch.from_numpy(o) for o in output2]
    new_model = convert(model_path)
    new_model.eval()
    if args.export_txts:
        with open(os.path.join(output_dir,'graph.txt'), 'w') as f:
            f.write(str(new_model.graph))
        with open(os.path.join(output_dir,'code.txt'), 'w') as f:
            f.write(str(new_model.code))
    if args.cuda:
        new_model = new_model.cuda()
        inp = inp.cuda()
    output3 = new_model(inp)
    if args.cuda:
        inp = inp.cpu()
        new_model = new_model.cpu()
    if len(output2)==1:
        output1 = [output1]
        output3 = [output3]
    output1 = [o.detach() if hasattr(o,'detach') else torch.tensor(o) for o in output1]
    output3 = [o.detach() if hasattr(o,'detach') else torch.tensor(o) for o in output3]
    
    output1 = [o.float() for o in output1]
    output2 = [o.float() for o in output2]
    output3 = [o.float() for o in output3]
    
    output_names = [o.name for o in session1.get_outputs()]
    # output3 = [torch.round(o.float(), decimals=5) for o in output3]
    def export_error(output1, output2, name):
        if args.export_txts:
            bv = [o1.shape == o2.shape and torch.all((o1-o2).abs() < (args.threshold1)) for o1, o2 in zip(output1, output2) ]
            with open(os.path.join(output_dir,f'{name}.txt'),'w') as f:
                s = 'status, (max_abs_error, max_abs_rel_error(non-zero), max_error(zero)),output1_name\n'
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
                    s += (('' if b else 'not ')+f'matched, ({(error[0])}, {error[1]}, {max_error} ) , '+output_names[i])
                    s += '\n'
                f.write(s)
    export_error(output1, output2, 'error1')
    export_error(output1, output3, 'error2')    
    export_error(output2, output3, 'error3')
    return new_model, output1, output2, output3

if __name__ == '__main__':
    # print(tv.models.list_models())
    # exit()
    model_names = ['resnet18', 'mobilenet_v2', 'vit_b_16']
    args = ['./workdir/torch2torch_test','-e','-s']
    for model_name in model_names:
        model = main([model_name] + args)[0]
        example_inputs = []
        for inp, info in model.input_info.items():
            example_inputs.append(torch.rand(info['shape'], dtype=info['dtype']))
            
        pt2e_model = torch.export.export(model, tuple(example_inputs)).module()

        # Step 2. quantization
        from torchao.quantization.pt2e.quantize_pt2e import (prepare_qat_pt2e, convert_pt2e,)
        # install executorch: `pip install executorch`
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (get_symmetric_quantization_config, XNNPACKQuantizer,)


        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        student_model = prepare_qat_pt2e(pt2e_model, quantizer)


        student_model1 = convert_pt2e(student_model)
        print('pt2e_model.graph',pt2e_model.graph, 'pt2e_model.code',pt2e_model.code, sep='\n\n')
        print('student_model.graph',student_model.graph, 'student_model.code',student_model.code, sep='\n\n')
        print('student_model1.graph',student_model1.graph, 'student_model1.code',student_model1.code, sep='\n\n')
        
        break
        # break
    