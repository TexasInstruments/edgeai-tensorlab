import onnxruntime
import numpy as np
import torch
from onnx2pytorch import convert
import onnx
import onnx_graphsurgeon as gs
import os

def add_all_output(model_path):
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    #%%
    for node in graph.nodes:
        for out in node.outputs:
            if out in graph.outputs:
                continue
            graph.outputs.append(out)
    #%%
    try:
        model = gs.export_onnx(graph)
        output_path = model_path.replace('.onnx', '_all.onnx')
        onnx.save_model(model, output_path)
    except Exception as e:
        print(f"Failed to add all outputs to model's output because of error {e}")
        output_path = model_path
    return output_path


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to ONNX model')
    parser.add_argument('--all_output' ,'-a', action='store_true', help='to export model with outputs of all nodes')
    parser.add_argument('--error_txts' ,'-e', action='store_true', help='to export error txt')
    
    args = parser.parse_args() if args is None else parser.parse_args(args)
    if args.all_output:
        args.model_path = add_all_output(args.model_path)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session1 = onnxruntime.InferenceSession(args.model_path, sess_options, providers=['CPUExecutionProvider'])
    torch_model = convert(args.model_path)
    inputs = []
    input_dict = {}
    dtype_mapping = {'tensor(float)' : np.float32, 
                 'tensor(int64)' : np.int64,
                 'tensor(uint8)' : np.uint8,
                 'tensor(int32)' : np.int32}
    for inp in session1.get_inputs():
        input_dict[inp.name] = np.ones(inp.shape, dtype=dtype_mapping[inp.type])
        inputs.append(  torch.from_numpy(input_dict[inp.name]))
    output1 = session1.run([], input_dict)
    output2 = torch_model(*inputs)
    output_names = [o.name for o in session1.get_outputs()]
    output1 = [torch.from_numpy(o) for o in output1]
    # output1 = [torch.round(o.float(), decimals=5) for o in output1]
    # output2 = [torch.round(o.float(), decimals=5) for o in output2]
    bv = [o1.shape == o2.shape and torch.all(o1==o2) for o1, o2 in zip(output1, output2) ]
    # torch.onnx.export(torch_model, tuple(inputs),args.model_path.replace('.onnx','1.onnx'))
    if args.error_txts:
        with open(args.model_path.replace('.onnx','_error1.txt'),'w') as f:
            s = ''
            for i, b in enumerate(bv):
                s += (('' if b else 'not ')+f'matched ({torch.abs(output1[i]-output2[i]).max() if isinstance(b, torch.Tensor) and output2[i].numel() else "NA"} ) : '+output_names[i])
                s += '\n'
            f.write(s)
    
    session3 = onnxruntime.InferenceSession(args.model_path, sess_options, providers=['CPUExecutionProvider'])
    output3 = session3.run([], input_dict)
    output3 = [torch.from_numpy(o) for o in output3]
    # output3 = [torch.round(o.float(), decimals=5) for o in output3]
    bv1 = [o1.shape == o2.shape and torch.all(o1==o2) for o1, o2 in zip(output1, output3) ]
    if args.error_txts:
        with open(args.model_path.replace('.onnx','_error2.txt'),'w') as f:
            s = ''
            for i, b in enumerate(bv1):
                s+= (('' if b else 'not ')+f'matched ({torch.abs(output1[i]-output3[i]).max()if  isinstance(b, torch.Tensor) and output2[i].numel() else "NA"}) : '+output_names[i])
                s += '\n'
            f.write(s)
    pass

if __name__ == '__main__':
    main(['/data/ssd/files/a0507161/exps/onnx_exp/opt_ar_rgb_modified_all.onnx', '-e'])