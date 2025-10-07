import onnx_graphsurgeon as gs
import onnx
import onnx_ops


def simplify_graph(graph:gs.Graph):
    for idx, node in enumerate(graph.nodes):  
        if node.name.isnumeric() or not node.name:
            node.name = f'{node.op}_{idx}'
        else:
            node.name = node.name.replace('.','_').replace('/','_').replace(':','_')
    for name, tensor in graph.tensors().items():
        if name.isnumeric():
            tensor. name = f'tensor_{name}'
        else:
            tensor.name = tensor.name.replace('.','_').replace('/','_').replace(':','_')

    return graph


def remove_identity(graph:gs.Graph):
    for i, node in enumerate(graph.nodes):
        if node.op == 'Identity':
            outputs = list(node.outputs)
            for out in outputs:
                for o in out.outputs:
                    o.inputs[o.inputs.index(out)] = node.inputs[0]
            graph.nodes.remove(node)


def convert(model_path):
    onnx_model = onnx.load(model_path)
    graph = gs.import_onnx(onnx_model)
    remove_identity(graph)
    simplify_graph(graph)
    torch_model = onnx_ops.get_torch_graph_module(graph)
    return torch_model
