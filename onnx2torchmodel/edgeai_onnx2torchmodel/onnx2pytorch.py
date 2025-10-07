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
    nodes = [node for node in graph.nodes if node.op == 'Identity']
    for i, node in enumerate(nodes):
        if node.op == 'Identity':
            outputs = list(node.outputs)
            for out in outputs:
                for o in out.outputs:
                    if isinstance( node.inputs[0], gs.Constant ):
                        o.inputs[o.inputs.index(out)] = gs.Constant(out.name, node.inputs[0].values)
                    elif isinstance(node.inputs[0], gs.Variable):
                        o.inputs[o.inputs.index(out)] = gs.Variable(out.name, node.inputs[0].dtype, node.inputs[0].shape)
            graph.nodes.remove(node)
            if out in graph.outputs:
                graph.outputs.remove(out)


def convert(model_path):
    onnx_model = onnx.load(model_path)
    graph = gs.import_onnx(onnx_model)
    remove_identity(graph)
    simplify_graph(graph)
    torch_model = onnx_ops.get_torch_graph_module(graph)
    model = gs.export_onnx(graph)
    onnx.save_model(model, model_path)
    return torch_model
