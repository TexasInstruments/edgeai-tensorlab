
# inspired by
# https://github.com/microsoft/onnxruntime/issues/7563

import onnx


def get_tensor(onnx_tensor):
    shape = None
    if isinstance(onnx_tensor, onnx.TensorProto):
        shape = onnx_tensor.dims
    elif onnx_tensor.type.tensor_type.HasField("shape"):
        shape = []
        for dim in onnx_tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            elif dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(None)
            #
        #
    #
    if isinstance(onnx_tensor, onnx.TensorProto):
        dtype = onnx_tensor.data_type
    else:
        dtype = onnx_tensor.type.tensor_type.elem_type
    #
    if dtype in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]
    else:
        dtype = None
    #
    tensor_details = dict(name=onnx_tensor.name, shape=shape, dtype=dtype)
    return onnx_tensor.name, tensor_details


def patch_onnx_file(onnx_load_path, onnx_save_path):
    '''
    Apply path to Reduce OP to make it run in onnxruntime
    More details here: https://github.com/microsoft/onnxruntime/issues/7563
    '''
    onnx_model = onnx.load(onnx_load_path)
    tensor_map = dict()
    for i, tensor in enumerate(onnx_model.graph.input):
        tensor_name, tensor_details = get_tensor(tensor)
        tensor_map[tensor_name] = tensor_details
    #
    for i, tensor in enumerate(onnx_model.graph.output):
        tensor_name, tensor_details = get_tensor(tensor)
        tensor_map[tensor_name] = tensor_details
    #
    for i, tensor in enumerate(onnx_model.graph.value_info):
        tensor_name, tensor_details = get_tensor(tensor)
        tensor_map[tensor_name] = tensor_details
    #
    for i, tensor in enumerate(onnx_model.graph.initializer):
        tensor_name, tensor_details = get_tensor(tensor)
        tensor_map[tensor_name] = tensor_details
    #
    for i, node in enumerate(onnx_model.graph.node):
        if "Reduce" in node.op_type:
            has_axes = False
            for attr in node.attribute:
                if attr.name == 'axes':
                    has_axes = True
                #
            #
            # reduce all axes except batch axis
            if not has_axes:
                for input_name in node.input:
                    shape = tensor_map[input_name]["shape"] if input_name in tensor_map else None
                    if shape is not None:
                        axes_attribute = onnx.helper.make_attribute('axes', [i for i in range(1, len(shape))])
                    else:
                        axes_attribute = onnx.helper.make_attribute('axes', [1]) # None
                    #
                    node.attribute.append(axes_attribute)
                #
            #
        #
    #
    onnx.save(onnx_model, onnx_save_path)


def patch_onnx_file_gs(onnx_load_path, onnx_save_path):
    '''
    Apply path to Reduce OP to make it run in onnxruntime
    More details here: https://github.com/microsoft/onnxruntime/issues/7563
    '''
    import onnx_graphsurgeon as gs
    gs_graph = gs.import_onnx(onnx.load(onnx_load_path))
    for i, node in enumerate(gs_graph.nodes):
        if "Reduce" in gs_graph.nodes[i].op and 'axes' not in node.attrs:
            # reduce all axes except batch axis
            if gs_graph.nodes[i].inputs[0].shape is not None:
                gs_graph.nodes[i].attrs['axes'] = [i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))]
            else:
                gs_graph.nodes[i].attrs['axes'] = [1] #None

    new_onnx_graph = gs.export_onnx(gs_graph)
    onnx.save(new_onnx_graph, onnx_save_path)


if __name__ == '__main__':
    import os
    import glob
    #src_folder = '/data/ssd/files/a0393608/work/code/ti/edgeai-algo/edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet'
    src_folder = '/data/ssd/files/a0393608/work/code/ti/edgeai-algo/edgeai-modelzoo/models/vision/detection/widerface/edgeai-mmdet'
    onnx_load_paths = glob.glob(f'{src_folder}/yolox*.onnx')

    for file_index in range(len(onnx_load_paths)):
        onnx_load_path = onnx_load_paths[file_index]
        onnx_save_path = os.path.splitext(onnx_load_path)[0] + 'patched' + '.onnx'
        onnx_reduce_patch(onnx_load_path, onnx_save_path)



