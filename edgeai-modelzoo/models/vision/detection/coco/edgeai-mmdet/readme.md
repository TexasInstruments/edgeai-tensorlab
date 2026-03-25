
## widerface detection models trained using edgeai-mmdetection

The exported YOLOX onnx models had an isue that prevented it from running in onnxruntime. It was fixed by using this example: https://github.com/microsoft/onnxruntime/issues/7563

using:
onnx-graphsurgeon
installed by:
pip install nvidia-pyindex
pip install onnx-graphsurgeon

or by using: https://github.com/TexasInstruments/edgeai-mmdetection/blob/r8.6/mmdet/utils/onnx_utils.py#L38


```
onnx_load_path = 'model.onnx'
onnx_save_path = 'patched.onnx'
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
```

