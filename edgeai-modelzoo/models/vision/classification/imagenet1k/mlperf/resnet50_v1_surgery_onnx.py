import onnx_graphsurgeon as gs
import onnx
#import onnxsim
import os

# This regarding the model resnet50-v1.5 d4scribed int eh following url:
# https://github.com/mlcommons/inference/tree/r2.1/vision/classification_and_detection
# In the actual file download link, it is referred to as resnet50_v1.onnx
# https://zenodo.org/record/4735647/files/resnet50_v1.onnx


os.system("wget https://zenodo.org/record/4735647/files/resnet50_v1.onnx")
os.system("python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param unk__616 --dim_value 1 resnet50_v1.onnx resnet50_v1_static.onnx")


model = onnx.load("resnet50_v1_static.onnx")

#model, checker = onnxsim.simplify(model)
model = onnx.shape_inference.infer_shapes(model)

graph = gs.import_onnx(model)

#print(graph.nodes)

output_tensor = None
for node in graph.nodes:
    for ot in node.outputs:
        if ot.name == "resnet_model/dense/BiasAdd:0":
            output_tensor = ot
            break

graph.outputs = [output_tensor]

graph.cleanup()
graph.toposort()

model = gs.export_onnx(graph)

onnx.save(model, "resnet50_v1_simp.onnx")
