import os
from edgeai_benchmark.datasets.onnx_backend_dataset import ONNXBackendDataset

def test_onnx_backend_dataset():
    curdir = os.path.dirname(__file__)
    backend_dataset = ONNXBackendDataset(os.path.join(curdir,"test_abs"))
    print(backend_dataset[0]['x'][1,2,3])

    # Check one element is correct
    assert float(backend_dataset[0][0][1,2,3])  == (-1.980796456336975)
