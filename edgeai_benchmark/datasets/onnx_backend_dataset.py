from .dataset_base import DatasetBase
from .random import RandomDataset
from .imagenet import ImageNetCls
import numpy as np
import os
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.onnx_ml_pb2 import TensorProto
from onnx import numpy_helper


class ONNXBackendDataset(DatasetBase):
    '''
    Dataset used for onnx backend tests
    '''

    def __init__(self, path : str, **kwargs):
        '''
        path - base path of test
            folder should include model.onnx
            subfolder test_data_set_0 should include inputs and outputs
        '''

        test_data_set_0 = os.path.join(path, "test_data_set_0")
        assert os.path.exists(test_data_set_0), "provided path must have test_data_set_0 subdirectory with protobuff i/o's"
        
        self.inputs           = []
        self.expected_outputs = []
        for fname in os.listdir(test_data_set_0):
            fpath = os.path.join(test_data_set_0, fname)
            assert os.path.splitext(fpath)[1] == ".pb", " non protobuf file found"

            file_bytes               = open(fpath, mode = 'rb').read()
            tensor                   = TensorProto.FromString(file_bytes)
            np_array                 = numpy_helper.to_array((tensor))
            if("input_" in fname):
                self.inputs.append(np_array)
            else:
                assert "output_" in fname
                self.expected_outputs.append(np_array)

        super().__init__(**kwargs)
        ...
        

    def __getitem__(self, idx, **kwargs):
        assert idx == 0
        return self.inputs


    def __len__(self):
        return 1
    


