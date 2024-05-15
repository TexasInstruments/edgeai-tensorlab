from .dataset_base import DatasetBase
from .imagenet import ImageNetCls
import numpy as np
import os


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
        super().__init__(**kwargs)

        # moving this import hre, to make it conditional
        import onnx
        from onnx.onnx_ml_pb2 import TensorProto
        from onnx import numpy_helper

        self.path = path

        test_data_set_0 = os.path.join(path, "test_data_set_0")
        assert os.path.exists(test_data_set_0), "provided path must have test_data_set_0 subdirectory with protobuff i/o's"
        
        self.inputs           = {}
        self.expected_outputs = {}
        in_counter     = 0
        out_counter    = 0
        onnx_model     = onnx.load(os.path.join(path,"model.onnx"))
        for fname in os.listdir(test_data_set_0):
            fpath = os.path.join(test_data_set_0, fname)
            assert os.path.splitext(fpath)[1] == ".pb", " non protobuf file found"

            file_bytes               = open(fpath, mode = 'rb').read()
            tensor                   = TensorProto.FromString(file_bytes)
            np_array                 = numpy_helper.to_array((tensor))
            tensor_name              = tensor.name 
            
            if("input_" in fname): 
                if(tensor_name == ""):
                    tensor_name = onnx_model.graph.input[in_counter].name

                self.inputs[tensor_name] = np_array
                in_counter += 1
            else:
                assert "output_" in fname
                if(tensor_name == ""):
                    tensor_name = onnx_model.graph.output[out_counter].name

                self.expected_outputs[tensor_name] = np_array
                out_counter += 1
        

    def __getitem__(self, idx, **kwargs):
        assert idx == 0
        return self.inputs


    def __len__(self):
        return 1
    
    # Evaluate inference outputs by reporting the maximum normalized mean-squared-error (max NMSE) of all network outputs
    def __call__(self, output_list, **kwargs):

        
        assert isinstance(output_list, list) and len(output_list) == 1, \
            "Expected output_list is a nested list with one element"
        output_list = output_list[0]

        # Convert output_list to output_dict based on output names
        out_info = onnx.load(os.path.join(self.path, "model.onnx")).graph.output
        output_dict = {}
        for output, info in zip(output_list, out_info):
            output_dict[info.name] = output

        # Compute the max_nmse
        max_nmse = 0
        for out_name, output in output_dict.items():
            expected_output = self.expected_outputs.get(out_name)
            assert expected_output is not None, f" No expected output for output named {out_name}"
            
            # Convert output to float
            output          = np.squeeze(output.astype(float))
            expected_output = np.squeeze(expected_output.astype(float))

            assert expected_output.shape == output.shape, f" Shape mismatch! Expected {expected_output.shape} got {output.shape}"       
            max_nmse = max(max_nmse, ((expected_output - output)**2/np.maximum(expected_output,1**-20)).mean())
            
        return {"max_nmse" : max_nmse}

    


