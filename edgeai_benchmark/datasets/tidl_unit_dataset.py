from .dataset_base import DatasetBase
from .imagenet import ImageNetCls
import numpy as np
import os

class TIDLUnitDataset(DatasetBase):
    '''
    Dataset used for tidl unit tests
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
            assert (os.path.splitext(fpath)[1] in [".pb",".bin"]), "Invalid file format - Allowed values are, protobuf(.pb), python list(.bin)"
            if os.path.splitext(fpath)[1] == ".pb":
                file_bytes               = open(fpath, mode = 'rb').read()
                tensor                   = TensorProto.FromString(file_bytes)
                np_array                 = numpy_helper.to_array((tensor))
                tensor_name              = tensor.name 
            elif os.path.splitext(fpath)[1] == ".bin":
                file_bytes               = open(fpath, mode = 'rb').read()
                if("input_" in fname): 
                    tensor_info = onnx_model.graph.input[in_counter]
                elif("output_" in fname):
                    tensor_info = onnx_model.graph.output[out_counter]
                else:
                    assert 0,"Incorrect file name, should start with input_ for input files and output_ for output files "
                shape_dims = tensor_info.type.tensor_type.shape.dim
                shape = []
                for d in shape_dims:
                    if d.dim_value > 0:
                        shape.append(d.dim_value)
                    else:
                        shape.append(1)  
                onnx_data_type = tensor_info.type.tensor_type.elem_type 
                if onnx_data_type == TensorProto.FLOAT:
                    dtype = np.float32
                elif onnx_data_type == TensorProto.UINT8:
                    dtype = np.uint8
                elif onnx_data_type == TensorProto.INT8:
                    dtype = np.int8
                elif onnx_data_type == TensorProto.UINT16:
                    dtype = np.uint16
                elif onnx_data_type == TensorProto.INT16:
                    dtype = np.int16
                elif onnx_data_type == TensorProto.INT32:
                    dtype = np.int32
                elif onnx_data_type == TensorProto.INT64:
                    dtype = np.int64
                else:
                    dtype = np.float32
                np_array = np.frombuffer(file_bytes, dtype=dtype)
                np_array = np_array.reshape(shape)
                tensor = numpy_helper.from_array(np_array, name=tensor_name)
                tensor_name = ""
            
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
        import onnx
        out_info = onnx.load(os.path.join(self.path, "model.onnx")).graph.output
        output_dict = {}
        for output, info in zip(output_list, out_info):
            output_dict[info.name] = output

        # Compute the max_nmse
        max_nmse = 0
        for out_name, output in output_dict.items():
            expected_output = self.expected_outputs.get(out_name)
            assert expected_output is not None, f" No expected output for output named {out_name}"
            
            # If output is of object type, assert exact equality
            if(output.dtype == object):
                np.testing.assert_array_equal(output, expected_output)
                max_nmse = 0
                continue

            output          = np.squeeze(output.astype(float))
            expected_output = np.squeeze(expected_output.astype(float))

            assert expected_output.shape == output.shape, f" Shape mismatch! Expected {expected_output.shape} got {output.shape}"       
            max_nmse = max(max_nmse, ((expected_output - output)**2/np.maximum(expected_output,1**-20)).mean())

            assert expected_output.shape == output.shape, f" Shape mismatch! Expected {expected_output.shape} got {output.shape}"       
            max_nmse = max(max_nmse, ((expected_output - output)**2/np.maximum(expected_output,1**-20)).mean())
            
        return {"max_nmse" : max_nmse}

    


