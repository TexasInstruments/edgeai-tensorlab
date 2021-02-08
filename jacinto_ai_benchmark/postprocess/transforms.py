import numpy as np

class IndexArray():
    def __init__(self, index=0):
        self.index = index

    def __call__(self, input):
        return input[self.index]


class ArgMax():
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor):
        if self.axis is None:
            axis = 1 if tensor.ndim == 4 else 0
        else:
            axis = self.axis
        #
        output = tensor.argmax(axis=axis)
        output = output[0]
        return output

class Concat():
    def __init__(self, axis=-1, start_inxdex=0, end_index=-1):
        self.axis = axis
        self.start_inxdex = start_inxdex
        self.end_index =end_index

    def __call__(self, tensor_list):
        if isinstance(tensor_list, (list,tuple)):
            max_dim = 0
            for t_idx, t in enumerate(tensor_list):
                max_dim = max(max_dim, t.ndim)
            #
            for t_idx, t in enumerate(tensor_list):
                if t.ndim < max_dim:
                    tensor_list[t_idx] = t[...,np.newaxis]
                #
            #
            tensor = np.concatenate(tensor_list[self.start_inxdex:self.end_index], axis=self.axis)
        else:
            tensor = tensor_list
        #
        return tensor