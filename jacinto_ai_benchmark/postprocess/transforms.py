
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