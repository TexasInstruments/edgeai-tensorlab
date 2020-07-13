from pytorch_jacinto_ai import xnn

class XMMDetQuantCalibrateModule(xnn.quantize.QuantCalibrateModule):
    def __init__(self, model,  dummy_input, *args, forward_analyze_method='forward_dummy', **kwargs):
        super().__init__(model, dummy_input, *args, forward_analyze_method=forward_analyze_method, **kwargs)

    def forward(self, img, *args, **kwargs):
        return super().forward(img, *args, **kwargs)


class XMMDetQuantTrainModule(xnn.quantize.QuantTrainModule):
    def __init__(self, model,  dummy_input, *args, forward_analyze_method='forward_dummy', **kwargs):
        super().__init__(model, dummy_input, *args, forward_analyze_method=forward_analyze_method, **kwargs)

    def forward(self, img, *args, **kwargs):
        return super().forward(img, *args, **kwargs)


class XMMDetQuantTestModule(xnn.quantize.QuantTestModule):
    def __init__(self, model,  dummy_input, *args, forward_analyze_method='forward_dummy', **kwargs):
        super().__init__(model, dummy_input, *args, forward_analyze_method=forward_analyze_method, **kwargs)

    def forward(self, img, *args, **kwargs):
        return super().forward(img, *args, **kwargs)


def is_mmdet_quant_module(model):
    return isinstance(model, (XMMDetQuantTrainModule, XMMDetQuantTestModule))