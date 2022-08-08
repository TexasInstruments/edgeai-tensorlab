import copy
import torch
import torch.quantization.quantize_fx as quantize_fx


class QuantTrainFx():
    def __init__(self, module, qconfig_dict=None):
        if qconfig_dict is None:
            qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
        #
        module.train()
        module = quantize_fx.prepare_qat_fx(module, qconfig_dict)
        self.module = module

    def forward(self, *args, **kwargs):
        self.module(*args, **kwargs)

    def convert(self):
        self.module = quantize_fx.convert_fx(self.module)
