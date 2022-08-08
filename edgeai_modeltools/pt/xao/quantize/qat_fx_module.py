import copy
import torch
import torch.quantization.quantize_fx as quantize_fx
from ... import xnn


class QATFxModule(torch.nn.Module):
    def __init__(self, module, qconfig_dict=None, pretrained=None, pretrained_after=True):
        super().__init__()
        if qconfig_dict is None:
            qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
        #
        module.train()
        if not pretrained_after:
            self.load_weights_(module, pretrained=pretrained)
        #
        module = quantize_fx.prepare_qat_fx(module, qconfig_dict)
        self.module = module
        if pretrained_after:
            self.load_weights_(module, pretrained=pretrained)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def convert(self):
        self.module = quantize_fx.convert_fx(self.module)
        return self

    def load_weights_(self, module, pretrained=None, change_names_dict=None):
        # Load weights for accuracy evaluation of a QAT model
        if pretrained is not None and pretrained is not False:
            print("=> using pre-trained model from {}".format(pretrained))
            if hasattr(module, 'load_weights'):
                module.load_weights(args.pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
            else:
                xnn.utils.load_weights(module, pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
            #
