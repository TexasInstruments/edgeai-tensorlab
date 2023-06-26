
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import FakeQuantize

from . import observer
from . import fake_quanitze
from . import qconfig


class QuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, qconfig_type=None, example_inputs=None, is_qat=True, backend="qnnpack",
                 total_epochs=0, num_batch_norm_update_epochs=None, num_observer_update_epochs=None,
                 adaptive_quantization=True):
        super().__init__()
        if not total_epochs:
            raise RuntimeError("total_epochs must be provided")
        #
        qconfig_type = [qconfig.QConfigType(qtype) for qtype in qconfig_type.split(",")]
        qconfig_mapping = qconfig.get_qconfig_mapping(is_qat, backend, qconfig_type[0])
        if is_qat:
            model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
        else:
            model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
        #
        self.module = model
        # other parameters
        self.is_qat = is_qat
        self.backend = backend
        self.qconfig_type = qconfig_type
        self.num_batch_norm_update_epochs = num_batch_norm_update_epochs
        self.num_observer_update_epochs = num_observer_update_epochs
        self.num_epochs_tracked = 0
        self.total_epochs = total_epochs
        self.adaptive_quantization = adaptive_quantization
        self.adaptive_quant_segment = 0
        # set the quantization backend - qnnpack, fbgemm, x86, onednn etc.
        self.set_quant_backend(backend)

    def set_quant_backend(self, backend=None):
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported: " + str(backend))
        torch.backends.quantized.engine = backend

    def load_weights(self, pretrained, *args, strict=True, state_dict_name=None, **kwargs):
        data_dict = torch.load(self, pretrained, *args, **kwargs)
        if state_dict_name:
            state_dict_names = state_dict_name if isinstance(state_dict_name, (list,tuple)) else [state_dict_name]
            for s_name in state_dict_names:
                data_dict = state_dict[s_name] if ((data_dict is not None) and s_name in data_dict) else data_dict
            #
        #
        self.load_state_dict(data_dict, strict=strict)

    def train(self, mode: bool = True):
        # set the default epoch at which freeze occurs during training (if missing)
        num_batch_norm_update_epochs = self.num_batch_norm_update_epochs or ((self.total_epochs//2)-1)
        num_observer_update_epochs = self.num_observer_update_epochs or ((self.total_epochs//2)+1)
        # put the model in expected mode
        super().train(mode=mode)
        # also freeze the params if required
        if mode is True:
            self.freeze(freeze_bn=(self.num_epochs_tracked>=num_batch_norm_update_epochs),
                        freeze_observers=(self.num_epochs_tracked>=num_observer_update_epochs))
            self.adaptive_quant_adjustment()
            self.num_epochs_tracked += 1
        else:
            self.freeze()
        #
        return self

    def adaptive_quant_adjustment(self):
        '''
        adjust quantization parameters on a per segment (group of epochs) basis
        '''
        num_segments = len(self.qconfig_type)
        if (not self.adaptive_quantization) or (num_segments <= 1):
            return
        #
        quant_segment = int(self.num_epochs_tracked * num_segments / self.total_epochs)
        print(f"adaptive_quantization is ON. current quant_segment:{quant_segment}, qconfig_type:{self.qconfig_type[quant_segment]}")
        if quant_segment == 0:
            return
        #
        self.adaptive_quant_segment = quant_segment
        qconfig_new = qconfig.get_qconfig(self.is_qat, self.backend, self.qconfig_type[self.adaptive_quant_segment])
        current_device = next(self.parameters()).device
        for np, mp in self.named_modules():
            for nc, mc in mp.named_children():
                if isinstance(mc, fake_quanitze.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
                    weight_fake_quant = qconfig_new.weight().to(current_device)
                    setattr(mp, nc, weight_fake_quant)
                #
                if isinstance(mc, fake_quanitze.ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
                    activation_fake_quant = qconfig_new.activation().to(current_device)
                    setattr(mp, nc, activation_fake_quant)
                #

    def freeze(self, freeze_bn=True, freeze_observers=True):
        if freeze_observers is True:
            self.apply(torch.ao.quantization.disable_observer)
        elif freeze_observers is False:
            self.apply(torch.ao.quantization.enable_observer)
        #
        if freeze_bn is True:
            self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        elif freeze_bn is False:
            self.apply(torch.nn.intrinsic.qat.update_bn_stats)
        #
        return self

    def unfreeze(self, model, unfreeze_bn=True, unfreeze_observers=True):
        freeze(model, not unfreeze_bn, not unfreeze_observers)
        return model

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)

    def convert(self, inplace=False, device='cpu'):
        # make a copy inorder not to alter the original
        model = self.module if inplace else copy.deepcopy(self.module)
        # convert requires cpu model
        model = model.to(torch.device(device))
        # now do the actual conversion
        model = quantize_fx.convert_fx(model)
        return model
