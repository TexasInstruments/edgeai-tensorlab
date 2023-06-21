
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from . import qconfig


class QuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, qconfig_mapping=None, example_inputs=None, is_qat=True, backend=None, qconfig_type=None,
                 total_epochs=0, num_batch_norm_update_epochs=None, num_observer_update_epochs=None):
        super().__init__()
        qconfig_mapping = qconfig_mapping or qconfig.get_default_qconfig_mapping(is_qat, backend, qconfig_type)
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
        num_batch_norm_update_epochs = self.num_batch_norm_update_epochs or max((self.total_epochs//2)-1, 4)
        num_observer_update_epochs = self.num_observer_update_epochs or max((self.total_epochs//2)+1, 6)
        # put the model in expected mode
        super().train(mode=mode)
        # also freeze the params if required
        if mode is True:
            self.freeze(freeze_bn=(self.num_epochs_tracked>=num_batch_norm_update_epochs),
                        freeze_observers=(self.num_epochs_tracked>=num_observer_update_epochs))
        else:
            self.freeze()
        #
        self.num_epochs_tracked += 1
        return self

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
