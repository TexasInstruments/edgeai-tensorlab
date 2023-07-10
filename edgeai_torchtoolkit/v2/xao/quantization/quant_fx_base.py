
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import FakeQuantize

from .. import surgery

from . import observer
from . import fake_quanitze
from . import qconfig


class QuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, qconfig_type=None, example_inputs=None, is_qat=True, backend="qnnpack",
                 total_epochs=0, num_batch_norm_update_epochs=None, num_observer_update_epochs=None,
                 qconfig_mode=qconfig.QConfigMode.DEFAULT):
        super().__init__()
        if not total_epochs:
            raise RuntimeError("total_epochs must be provided")
        #

        # replace ReLU6() by ReLU() as torch.ao.quantization currently does not handle ReLU6() correctly
        replacement_dict = {torch.nn.ReLU6(): torch.nn.ReLU()}
        model = surgery.replace_unsuppoted_layers(model, replacement_dict=replacement_dict)

        # split if qconfig is a comma separated list of segments
        # (qconfig will change after some epochs if this has comma separated values)
        qconfig_type = tuple(qconfig_type.split(","))
        # further split based on + for mixed precision
        qconfig_type = [qconf.split("+") for qconf in qconfig_type]
        if any([len(qconf) > 2 for qconf in qconfig_type]):
            raise RuntimeError(f"maximum of 2 entries are supported in qconfig_type:{qconfig_type}")
        #
        qconfig_mapping = qconfig.get_qconfig_mapping(is_qat, backend, qconfig_type[0])
        if is_qat:
            model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
        else:
            model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
        #
        model = qconfig.adjust_mixed_precision_qconfig(model, is_qat, backend, qconfig_type[0])
        self.module = model

        # other parameters
        self.is_qat = is_qat
        self.backend = backend
        self.qconfig_type = qconfig_type
        self.qconfig_mode = qconfig.QConfigMode(qconfig_mode)
        self.num_batch_norm_update_epochs = num_batch_norm_update_epochs
        self.num_observer_update_epochs = num_observer_update_epochs
        self.num_epochs_tracked = 0
        self.total_epochs = total_epochs
        # set the quantization backend - qnnpack, fbgemm, x86, onednn etc.
        self.set_quant_backend(backend)
        self.quant_segment_index = 0

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
            self.adjust_gradual_quantization()
            self.num_epochs_tracked += 1
        else:
            self.freeze()
        #
        return self

    def adjust_gradual_quantization(self):
        '''
        adjust quantization parameters on epoch basis
        '''
        if len(self.qconfig_type) > 1:
            quant_segment_index = self.num_epochs_tracked//(self.total_epochs//len(self.qconfig_type))
            qconfig_type = self.qconfig_type[quant_segment_index]
            if quant_segment_index != self.quant_segment_index:
                print(f"updating qconfig - quant_segment_index:{quant_segment_index}, qconfig_type:{qconfig_type}")
                # change the qconfig if it is a comma separated list and this is a new segment
                current_device = next(self.parameters()).device
                qconfig_dict = qconfig.get_qconfig(self.is_qat, self.backend, qconfig_type[0])
                for np, mp in list(self.named_modules()):
                    for nc, mc in list(mp.named_children()):
                        if isinstance(mc, fake_quanitze.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
                            setattr(mp, nc, qconfig_dict.weight().to(current_device))
                        #
                        if isinstance(mc, fake_quanitze.ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
                            setattr(mp, nc, qconfig_dict.activation().to(current_device))
                        #
                    #
                #
                self.module = qconfig.adjust_mixed_precision_qconfig(
                    self.module, self.is_qat, self.backend, qconfig_type)
                self.quant_segment_index = quant_segment_index
            #
        #
        if self.qconfig_mode == qconfig.QConfigMode.GRADUAL_QUANTIZATION:
            total_epochs_knee = max((self.total_epochs//2)-3, 1)
            alpha = min(self.num_epochs_tracked/total_epochs_knee, 1.0)
            adaptive_quant_bypass = (1 - alpha)
            print(f"updating qconfig adaptive_quant_bypass - qconfig_mode:{self.qconfig_mode}, adaptive_quant_bypass:{adaptive_quant_bypass}")
            for n, m in list(self.named_modules()):
                # low-bit weight quantization is usually okay - so not using gradual mode for weights
                # if isinstance(m, fake_quanitze.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
                #     m.set_adaptive_params(adaptive_quant_bypass=adaptive_quant_bypass)
                # #
                if isinstance(m, fake_quanitze.ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
                    m.set_adaptive_params(adaptive_quant_bypass=adaptive_quant_bypass)
                #
            #
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
