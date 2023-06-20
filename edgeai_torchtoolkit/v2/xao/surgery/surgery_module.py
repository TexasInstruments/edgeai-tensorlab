from . import replacer


class SurgeryModule(torch.nn.Module):
    def __int__(self, model, replace_unsupported_ops=True, replacement_dict=None):
        super().__init__()
        self.module = replacer.replace_unsuppoted_layers(model, replacement_dict)

    def get_replacement_dict_default(self):
        return replacer.get_replacement_dict_default()

