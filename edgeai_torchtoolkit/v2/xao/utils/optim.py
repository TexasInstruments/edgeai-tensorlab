import copy
import torch


class AdaptiveSGD(torch.optim.SGD):
    def step(self, *args, **kwargs):
        # take backup
        param_groups_backup = [
            [copy.deepcopy(param.data) if self.skip_update_fn(param) else None for param in param_group["params"]]
                for param_group in self.param_groups]

        loss = super().step(*args, **kwargs)

        # restore the backup
        for param_group_idx, param_group in enumerate(self.param_groups):
            for param_idx, param in enumerate(param_group["params"]):
                if self.skip_update_fn(param):
                    param.data.copy_(param_groups_backup[param_group_idx][param_idx].data)
                #
            #
        #

        return loss

    def skip_update_fn(self, param):
        if hasattr(param, 'requires_update'):
            return not param.requires_update
        else:
            return False
