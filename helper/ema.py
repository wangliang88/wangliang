import copy
from copy import deepcopy
import torch
class EMA(object):
    def __init__(self,  model, decay):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ema.to(device)
        self.ema_has_module = hasattr(self.ema, 'module')
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
