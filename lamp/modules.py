import torch
from lamp.utils import get_default_device, get_default_dtype, get_dtype, get_device

class BaseModule(torch.nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def num_trainable_parameters(self):
        return sum((parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    @property
    def num_parameters(self):
        return sum((parameter.numel() for parameter in self.parameters()))

    def gradient_norm(self):

        return torch.mean(torch.cat([torch.norm(param.grad) for param in self.parameters() if param.grad is not None]))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def load(self, *args, **kwargs):
        self.load_state_dict(*args, **kwargs)

    def copy_values_from(self, module):

        r = self.load_state_dict(module.state_dict())

        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError

    def state(self, *args, **kwargs):
        return self.state_dict(*args, **kwargs)

    def _to(self, **kwargs):

        try:
            dtype = kwargs.pop('dtype')

            if dtype is not None:
                dtype = get_dtype(dtype)
        except KeyError:
            dtype = None

        try:
            device = kwargs.pop('device')
            if device is not None:
                device = get_device(device)
        except KeyError:
            device = None

        if device is None:
            device = get_default_device()
        if dtype is None:
            dtype = get_default_dtype()

        self.to(dtype=dtype, device=device)



