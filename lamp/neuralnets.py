import lamp.modules
import torch
import numpy as np
from lamp.utils import get_activation_function


class FeedforwardNeuralNetwork(lamp.modules.BaseModule):

    def __init__(self, dim_in, dim_out, architecture, dropout, outf=None, dtype = None, device = None):

        super(FeedforwardNeuralNetwork, self).__init__()

        architecture = [dim_in] + architecture + [dim_out]

        self.layers = torch.nn.Sequential()

        for n in range(len(architecture)-1):

            self.layers.add_module('fc{}'.format(n+1), torch.nn.Linear(architecture[n], architecture[n+1]))

            if dropout is not None:
                self.layers.add_module('dropout{}'.format(n+1), torch.nn.Dropout(p=0.5))

            if n != len(architecture) - 2:
                self.layers.add_module('activ{}'.format(n+1), torch.nn.ReLU())
            else:
                if outf is not None:
                    self.layers.add_module('out_fct', get_activation_function(outf))

        self._to(device=device, dtype=dtype)

    def forward(self, x):

        return self.layers(x)

    @classmethod
    def FromLinearDecay(cls, dim_in, dim_out, num_hidden_layers, outf = None, dropout=None, dtype=None, device=None):


        architecture = list(np.linspace(dim_in, dim_out, num_hidden_layers+2).astype(int))

        architecture_hidden = architecture[1:-1]

        return cls(dim_in, dim_out, architecture_hidden, dropout, outf, dtype, device)









