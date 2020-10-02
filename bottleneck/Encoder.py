import numpy as np
from bottleneck.codec import _DenseBlock, _Transition, FlattenImage, SplitModule, activation, module_size
import torch
import lamp.modules
import torch.nn as nn
import lamp.neuralnets


class BaseEncoder(lamp.modules.BaseModule):

    def __init__(self):

        super(BaseEncoder, self).__init__()

    @property
    def dim_in(self):
        raise NotImplementedError

    @property
    def dim_out(self):
        raise NotImplementedError



class LinearEncoder(BaseEncoder):

    def __init__(self, input, output, binary, *, dtype, device):

        super(LinearEncoder, self).__init__()

        assert isinstance(input, tuple) or isinstance(input, int)
        assert isinstance(output, int)

        if isinstance(input, int):
            input = (input,)

        self._input_shape = input
        self._output_shape = output

        self._dim_in = np.prod(np.array(input))
        self._dim_out = output

        self._linear = torch.nn.Linear(self._dim_in, self._dim_out)

        self._logsigma = torch.nn.Parameter(torch.zeros(output))

        self._binary = binary

        self._to(dtype=dtype, device=device)

    @torch.no_grad()
    def init_logsigma(self, value):
        self._logsigma.data.fill_(value)

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        return self._dim_out

    def forward(self, x):

        batch_dim = x.shape[0]

        if self._binary:
            return self._linear(x.view(batch_dim,-1)), self._logsigma.unsqueeze(0).repeat((batch_dim,) + (1,)*self._logsigma.dim())
        else:
            # use expand, to avoid explicit copies
            return self._linear(x.view(batch_dim,-1)), self._logsigma.unsqueeze(0).repeat((batch_dim,) + (1,)*self._logsigma.dim())

    def __repr__(self):
        s = 'Linear Encoder | {} -> {} | Trainable parameters: {}'.format(self._input_shape, self._output_shape, self.num_trainable_parameters)
        return s



class NeuralNetworkEncoder(BaseEncoder):

    def __init__(self, input, output, num_hidden_layers, binary, homoscedastic = False, *, dtype, device):

        super(NeuralNetworkEncoder, self).__init__()

        assert isinstance(input, tuple) or isinstance(input, int)
        assert isinstance(output, int)

        if isinstance(input, int):
            input = (input,)

        self._input_shape = input
        self._output_shape = output

        self._dim_in = np.prod(np.array(input))
        self._dim_out = output

        if binary:
            self.fc = lamp.neuralnets.FeedforwardNeuralNetwork.FromLinearDecay(self._dim_in, self._dim_out,
                                                                               num_hidden_layers, outf='relu',
                                                                               dtype=dtype, device=device)
        else:
            if homoscedastic:
                self.fc = lamp.neuralnets.FeedforwardNeuralNetwork.FromLinearDecay(self._dim_in, self._dim_out,
                                                                                   num_hidden_layers, outf='relu',
                                                                                   dtype=dtype, device=device)
                self.logsigma = torch.nn.Parameter(torch.zeros(output))
            else:
                raise NotImplementedError

        self._binary = binary
        self._num_hidden_layers = num_hidden_layers
        self._homoscedastic = homoscedastic

        self._to(dtype=dtype, device=device)

    def forward(self, z):

        batch_dim = z.shape[0]

        if self._binary:
            return self.fc(z)
        else:
            if self._homoscedastic:
                return self.fc(z), self.logsigma.expand((batch_dim, self._output_shape))
            else:
                raise NotImplementedError

    def __repr__(self):
        return 'MLP Neural Network Encoder: {} -> {} with {} layers and {} parameters'.format(self._dim_in, self._dim_out, self._num_hidden_layers, self.num_trainable_parameters)



class CNNEncoder(BaseEncoder):

    def __init__(self, imsize, latent_dim, blocks=[3,5,3], growth_rate=8,
                 init_features=32, drop_rate=0, makedeterministic=False):

        # based on https://github.com/cics-nd/pde-surrogate

        super(CNNEncoder, self).__init__()

        bottleneck = True
        in_channels = 1
        bn_size = 8 #
        out_activation = None

        enc_block_layers = blocks

        self.features = nn.Sequential()
        pad = 3 if imsize % 2 == 0 else 2
        self.features.add_module('In_conv', nn.Conv2d(in_channels, init_features,
                                                      kernel_size=7, stride=2, padding=pad, bias=False))
        num_features = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans_down = _Transition(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True,
                                     drop_rate=drop_rate)

            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2


        imsize_new = int(imsize /  (2**(len(blocks)+1)))
        input_dim_dense_layer = num_features * imsize_new * imsize_new
        self.features.add_module('FlattenImage', FlattenImage())
        self.features.add_module('FC', torch.nn.Linear(input_dim_dense_layer, input_dim_dense_layer))
        self.features.add_module('ActivRelu', torch.nn.ReLU())

        if makedeterministic:
            self.features.add_module('FinalFC', torch.nn.Linear(input_dim_dense_layer, latent_dim))
        else:
            self.features.add_module('SplitDense', SplitModule(input_dim_dense_layer, latent_dim))



        if out_activation is not None:

            self.features.add_module(out_activation, activation(out_activation))


    def forward(self, x):

        if x.dim() < 4:
            x = x.unsqueeze(1)

        return self.features(x)


    def forward_test(self, x):

        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            try:
                print('{}: {}'.format(name, x.data.size()))
            except:
                pass
        return x

    @property
    def model_size(self):
        return module_size(self)

    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))




