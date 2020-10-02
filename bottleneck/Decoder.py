import numpy as np
import logging
import torch
from bottleneck.codec import _DenseBlock, _Transition, last_decoding, module_size, UnflattenLatentDimension
from torch import nn
import lamp.modules
import lamp.neuralnets
logger = logging.getLogger('CoarseGraining')


class BaseDecoder(lamp.modules.BaseModule):

    def __init__(self):

        super(BaseDecoder, self).__init__()
        self._binary = None

    @property
    def dim_in(self):
        raise NotImplementedError

    @property
    def dim_out(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def propagate_samples(self, Z):

        means, logsigmas = self.forward(Z)

        if means.shape != logsigmas.shape:
            raise RuntimeError('Implementation assumes that full logsigmas matrix is given; check for broadcasting')

        return means + torch.exp(logsigmas) * torch.randn_like(logsigmas)


    @property
    def dim_latent(self):
        # compatibility reasons
        return self.dim_in

    def freeze_partial(self, num_layers = 1):
        raise NotImplementedError


class LinearDecoder(BaseDecoder):

    def __init__(self, input, output, binary, *, dtype, device):

        super(LinearDecoder, self).__init__()

        assert isinstance(input, int)
        assert isinstance(output, tuple) or isinstance(output, int)

        if isinstance(output, int):
            output = (output,)

        self._input_shape = input
        self._output_shape = output

        self._dim_in = input
        self._dim_out = np.prod(np.array(output))

        self.linear = torch.nn.Linear(self._dim_in, self._dim_out)

        if not binary:
            self.logsigma = torch.nn.Parameter(torch.zeros(output))

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

    def freeze_partial(self, num_layers = 1):
        raise RuntimeError('Cannot freeze part of decoder for type LinearDecoder')

    def forward(self, z):

        batch_dim = z.shape[0]

        if self._binary:
            return torch.nn.functional.sigmoid(self.linear(z).view((batch_dim,) + self._output_shape))
        else:
            return self.linear(z).view((batch_dim,) + self._output_shape), self.logsigma.expand((batch_dim,) + self._output_shape)

    def __repr__(self):
        s = 'Linear Decoder | {} -> {} | Trainable parameters: {}'.format(self._input_shape, self._output_shape, self.num_trainable_parameters)
        return s


class NeuralNetworkDecoder(BaseDecoder):

    def __init__(self, input, output, num_hidden_layers, binary, homoscedastic = False, *, dtype, device):

        super(NeuralNetworkDecoder, self).__init__()

        assert isinstance(input, int)
        assert isinstance(output, tuple) or isinstance(output, int)

        if isinstance(output, int):
            output = (output,)

        self._input_shape = input
        self._output_shape = output

        self._dim_in = input
        self._dim_out = np.prod(np.array(output))

        self._binary = binary
        self._homoscedastic = homoscedastic
        self._num_hidden_layers = num_hidden_layers


        self.nn = None

        if binary:
            self.fc = lamp.neuralnets.FeedforwardNeuralNetwork.FromLinearDecay(self._dim_in, self._dim_out,
                                                                               num_hidden_layers, outf='relu',
                                                                               dtype=dtype, device=device)
        else:
            if homoscedastic:
                self.fc = lamp.neuralnets.FeedforwardNeuralNetwork.FromLinearDecay(self._dim_in, self._dim_out, num_hidden_layers, outf = 'relu', dtype=dtype, device=device)
                self.logsigma = torch.nn.Parameter(torch.zeros(output))
            else:
                raise NotImplementedError

        self._to(dtype=dtype, device=device)

    def forward(self, z):

        batch_dim = z.shape[0]

        if self._binary:
            return self.fc(z).view((batch_dim,) + self._output_shape)
        else:
            # use expand, to avoid explicit copies
            if self._homoscedastic:
                return self.fc(z).view((batch_dim,) + self._output_shape), self.logsigma.expand((batch_dim,) + self._output_shape)
            else:
                raise NotImplementedError

    def freeze_partial(self, num_layers = 1):
        raise RuntimeError('Cannot freeze part of decoder for type NeuralNetworkDecoder')

    def __repr__(self):
        return 'MLP Neural Network Encoder: {} -> {} with {} layers and {} parameters'.format(self._dim_in, self._dim_out, self._num_hidden_layers, self.num_trainable_parameters)



class CNNDecoder(BaseDecoder):

    # based on https://github.com/cics-nd/pde-surrogate
    def __init__(self, target_img_size, dim_latent, latent_img_size=(4,4), latent_img_features=16, init_features = 32, blocks = [3,5,3], binary=False,
                 growth_rate=8, drop_rate=0., upsample='nearest', force_single_output=False, homoscedastic = False):

        super(CNNDecoder, self).__init__()

        if isinstance(target_img_size, tuple):
            # can only deal with same resolutions in all directions
            assert all(element == target_img_size[0] for element in target_img_size)
            target_img_size = target_img_size[0]
        self._output_shape = (target_img_size, target_img_size)


        if isinstance(latent_img_size, tuple):
            # can only deal with same resolution in all directions
            assert all(element == latent_img_size[0] for element in latent_img_size)
            latent_img_size = latent_img_size[0]


        latent_img_dimension = int((latent_img_size**2))*latent_img_features
        self._latent_img_dimension = latent_img_dimension

        num_refinements = len(blocks)
        out_img_size = int(latent_img_size*2**len(blocks))
        if out_img_size != target_img_size:
            s = ('Latent image size of ({},{}) with {} blocks'.format(latent_img_size, latent_img_size, num_refinements) + \
                '(i.e. refinements, doubling of resolution) yields a {}x{} output image size'.format(out_img_size, out_img_size) + \
                ' - target however is ({}x{}). '.format(target_img_size, target_img_size))
            if out_img_size > target_img_size:
                s_add = 'Either reduce the number of blocks (from currently {}) or the latent image size ({}x{})'.format(num_refinements, latent_img_size, latent_img_size)
            else:
                s_add = 'Either increase the number of blocks (from currently {}) or the latent image size ({}x{})'.format(num_refinements, latent_img_size, latent_img_size)
            s = s + s_add

            raise ValueError(s)

        self._dim_in = dim_latent
        self._dim_out = target_img_size**2

        if binary or force_single_output or homoscedastic:
            out_channels = 1
        else:
            out_channels = 2

        if not binary and homoscedastic:
            self.logsigma = torch.nn.Parameter(torch.zeros(target_img_size, target_img_size))

        self._homoscedastic = homoscedastic
        self.latent_map = nn.Linear(dim_latent, latent_img_dimension)

        self.features = nn.Sequential()
        self.features.add_module('unflatten_latent', UnflattenLatentDimension(latent_img_size))
        self.features.add_module('conv0', nn.Conv2d(latent_img_features, init_features, 3, 1, 1, bias=False))

        num_features = init_features
        for i, num_layers in enumerate(blocks):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i < len(blocks) - 1:
                trans_up = _Transition(in_features=num_features,
                                       out_features=num_features // 2,
                                       down=False,
                                       drop_rate=drop_rate,
                                       upsample=upsample)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2

        last_trans_up = last_decoding(num_features, out_channels,
                                      drop_rate=drop_rate, upsample=upsample)
        self.features.add_module('LastTransUp', last_trans_up)

        if binary:
            self.features.add_module('Sigmoid', torch.nn.Sigmoid())

    @property
    def dim_latent_img(self):
        return self._latent_img_dimension

    @property
    def dim_latent(self):
        return self.dim_in

    @property
    def dim_in(self):
        return self._dim_in

    @property
    def dim_out(self):
        raise self._dim_out


    def freeze_partial(self, num_free_layers = 1):

        if num_free_layers != 1:
            raise NotImplementedError
        for param in self.features.parameters():
            param.requires_grad = False

    @property
    def model_size(self):
        return module_size(self)

    def _forward_homoscedastic(self, x, flatten = False):

        batch_dim = x.shape[0]
        out = self.features(self.latent_map(x)).squeeze(1)

        if flatten:
            out = out.view(out.shape[0], -1)

        if self._binary:
            return out
        else:
            if flatten:
                return out, self.logsigma.expand((batch_dim,) + self._output_shape).view(batch_dim,-1)
            else:
                return out, self.logsigma.expand((batch_dim,) + self._output_shape)


    def forward(self, x, flatten=False):

        if self._homoscedastic:
            return self._forward_homoscedastic(x, flatten)

        out = self.features(self.latent_map(x))
        if out.shape[1] > 1:
            mean = out[:,0,:,:]
            logsigmas = out[:,1,:,:]
            if flatten:
                mean = mean.view(mean.shape[0], -1)
                logsigmas = logsigmas.view(logsigmas.shape[0], -1)
            return mean, logsigmas
        else:
            if flatten:
                return out.view(out.shape[0], -1)
            else:
                return out.squeeze(1)

    def forward_test(self, x):
        print('input: {}'.format(x.data.size()))
        for name, module in self.features._modules.items():
            x = module(x)
            print('{}: {}'.format(name, x.data.size()))
        return x

    def __repr__(self):

        num_latent_map_params = sum([param.numel() for param in self.latent_map.parameters()])

        s = 'DenseNet based CNN decoder. \n ' + \
            'Latent dimension: {} \n'.format(self.dim_latent) + \
            'Latent img size: {} \n'.format(self.dim_latent_img) + \
            'Number of parameters: {} \n'.format(self.num_parameters) + \
            ' ==> Convolutions: {} parameters \n'.format(self.num_parameters - num_latent_map_params) + \
            ' ===> Latent map: {} paraneters \n'.format(num_latent_map_params)

        return s


