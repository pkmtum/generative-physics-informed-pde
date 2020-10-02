import numpy as np
import torch
import dolfin as df
from utils.conversions import  Convert_ScipySparse_PyTorchSparse
from fawkes.utils import AssembleBasisFunctionMatrix
from bottleneck.utils import UnitGaussianKullbackLeiblerDivergence, relative_error
from lamp.neuralnets import  FeedforwardNeuralNetwork
from bottleneck.ROM import ROM
import lamp
from lamp.utils import coefficient_of_determination
import copy

class PhysicsResolutionInterpolator(lamp.modules.BaseModule):

    def __init__(self, physics, mode = 'ManualInterpolation', only_free_dofs = True, dtype = None, device = None):

        # note: does NOT assume conforming meshes
        super(PhysicsResolutionInterpolator, self).__init__()

        self._physics = physics
        self._only_free_dofs = only_free_dofs
        self._mode = mode

        self._W = None

        self._assemble(dtype, device)

        self._to(dtype = dtype, device = device)

    @property
    def dim_in(self):
        return self._W.shape[0]

    @property
    def dim_out(self):
        return self._W.shape[1]

    def _assemble(self, dtype, device):

        if self._mode.lower() == 'manualinterpolation':

            Vf, Vc = self._physics['fom'].V, self._physics['rom'].V
            free_dofs = self._physics['fom'].free_dofs

            coords = Vf.mesh().coordinates()
            dvmap = df.dof_to_vertex_map(Vf)
            points = np.zeros((Vf.dim(), self._physics['fom'].tdim ))

            for i, mapped_dof in enumerate(dvmap):
                points[i, :] = coords[mapped_dof, :]

            if self._only_free_dofs:
                points = points[free_dofs, :]

            W = AssembleBasisFunctionMatrix(Vc, points, ReturnType='scipy')

            self._W = Convert_ScipySparse_PyTorchSparse(W.T, dtype=dtype, device=device)

            # ugly fix: cast to dense tensor
            self._W = self._W.to_dense()

        else:
            raise ValueError('Interpolation mode unknown')

    def forward(self, x):

        return torch.matmul(x, self._W)



class VariationalApproximation(lamp.modules.BaseModule):

    def __init__(self, dim, N, X = None, *, dtype = None, device = None, requires_grad=True):

        super(VariationalApproximation, self).__init__()

        if dtype is None and X is not None:
            dtype = X.dtype
        if device is None and X is not None:
            device = X.device

        self._logsigma = torch.nn.Parameter(torch.zeros(N, dim, requires_grad = requires_grad, dtype=dtype, device=device))
        self._mean = torch.nn.Parameter(torch.zeros(N, dim, requires_grad = requires_grad, dtype=dtype, device=device))

        # auxiliary information
        self._X = X
        self._N = N
        self._dim = dim

        if self._X is not None:
            self._identifer = hash(self._X)
        else:
            self._identifier = None

        self._to(dtype=dtype, device=device)

    @property
    def N(self):
        return self._N

    @property
    def dim(self):
        return self._dim

    @property
    def dtype(self):
        return self._mean.dtype

    @property
    def device(self):
        return self._mean.device

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        assert value.shape[0] == self._N
        assert value.shape[1] == self._dim
        assert value.dtype == self.dtype
        assert value.device == self.device
        self._mean.data = value

    @property
    def logsigma(self):
        return self._logsigma

    @logsigma.setter
    def logsigma(self, value):
        assert value.shape[0] == self._N
        assert value.shape[1] == self._dim
        assert value.dtype == self.dtype
        assert value.device == self.device
        self._logsigma.data = value


    def init(self, mean, logsigma):

        self.mean = mean
        self.logsigma = logsigma

    def init_standard_deviation(self, stddev):

        self._logsigma.data.fill_(np.log(stddev))

    def freeze(self):

        self._logsigma.requires_grad = False
        self._mean.requires_grad = False

    def freeze_mean(self):
        self._mean.requires_grad = False

    def unfreeze(self):

        self._logsigma.requires_grad = True
        self._mean.requires_grad = True

    def init_by_encoder(self, encoder):

        with torch.no_grad():
            mu, logsigma = encoder(self._X.detach())
            self._mean.data = mu
            self._logsigma.data = logsigma

    def sample(self, batch_size=1):
        if batch_size != 1:
            raise NotImplementedError
        else:
            eps = torch.randn_like(self._logsigma)
            return self._mean + torch.exp(self._logsigma) * eps

    def sample_batch_component(self, index, batch_size=1):

        if batch_size > 2048:
            raise RuntimeError('Batchsize will lead to memory issues')

        return self._mean[index, :] + torch.exp(self._logsigma[index, :]) * torch.randn(batch_size, self.dim, dtype=self.dtype,
                                                                              device=self.device)

    def encode(self, X, batch_size=1):

        if batch_size != 1:
            raise NotImplementedError

        if hash(X) != self._identifier:
            raise ValueError('This encoder is tied to a specific pre-defined set of values')

        return self.sample(batch_size)

    def KLD(self):
        return UnitGaussianKullbackLeiblerDivergence(self._mean, 2 * self._logsigma)

    def entropy(self, sample):
        const = self.N*0.5*(np.log(2*np.pi) + 1)
        return torch.sum(self._logsigma) + const



class EffectivePropertyMap(lamp.modules.BaseModule):

    def __init__(self, latent_dim, dim_effective_property, num_hidden_layers = 0, independent_X = True, *, dtype = None, device = None):

        super(EffectivePropertyMap, self).__init__()

        if num_hidden_layers == 0:
            self.fc = torch.nn.Linear(latent_dim, dim_effective_property)
        else:
            self.fc = FeedforwardNeuralNetwork.FromLinearDecay(latent_dim, dim_effective_property, num_hidden_layers, outf = None, dropout = None, dtype=dtype, device=device)

        if independent_X:
            self.logsigmas_X = torch.nn.Parameter(torch.ones(dim_effective_property, requires_grad = True))

        self._latent_dim = latent_dim
        self._independent_X = independent_X

        self._to(dtype=dtype, device=device)

    @property
    def independent_X(self):
        return self._independent_X

    def forward(self, z):

        if self._independent_X:
            return self.fc(z), self.logsigmas_X.expand(z.shape[0], -1)
        else:
            return self.fc(z)

    @property
    def dim_in(self):
        return self._latent_dim

    def forward_mean(self, z):
        return self.fc(z)

    def propagate_samples(self, z):

        if self._independent_X:

            means, logsigmas = self.forward(z)

            if means.shape != logsigmas.shape:
                raise RuntimeError('Implementation assumes that full logsigmas matrix is given; check for broadcasting')

            return means + torch.exp(logsigmas) * torch.randn_like(logsigmas)
        else:
            return self.forward(z)

    def extract_deterministic_map(self, *,  duplicate=True):

        if duplicate:
            return copy.deepcopy(self.fc)
        else:
            raise NotImplementedError



class ReducedOrderModelOperator(lamp.modules.BaseModule):

    def __init__(self, rom, W, *, dtype=None, device=None):

        super(ReducedOrderModelOperator, self).__init__()

        self.W = W
        self.rom = rom

        self._dtype = dtype
        self._device = device

        self.logsigmas_y = torch.nn.Parameter(torch.ones(W.shape[0], requires_grad=True))
        self._to(dtype=dtype, device=device)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def dim_effective_property(self):
        return self.rom.Vc_dim

    @property
    def dim_in(self):
        return self.dim_effective_property

    @property
    def dim_out(self):
        return self.W.shape[0]


    def forward(self, effprop, F):

        return torch.einsum('sk,nk->ns', [self.W, self.rom(torch.exp(effprop) + 1e-8, F)]), self.logsigmas_y.repeat(effprop.shape[0], 1)

    def forward_mean(self, effprop, F):

        return torch.einsum('sk,nk->ns', [self.W, self.rom(torch.exp(effprop) + 1e-8, F)])

    def propagate_samples(self, effprops, F):

        means, logsigmas = self.forward(effprops, F)

        if means.shape != logsigmas.shape:
            raise RuntimeError('Implementation assumes that full logsigmas matrix is given; check for broadcasting')

        return means + torch.exp(logsigmas) * torch.randn_like(logsigmas)


    @classmethod
    def FromPhysics(cls, physics,*, dtype=None, device=None):

        W = torch.tensor(physics['W'].T, dtype=dtype, device=device).t()

        if W.shape[0] < W.shape[1]:
            raise ValueError

        rom = ROM.FromPhysics(physics['rom'], dtype=dtype, device=device)
        return cls(rom, W, dtype=dtype, device=device)


class PredictionEnsemble(object):

    def __init__(self, model, dataset, scheduler_wrapper, lr=1e-2, writer = None):

        self._model = model
        self._dataset = dataset

        X = dataset.get('X')
        self._q_z = VariationalApproximation(model.dim_latent, X.shape[0], X)
        self._optimizer = torch.optim.Adam(self._q_z.parameters(), lr=lr)
        self._scheduler_wrapper = scheduler_wrapper
        self._scheduler_wrapper.register_optimizer(self._optimizer, 'validation')
        self.writer = writer

        self._elbo_history = list()

    def set_lr_manually(self, lr):
        raise NotImplementedError

    @property
    def q_z(self):
        return self._q_z

    @property
    def model(self):
        return self._model

    @property
    def dataset(self):
        return self._dataset

    def _elbo(self, X):

        Z = self._q_z.sample()
        predict_x = self._model.f(Z)
        logL = self._model.random_field_likelihood(predict_x, X)
        KLD = self._q_z.KLD()
        return logL, KLD

    def update(self, numIter=1, record=True, step = None):

        X = self._dataset.get('X')

        for n in range(numIter):

            logL, KLD = self._elbo(X.detach())
            elbo = logL - KLD

            J = -elbo
            self._optimizer.zero_grad()
            J.backward()
            self._optimizer.step()

            if n == numIter - 1:
                if record:
                    self.writer.add_scalar('PredictionEnsemble/elbo', elbo.item(), global_step=step)
                    self.writer.add_scalar('PredictionEnsemble/logL', logL.item(), global_step=step)
                    self.writer.add_scalar('PredictionEnsemble/KLD', KLD.item(), global_step=step)
                    self.writer.add_scalar('PredictionEnsemble/AvgLatentStddev',  torch.mean(torch.exp(self._q_z.logsigma)), global_step=step)
                self._scheduler_wrapper.step('validation', None, None, None, elbo)

        del J
        del elbo



    def __repr__(self):
        s = 'PredictionEnsemble | Wraps a dataset with {} points for validation purposes'.format(self._dataset.N)
        return s

class DataPair(object):

    def __init__(self, writer = None, label = '', name=None):

        if writer is not None and name is None:
            raise ValueError('Required to provide a name for the writer')

        self.iteration = list()
        self.value = list()
        self._writer = writer
        self._label = label
        self._name = name

    def append(self, iteration, value):

        self.iteration.append(iteration)
        self.value.append(value)

        if self._writer is not None:
            self._writer.add_scalar(self._label + '/' + self._name, value, global_step=iteration)

    def min(self):
        return min(self.value)

    def max(self):
        return max(self.value)

    def final(self):
        return self.value[-1]


class Analysis(object):


    def __init__(self, q, model, dataset, identifier = ''):

        # issue: identifier not set
        self._q = q
        self._model = model
        self._dataset = dataset
        self.description = None

        self.data = dict()
        items = ['relerr_x', 'relerr_y', 'logscore_x', 'logscore_y', 'r2_y']
        for item in items:
            self.data[item] = DataPair(writer=self._model.writer, label=dataset.label, name=item)


    @property
    def dataset(self):
        return self._dataset

    @classmethod
    def FromPredictionEnsemble(cls, pe):
        return cls(pe.q_z, pe.model, pe.dataset)

    @classmethod
    def FromEncoder(cls, model, dataset):

        Z_mean, Z_logsigma = model.encoder(dataset.get('X'))
        q = VariationalApproximation( Z_mean.shape[1], Z_mean.shape[0], dataset.get('X'), dtype = Z_mean.dtype, device=Z_mean.device, requires_grad=False )
        return cls(q, model, dataset)

    @property
    def X(self):
        return self._dataset.get('X')

    @property
    def Y(self):
        return self._dataset.get('Y')

    @property
    def F(self):
        return self._dataset.get('F_ROM_BC')

    @torch.no_grad()
    def sample_predictive_y(self, N_monte_carlo, index):

        Z_samples = self._q.sample_batch_component(index, batch_size=N_monte_carlo)
        X_samples = self._model.gp.propagate_samples(Z_samples)
        Y_samples = self._model.g.propagate_samples(X_samples, self.F[index,:].unsqueeze(0).expand(N_monte_carlo, self.F.shape[1]))

        return Y_samples

    @torch.no_grad()
    def sample_predictive_x(self, N_monte_carlo, index):

        Z_samples = self._q.sample_batch_component(index, batch_size=N_monte_carlo)
        return self._model.f.propagate_samples(Z_samples)

    def eval_all(self, N_monte_carlo, iteration):

        self.relative_error_x(N_monte_carlo, iteration)
        self.relative_error_y(N_monte_carlo, iteration)
        self.predictive_log_probability_x(N_monte_carlo, iteration)
        self.predictive_log_probability_y(N_monte_carlo, iteration)

    @torch.no_grad()
    def eval_all_y(self, N_monte_carlo, iteration = None, return_mean_std = False):

        dtype = self._q.dtype
        device = self._q.device
        y_mean = torch.zeros(self._q.N, self.Y.shape[1], dtype=dtype, device=device)
        y_std = torch.zeros(self._q.N, self.Y.shape[1], dtype=dtype, device=device)

        relerrs_y = torch.zeros(self._q.N, dtype=dtype, device=device)
        logscore_y = torch.zeros(self._q.N, dtype=dtype, device=device)

        for index in range(self._q.N):
            Y_samples = self.sample_predictive_y(N_monte_carlo, index)
            y_mean[index,:] = torch.mean(Y_samples,0)
            y_std[index,:] = torch.std(Y_samples,0)
            relerrs_y[index] = self._relative_error_indexed_y(N_monte_carlo=None, index=index, Y_samples=Y_samples)
            logscore_y[index] = self._predictive_log_probablity_y_indexed(N_monte_carlo=None, index=index, Y_samples=Y_samples)

        relerr_y = torch.mean(relerrs_y).item()
        r2_y = coefficient_of_determination(y_mean, self.Y, global_average=False)
        logscore_y = torch.mean(logscore_y).item()

        if iteration is None:
            if return_mean_std:
                raise RuntimeError('nope')
            return logscore_y, r2_y, relerr_y
        else:
            self.data['relerr_y'].append(iteration, relerr_y)
            self.data['logscore_y'].append(iteration, logscore_y)
            self.data['r2_y'].append(iteration, r2_y)
            if return_mean_std:
                return y_mean, y_std


    @torch.no_grad()
    def coefficent_of_determination_y(self, N_monte_carlo, index, Y_samples = None):
        raise NotImplementedError


    @torch.no_grad()
    def _coefficient_of_determiniation_y_indexed(self, N_monte_carlo, index, Y_samples = None):

        if Y_samples is None:
            Y_samples = self.sample_predictive_y(N_monte_carlo, index=index)

        y_mean = torch.mean(Y_samples,0).view(1,-1)
        return coefficient_of_determination(y_mean, self.Y[index,:].view(1,-1))

    @torch.no_grad()
    def relative_error_y(self, N_monte_carlo, iteration = None, ReturnValue = False):

        relerrs = np.zeros(self._q.N)

        for index in range(self._q.N):
            relerrs[index] = self._relative_error_indexed_y(N_monte_carlo, index)

        relerr = np.mean(relerrs)

        if iteration is None:
            return relerr
        else:
            self.data['relerr_y'].append(iteration, relerr)
            if ReturnValue:
                return relerr

    @torch.no_grad()
    def _relative_error_indexed_y(self, N_monte_carlo, index, Y_samples = None):

        if Y_samples is None:
            Y_samples = self.sample_predictive_y(N_monte_carlo, index=index)

        mean_y = torch.mean(Y_samples, 0)
        return relative_error(mean_y.flatten(), self.Y[index, :].flatten())

    @torch.no_grad()
    def relative_error_x(self, N_monte_carlo, iteration = None, ReturnValue = False):

        relerrs = np.zeros(self._q.N)

        for index in range(self._q.N):
            relerrs[index] = self._relative_error_indexed_x(N_monte_carlo, index)

        relerr = np.mean(relerrs)

        if iteration is None:
            return relerr
        else:
            self.data['relerr_x'].append(iteration, relerr)
            if ReturnValue:
                return relerr

    @torch.no_grad()
    def _relative_error_indexed_x(self, N_monte_carlo, index, X_samples = None):

        if X_samples is None:
            X_samples = self.sample_predictive_x(N_monte_carlo, index=index)
        mean_x = torch.mean(X_samples, 0)
        return relative_error(mean_x.flatten(), self.X[index,:].flatten())

    def _create_prediction_samples_y(self, index, N_monte_carlo):

        dtype, device = self.q_z._mean.dtype, self.q_z._mean.device
        Q = self.q_z._mean.shape[1]
        Z_samples = self.q_z._mean[index, :] + torch.exp(self.q_z._logsigma[index, :]) * torch.randn(N_monte_carlo, Q, dtype=dtype,
                                                                              device=device)

        X_samples = self._model.gp.propagate_samples(Z_samples)
        Y_samples = self._model.g.propagate_samples(X_samples)

        return Y_samples

    @torch.no_grad()
    def predictive_log_probability_y(self, N_monte_carlo, iteration = None):

        logp = np.zeros(self._q.N)

        for index in range(self._q.N):
            logp[index] = self._predictive_log_probablity_y_indexed(N_monte_carlo, index)

        logp = np.mean(logp)

        if iteration is None:
            return logp
        else:
            self.data['logscore_y'].append(iteration, logp)

    @torch.no_grad()
    def _predictive_log_probablity_y_indexed(self, N_monte_carlo, index, Y_samples = None):

        if Y_samples is None:
            Y_samples = self.sample_predictive_y(N_monte_carlo, index)

        Y_mean = torch.mean(Y_samples, 0).flatten()
        Y_std = torch.std(Y_samples, 0).flatten()

        logp = torch.mean(-torch.log(Y_std) - 0.5 * (
                (self.Y[index, :].flatten() - Y_mean) ** 2 / (Y_std ** 2)) - 0.5 * np.log(
            2 * np.pi)).item()  

        return logp


    @torch.no_grad()
    def predictive_log_probability_x(self, N_monte_carlo, iteration = None):

        logp = np.zeros(self._q.N)

        for index in range(self._q.N):
            X_pred = self.sample_predictive_x(N_monte_carlo, index)
            x_mean = torch.mean(X_pred, 0).flatten()
            x_std = torch.std(X_pred, 0).flatten()
            logp[index] = torch.mean(-torch.log(x_std) - 0.5 * (
                        (self.X[index, :].flatten() - x_mean) ** 2 / x_std ** 2) - 0.5 * np.log(
                2 * np.pi)).item()  

        logp = np.mean(logp)

        if iteration is None:
            return logp
        else:
            self.data['logscore_x'].append(iteration, logp)
