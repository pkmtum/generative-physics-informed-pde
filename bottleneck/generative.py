import torch
import numpy as np
import copy
import time
import lamp
from bottleneck.components import VariationalApproximation
from bottleneck.utils import DiagonalGaussianLogLikelihood, UnitGaussianKullbackLeiblerDivergence, reparametrize, relative_error_batched


class GenerativeModel(lamp.modules.BaseModule):

    def __init__(self, f, g, gp, writer = None, *,  dtype, device,):

        super(GenerativeModel, self).__init__()

        self.writer = writer

        self.f = f
        self.g = g
        self.gp = gp
        self.encoder = None

        self.q_z = torch.nn.ModuleDict()
        self.q_X = torch.nn.ModuleDict()

        self._dtype = dtype
        self._device = device

        self._datasets = dict()
        self.VO = None

        self.disable_elbo_vo = False
        self.disable_elbo_supervised = False
        self.disable_elbo_unsupervised = False

        self.manual_logging = False
        self.manual_log = dict()
        self.manual_log['elbo_supervised'] = list()
        self.manual_log['elbo_unsupervised'] = list()
        self.manual_log['elbo_vo'] = list()

        self._independent_X = gp.independent_X

        self.config = dict()
        self.config['reconstruct_log_eff_property'] = True

        self._tensorboard_logging_interval = 1

        self.preprocess_y_fct = None

    def _preprocess_y(self, y):

        if self.preprocess_y_fct is None:
            return y
        else:
            return self.preprocess_y_fct(y)

    def set(self, ckey, value):
        if ckey not in self.config:
            raise KeyError('{} is not a valid property for the generative model'.format(ckey))
        self.config[ckey] = value

    @property
    def tensorboard_logging_interval(self):
        return self._tensorboard_logging_interval

    @tensorboard_logging_interval.setter
    def tensorboard_logging_interval(self, value):
        assert value > 0
        assert isinstance(value, int)
        self._tensorboard_logging_interval = value

    def _tensor_board_logging_enabled(self, step):
        return np.mod(step, self.tensorboard_logging_interval) == 0

    def _add_scalar(self, *args, **kwargs):
        if self._tensor_board_logging_enabled(kwargs['global_step']):
            self.writer.add_scalar(*args, global_step=kwargs['global_step'])

    @property
    def datasets(self):
        return self._datasets

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def dim_effective_property(self):
        return self.g.dim_effective_property

    @property
    def dim_y(self):
        return self.g.dim_out

    def register_encoder(self, encoder):
        self.encoder = encoder

    @property
    def dim_latent(self):
        return self.f.dim_latent

    @property
    def N_s(self):
        raise NotImplementedError


    @torch.no_grad()
    def record(self, step):

        if self._independent_X:
            self.writer.add_scalar('Monitoring/logEffProp_sup_mean', torch.mean(self.q_X['supervised'].mean), global_step=step)
            self.writer.add_scalar('Monitoring/logEffProp_sup_sigma', torch.mean(self.q_X['supervised'].logsigma), global_step=step)

        self.writer.add_scalar('Monitoring/S_avg_precisions', torch.mean(1/(torch.exp(self.g.logsigmas_y)**2)), global_step=step)

    def init_by_encoder(self, encoder):
        for qz in self.q_z.values():
            qz.init_by_encoder(encoder)

    def set_encoder_decoder_states(self, results):

        if self.encoder is not None:
                self.encoder.load(results.encoder_state)
        else:
            raise Exception('The encoder is not set')

        self.f.load(results.decoder_state)

    def get_encoder_decoder_states(self):

        if self.encoder is None:
            raise RuntimeError('Encoder is not set - cannot return encoder state')

        return self.encoder.state(), self.f.state()


    def register_supervised_data(self, dataset):

        X = dataset.get('X')

        self.q_z['supervised'] = VariationalApproximation(self.dim_latent, X.shape[0],  X=X)

        if self._independent_X:
            self.q_X['supervised'] = VariationalApproximation(self.dim_effective_property, X.shape[0], X=X)

        self._datasets['supervised'] = dataset

    def register_unsupervised_data(self, dataset, create_variational_approximation = True):

        if create_variational_approximation:
            X = dataset.get('X')
            self.q_z['unsupervised'] = VariationalApproximation(self.dim_latent, X.shape[0], X=X)

        self._datasets['unsupervised'] = dataset

    def register_virtual_observables(self, dataset, VO):

        X = dataset.get('X')
        self.q_z['vo'] = VariationalApproximation(self.dim_latent, X.shape[0], X=X)
        if self._independent_X:
            self.q_X['vo'] = VariationalApproximation(self.dim_effective_property, X.shape[0], X=X)
        self.VO = VO
        self._datasets['vo'] = dataset

    def register_datasets(self, datasets, VO=None, create_unsupervised_variational_approximation = True):

        if 'supervised' in datasets and datasets['supervised']:
            self.register_supervised_data(datasets['supervised'])
        if 'unsupervised' in datasets and datasets['unsupervised']:
            self.register_unsupervised_data(datasets['unsupervised'], create_variational_approximation=create_unsupervised_variational_approximation)
        if 'vo' in datasets and datasets['vo']:
            if VO is None:
                raise ValueError('If datasets contains vo, need to pass virtual ensemble')
            self.register_virtual_observables(datasets['vo'], VO)


    @torch.no_grad()
    def update_virtual_observables(self, N_monte_carlo, Y_mean=None, Y_std=None, return_mean_stddev = False, step = None, Resample = True):

        if step is None:
            raise ValueError('We now require a step parameter to be passed to update VOs')

        t1 = time.time()
        if Y_mean is None or Y_std is None:

            N = self._datasets['vo'].N
            d = self.dim_y

            Y_mean = torch.zeros(N, d, dtype=self.dtype, device=self.device)
            Y_std = torch.zeros(N, d, dtype=self.dtype, device=self.device)
            F = self._datasets['vo'].get('F_ROM_BC')

            for n in range(N):

                if self._independent_X:
                    Y_sub_samples = self.g.propagate_samples(self.q_X['vo'].sample_batch_component(n, batch_size=N_monte_carlo), F[n,:].expand(N_monte_carlo,-1))
                else:
                    Y_sub_samples = self.g.propagate_samples(self.gp.propagate_samples(self.q_z['vo'].sample_batch_component(n, batch_size=N_monte_carlo)),
                                                             F[n,:].expand(N_monte_carlo,-1))

                Y_mean[n,:] = torch.mean(Y_sub_samples,0)
                Y_std[n,:] = torch.std(Y_sub_samples, 0)

        if Resample:
            self.VO.resample()


        self.VO.update(Y_mean, 1/(Y_std**2), step, writer=self.writer)

        t2 = time.time()

        if step is not None and self.writer is not None:
            self.writer.add_scalar('vo/q_y_mean_rel_err', relative_error_batched(self.VO.mean, self._datasets['vo'].get('Y').detach()), global_step=step)
            self.writer.add_scalar('vo/likelihood', torch.mean(DiagonalGaussianLogLikelihood(self.datasets['vo'].get('Y').detach(), self.VO.mean, 2*self.VO.logsigma)), global_step=step)

        if return_mean_stddev:
            return Y_mean, Y_std

    def alter_latent_dimension(self):
        raise NotImplementedError


    def forward(self, X, Y):
        raise NotImplementedError


    def random_field_likelihood(self, predict, target):


        if isinstance(predict, tuple):
            if self.config['reconstruct_log_eff_property']:
                return DiagonalGaussianLogLikelihood(target, predict[0], 2 * predict[1])
            else:
                return DiagonalGaussianLogLikelihood(torch.exp(target), torch.exp(predict[0]), 2*predict[1])
        else:
            target_new = torch.ones_like(target)
            target_new[target == target.min()] = 0
            bs = predict.shape[0]
            return -torch.nn.functional.binary_cross_entropy(predict.view(bs,-1), target_new.view(bs,-1), reduction='sum')


    def elbo(self, step, vo_holdoff = False, disable_vo = False, armortized_bs = None, normalize = False, l1_penalty = None, l2_penalty = None):

        myelbo = 0

        assert not (armortized_bs is not None and self.encoder is None)

        if 'unsupervised' in self._datasets and self._datasets['unsupervised']:
            if self.encoder is None:
                myelbo += self.elbo_unsupervised(self._datasets['unsupervised'].get('X').detach(), step=step, normalize=normalize)
            else:
                if armortized_bs is None:
                    raise ValueError('If armortized learning is used, we need to provide a batch size')

                myelbo += self.elbo_unsupervised_armortized(self._datasets['unsupervised'].get('X', random_subset = armortized_bs).detach(), step=step, normalize=normalize)

        if 'supervised' in self._datasets and self._datasets['supervised']:
            myelbo += self.elbo_supervised(self._datasets['supervised'].get('X').detach(), self._datasets['supervised'].get('Y').detach(), step=step, normalize=normalize)

        if 'vo' in self._datasets and self._datasets['vo'] and not disable_vo:
            myelbo += self.elbo_virtual_observables(self._datasets['vo'].get('X').detach(), holdoff=vo_holdoff, step=step, normalize=normalize)

        if l2_penalty is not None:

            pen = 0
            for param in self.f.parameters():
                pen += torch.norm(param)
            if self.encoder is not None:
                for param in self.encoder.parameters():
                    pen += torch.norm(param)
            myelbo -= l2_penalty*pen

            self._add_scalar('elbo_l2_penalty', pen, global_step=step)

        if l1_penalty is not None:
            raise NotImplementedError


        if self.writer is not None:
            self._add_scalar('elbo', myelbo, global_step=step)

        return myelbo


    def elbo_virtual_observables(self, X, step, holdoff = False, normalize = False):

        if self.disable_elbo_vo:
            return 0

        if self._independent_X:
            return self._elbo_virtual_observables_freeX(X, step, holdoff, normalize=normalize)
        else:
            return self._elbo_virtual_observables_lockX(X, step, holdoff, normalize=normalize)

    def _elbo_virtual_observables_lockX(self, X, step, holdoff = False, normalize = False):

        Z_sample = self.q_z['vo'].sample()
        DKL = self.q_z['vo'].KLD()

        predict_x = self.f(Z_sample)
        logL_x = self.random_field_likelihood(predict_x, X.detach())

        if not holdoff:
            X_sample = self.gp(Z_sample)
            mu_y, logsigmas_y = self.g(X_sample, self._datasets['vo'].get('F_ROM_BC'))
            y_sample = reparametrize(self.VO.mean, self.VO.logsigma)
            logL_y = DiagonalGaussianLogLikelihood(y_sample, mu_y, 2*logsigmas_y)
        else:
            logL_y = 0

        if normalize:
            bs = X.shape[0]
            logL_x /= bs
            logL_y /= bs
            DKL /= bs

        elbo = logL_x + logL_y - DKL


        if self.writer is not None:

            if not holdoff:
                self._add_scalar('objective/vo_logL_y', logL_y, global_step=step)
                self._add_scalar('objective/vo_DKL', DKL, global_step=step)

            self._add_scalar('objective/vo_logL_x', logL_x, global_step=step)
            self._add_scalar('objective/vo_elbo', elbo, global_step=step)


        if self.manual_logging:
            self.manual_log['elbo_vo'].append((step, elbo.item()))

        return elbo


    def _elbo_virtual_observables_freeX(self, X, step, holdoff = False, normalize = False):

        Z_sample = self.q_z['vo'].sample()
        DKL = self.q_z['vo'].KLD()

        predict_x = self.f(Z_sample)
        logL_x = self.random_field_likelihood(predict_x, X.detach())

        if not holdoff:

            X_sample = self.q_X['vo'].sample()
            mu_X, logsigmas_X = self.gp(Z_sample)
            logL_X = DiagonalGaussianLogLikelihood(X_sample, mu_X, 2 * logsigmas_X)

            mu_y, logsigmas_y = self.g(X_sample, self._datasets['vo'].get('F_ROM_BC'))
            y_sample = reparametrize(self.VO.mean, self.VO.logsigma)
            logL_y = DiagonalGaussianLogLikelihood(y_sample, mu_y, 2*logsigmas_y)

            entropy = self.q_X['vo'].entropy(X_sample)

        else:
            logL_X = 0
            logL_y = 0
            entropy = 0

        if normalize:
            bs = X.shape[0]
            logL_x /= bs
            logL_y /= bs
            logL_X /= bs
            entropy /= bs
            DKL /= bs

        elbo = logL_x + logL_y + logL_X + entropy - DKL



        if self.writer is not None:
            self._add_scalar('objective/vo_logL_y', logL_y, global_step=step)
            self._add_scalar('objective/vo_DKL', DKL, global_step=step)
            self._add_scalar('objective/vo_elbo', elbo, global_step=step)

            if not holdoff:
                self._add_scalar('objective/vo_logL_x', logL_x, global_step=step)
                self._add_scalar('objective/vo_logL_X', logL_X, global_step=step)
                self._add_scalar('objective/vo_entropy', entropy, global_step=step)


        if self.manual_logging:
            self.manual_log['elbo_vo'].append((step, elbo.item()))

        return elbo

    def extract_discriminative_model(self, *,  FromLatentEncoding, duplicate, encoder = None):

        if not duplicate:
            raise RuntimeError('We are only able to return duplicates (i.e. copies, but parameters are not reset')

        if FromLatentEncoding:
            encoder_ = None
            gp_det = copy.deepcopy(self.gp.extract_deterministic_map(duplicate=True))
            g = copy.deepcopy(self.g)
            return DiscriminativeModel(encoder=encoder_, gp = gp_det, g=g, dim_latent=self.dim_latent)
        else:
            if self.encoder is None and encoder is None:
                raise RuntimeError

            if encoder is not None:
                encoder_ = copy.deepcopy(encoder)
            else:
                encoder_ = copy.deecopy(self.encoder)

            gp_det = copy.deepcopy(self.gp.extract_deterministic_map(duplicate=True))
            g = copy.deepcopy(self.g)
            return DiscriminativeModel(encoder=encoder_, gp = gp_det, g=g, dim_latent=self.dim_latent)


    def elbo_supervised(self, X, Y, step, normalize = False):

        if self.disable_elbo_supervised:
            return 0

        if self._independent_X:
            return self._elbo_supervised_freeX(X,Y,step, normalize=normalize)
        else:
            return self._elbo_supervised_lockX(X,Y,step, normalize=normalize)


    def _elbo_supervised_lockX(self, X, Y, step, normalize = False):

        Z_sample = self.q_z['supervised'].sample()
        X_sample = self.gp(Z_sample)


        predict_x = self.f(Z_sample)
        logL_x = self.random_field_likelihood(predict_x, X.detach())

        mu_y, logsigmas_y = self.g(X_sample, self._datasets['supervised'].get('F_ROM_BC'))
        logL_y = DiagonalGaussianLogLikelihood(self._preprocess_y(Y.detach()), self._preprocess_y(mu_y), 2 * self._preprocess_y(logsigmas_y))

        DKL = self.q_z['supervised'].KLD()

        if normalize:
            bs = X.shape[0]
            logL_x /= bs
            logL_y /= bs
            DKL /= bs

        if self.writer is not None:
            self._add_scalar('objective/supervised_logL_x', logL_x, global_step=step)
            self._add_scalar('objective/supervised_logL_y', logL_y, global_step=step)
            self._add_scalar('objective/supervised_DKL_z', DKL, global_step=step)

        elbo = logL_x + logL_y - DKL

        if self.manual_logging:
            self.manual_log['elbo_supervised'].append((step, elbo.item()))

        return elbo

    def _elbo_supervised_freeX(self, X, Y, step, normalize = False):

        Z_sample = self.q_z['supervised'].sample()
        X_sample = self.q_X['supervised'].sample()

        predict_x = self.f(Z_sample)
        logL_x = self.random_field_likelihood(predict_x, X.detach())

        mu_X, logsigmas_X = self.gp(Z_sample)
        logL_X = DiagonalGaussianLogLikelihood(X_sample, mu_X, 2*logsigmas_X)

        mu_y, logsigmas_y = self.g(X_sample, self._datasets['supervised'].get('F_ROM_BC'))
        logL_y = DiagonalGaussianLogLikelihood(self._preprocess_y(Y.detach()), self._preprocess_y(mu_y), 2 * self._preprocess_y(logsigmas_y))


        DKL = self.q_z['supervised'].KLD()
        entropy = self.q_X['supervised'].entropy(X_sample)

        if normalize:
            bs = X.shape[0]
            logL_x /= bs
            logL_y /= bs
            logL_X /= bs
            entropy /= bs
            DKL /= bs

        elbo = logL_x + logL_y + logL_X + entropy - DKL

        if self.writer is not None:
            self._add_scalar('objective/supervised_logL_X', logL_X, global_step=step)
            self._add_scalar('objective/supervised_logL_x', logL_x, global_step=step)
            self._add_scalar('objective/supervised_logL_y', logL_y, global_step=step)
            self._add_scalar('objective/supervised_DKL_z', DKL, global_step=step)
            self._add_scalar('objective/supervised_entropy_X', entropy, global_step=step)
            self._add_scalar('objective/supervised_elbo', elbo, global_step=step)

        if self.manual_logging:
            self.manual_log['elbo_supervised'].append((step, elbo.item()))

        return elbo


    @property
    def has_registered_unsupervised(self):
        return not ('unsupervised' not in self.q_z or not self.q_z['unsupervised'])

    @property
    def has_registered_vo(self):
        return not('vo' not in self.q_z or not self.q_z['vo'])

    @property
    def has_registered_supervised(self):
        return not('supervised' not in self.q_z or not self.q_z['supervised'])

    def elbo_unsupervised(self, X, step, normalize = False):

        if self.disable_elbo_unsupervised:
            return 0

        Z_sample = self.q_z['unsupervised'].sample()

        predict_x = self.f(Z_sample)
        logL_x = self.random_field_likelihood(predict_x, X.detach())

        DKL = self.q_z['supervised'].KLD()

        if normalize:
            logL_x /= X.shape[0]
            DKL /= X.shape[0]

        elbo = logL_x - DKL


        if self.writer is not None:
            self._add_scalar('objective/unsupervised_logL_x', logL_x, global_step=step)
            self._add_scalar('objective/unsupervised_DKL_z', DKL, global_step=step)
            self._add_scalar('objective/unsupervised_elbo', elbo, global_step=step)


        if self.manual_logging:
            self.manual_log['elbo_unsupervised'].append((step, elbo.item()))


        return elbo

    def elbo_unsupervised_armortized(self, X, step, encoder = None, return_reconstruction = False, normalize = False):

        if self.disable_elbo_unsupervised:
            if return_reconstruction:
                raise ValueError('not supported')
            return 0

        if encoder is None and self.encoder is None:
            raise RuntimeError('Cannot use armortized inference, if encoder has not been registered with generative model')
        if 'unsupervised' in self.q_z:
            raise RuntimeError("Cannot use armortized inference, if q_z['unsupervised'] has been registered")

        mean, logsigma = self.encoder(X)

        Z = reparametrize(mean, logsigma)
        predict_x = self.f(Z)

        logL_x = self.random_field_likelihood(predict_x, X.detach())
        DKL = UnitGaussianKullbackLeiblerDivergence(mean, 2*logsigma) # takes logvars

        if normalize:
            batch_size = X.shape[0]
            logL_x /=  batch_size
            DKL /= batch_size

        elbo = logL_x - DKL


        if self.writer is not None:
            self._add_scalar('objective/ARM_unsupervised_logL_x', logL_x, global_step=step)
            self._add_scalar('objective/ARM_unsupervised_DKL_z', DKL, global_step=step)
            self._add_scalar('objective/ARM_unsupervised_elbo', elbo, global_step=step)

        if self.manual_logging:
            self.manual_log['elbo_unsupervised'].append((step, elbo.item()))

        if return_reconstruction:
            return elbo, predict_x
        else:
            return elbo


class DummyEffectivePropertyMap(lamp.modules.BaseModule):

    # for discriminative model
    def __init__(self, map):
        super(DummyEffectivePropertyMap, self).__init__()
        self._map = map

    def forward(self, z):
        return self._map(z)

    def propagate_samples(self, z):
        return self._map(z)

    def forward_mean(self, z):
        raise NotImplementedError


class DiscriminativeModel(lamp.modules.BaseModule):

    #
    def __init__(self, encoder, gp, g, dim_latent):

        super(DiscriminativeModel, self).__init__()

        if gp is None or g is None:
            # dirty fix
            raise ValueError

        self._encoder = encoder
        self._gp = DummyEffectivePropertyMap(gp)
        self._g = g
        self._dim_latent = dim_latent

    def copy(self):
        return copy.deepcopy(self)

    def curtail(self):
        self._encoder = None

    @property
    def dim_in(self):

        if self._encoder is not None:
            return self._encoder.dim_in
        else:
            return self._dim_latent

    @property
    def dim_out(self):
        return self._gp.dim_out

    def forward(self, x, F):

        if self._encoder is not None:
            x = self._encoder(x)

        return self._g(self._gp(x), F=F)