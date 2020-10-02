import numpy as np
import torch
from fawkes.Expressions import FastRadialBasisFunction
from bottleneck.flux import FluxConstraintReducedOrderModel
import dolfin as df


class QuerryPoint(object):

    def __init__(self, physics, x, bc):

        assert isinstance(x, np.ndarray)
        assert not isinstance(physics, dict)
        assert physics.Vc.dim() == x.size
        assert x.ndim == 1

        self._physics = physics
        self._x = x
        self._bc = bc

        self._K = None
        self._f = None

    @property
    def physics(self):
        return self._physics

    @property
    def bc(self):
        return self._bc

    @property
    def x(self):
        # beware of naming: this is log-transformed x
        return self._x

    @property
    def K(self):
        if self._K is None:
            self._assemble_system()
        return self._K

    @property
    def f(self):
        if self._f is None:
            self._assemble_system()
        return self._f

    @property
    def dim_in(self):
        return self._x.size

    @property
    def dim_out(self):
        return self._physics.dim_out

    def _assemble_system(self):
        # caching
        self._K, self._f = self._physics.assemble_system(np.exp(self._x), bc=self._bc, only_free_dofs = True)

    def construct_querry_weak_galerkin(self, V):

        assert V.shape[0] == self.K.shape[0]
        assert V.shape[0] == self.f.shape[0]

        Gamma = V.T @ self.K
        alpha = V.T @ self.f

        return Gamma, alpha


class QuerryPointEnsemble(object):

    def __init__(self, QPs):

        self._QPs = QPs

    def X(self, dtype, device):
        X = torch.zeros(len(self), self._QPs[0].dim_in, dtype=dtype, device=device)
        for n, qp in enumerate(self._QPs):
            X[n,:] = torch.tensor(qp.x, dtype=dtype, device=device)
        return X

    def __iter__(self):
        yield from self._QPs

    def __getitem__(self, item):
        return self._QPs[item]

    def __len__(self):
        return len(self._QPs)

    @property
    def dim_out(self):
        return self._QPs[0].dim_out

    @property
    def N(self):
        return len(self)

    @classmethod
    def FromDataSet(cls, dataset, physics):

        assert not isinstance(physics, dict)

        QPs = list()
        X_DG = dataset.get('X_DG')
        BCE = dataset.get('BCE')

        assert X_DG.dtype == torch.double

        for n in range(dataset.N):
            x = X_DG[n,:].detach().numpy().flatten()
            QPs.append(QuerryPoint(physics, x, BCE[n]))

        return cls(QPs)



class BaseSampler():

    def __init__(self, qp):

        self._qp = qp

    @property
    def m(self):
        raise NotImplementedError

    @property
    def qp(self):
        return self._qp

    @property
    def dim(self):
        return self.qp.dim_out

    @property
    def type(self):
        return self._type.lower()

    def sample_V(self):
        return self._sample()

    def sample(self):
        raise NotImplementedError

    @property
    def precision_mask(self):
        raise NotImplementedError

    @property
    def is_constant(self):
        raise NotImplementedError

    @property
    def fixed_precision(self):
        return np.all(self.precision_mask < 0)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

class ConstLengthScaleGenerator():

    def __init__(self, l):
        self._l = l

    def __call__(self):
        return self._l


class RadialBasisFunctionSampler(BaseSampler):

    def __init__(self, qp, l, N_aux):

        super().__init__(qp=qp)
        assert l is not None
        self._V = self._qp.physics.V
        self._free_dofs = self._qp.bc.free_dofs('fom')
        self._rbf, self._r0, self._l_handle = FastRadialBasisFunction(self._V.ufl_element())
        self._l_scale = l
        self._N = N_aux

    def _sample_rbf(self, only_free_dofs = True):

        a = np.random.uniform()
        b = np.random.uniform()
        r0_ = df.Constant(( a ,b ))

        self._r0.assign(r0_)
        self._l_handle.assign(self._l_scale)

        vec = df.interpolate(self._rbf, self._V).vector().get_local()

        if only_free_dofs:
            return vec[self._free_dofs]
        else:
            return vec

    @property
    def m(self):
        return self._N

    @property
    def is_constant(self):
        return False

    def illustrate(self):
        v = self._sample_rbf(only_free_dofs = False)
        f = df.Function(self._V)
        f.vector()[:] = v
        df.plot(f)

    @property
    def precision_mask(self):
        return -np.ones(self.m)

    def _sample(self):

        V = np.zeros((self._qp.dim_out, self._N))
        for n in range(self._N):
            V[:,n] = self._sample_rbf(only_free_dofs=True)
        return V


    def sample(self):
        V = self._sample()
        return self._qp.construct_querry_weak_galerkin(V)

class GaussianSketchingSampler(BaseSampler):

    def __init__(self, qp, N_aux):
        super().__init__(qp=qp)
        self._N = N_aux

    @property
    def m(self):
        return self._N

    @property
    def is_constant(self):
        return False

    @property
    def precision_mask(self):
        return -np.ones(self.m)

    def _sample(self):

        V = np.zeros((self._qp.dim_out, self._N))
        for n in range(self._N):
            V[:,n] = np.random.normal(0,1,(self._qp.dim_out))
        return V

    def sample(self):

        V = self._sample()
        return self._qp.construct_querry_weak_galerkin(V)

class ConcatenatedSamplers(BaseSampler):

    def __init__(self, samplers):

        super().__init__(qp=None)
        self._samplers = samplers

    @property
    def qp(self):

        return self.samplers[0].qp

    @property
    def m(self):

        return sum([v.m for v in self._samplers])

    @property
    def is_constant(self):

        return all((v.is_constant for v in self._samplers))

    @property
    def precision_mask(self):

        return np.concatenate([sampler.precision_mask for sampler in self._samplers])

    def _sample(self):

        return np.hstack([sampler.sample_V() for sampler in self._samplers])

    def sample(self):

        cache = [sampler() for sampler in self._samplers]
        return np.vstack([c[0] for c in cache]), np.concatenate([c[1] for c in cache])


class CoarseGrainedResidualSampler(BaseSampler):

    def __init__(self, qp, W):
        super().__init__(qp=qp)
        self._V = W
        self._Gamma_cgr, self._alpha_cgr = self._qp.construct_querry_weak_galerkin(W)

    @property
    def m(self):
        return self._alpha_cgr.size

    @property
    def is_constant(self):
        return True

    def _sample(self):
        return self._V

    @property
    def precision_mask(self):
        # infinite precision
        return -np.ones(self.m)

    def sample(self):
        return self._Gamma_cgr, self._alpha_cgr

class FluxConstrainSampler(BaseSampler):

    def __init__(self, qp, FluxConstrain):
        super().__init__(qp=qp)

        if not FluxConstrain.initialized:
            raise RuntimeError('Initialize flux-constrain first')

        self._Gamma_fc, self._alpha_fc = FluxConstrain.assemble_reduced(np.exp(qp.x), qp.bc)

    @property
    def m(self):
        return self._alpha_fc.size

    @property
    def is_constant(self):
        return True

    @property
    def precision_mask(self):
        return np.ones(self.m)

    def sample(self):
        return self._Gamma_fc, self._alpha_fc

    def _sample(self):
        raise NotImplementedError



class LinearQuerry(object):

    def __init__(self, querry_point, sampler, dtype, device):

        self._sampler = sampler
        self._querry_point = querry_point

        self._Gamma = None
        self._GammaTransposed = None
        self._alpha = None

        self._dtype = dtype
        self._device = device

        self._init()

    def _init(self):

        self.resample(ForceResample=True)


    @property
    def Gamma(self):
        return self._Gamma

    @property
    def GammaTransposed(self):
        return self._GammaTransposed

    @property
    def alpha(self):
        return self._alpha

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @Gamma.setter
    def Gamma(self, value):
        assert value.dtype == torch.double
        self._Gamma = value

    @GammaTransposed.setter
    def GammaTransposed(self, value):
        assert value.dtype == torch.double
        self._GammaTransposed = value

    @alpha.setter
    def alpha(self, value):
        assert value.dtype == torch.double
        self._alpha = value

    @property
    def m(self):
        # number of 'virtual observables'
        return self.Gamma.shape[0]

    def resample(self, ForceResample = False):


        if not self._sampler.is_constant or ForceResample:
            Gamma, alpha = self._sampler()
            self.Gamma = torch.tensor(Gamma, dtype=torch.double, device=self.device)
            self.alpha = torch.tensor(alpha, dtype=torch.double, device=self.device)
            self.GammaTransposed = self.Gamma.t()

    @property
    def dim_out(self):
        # i.e. dimension of y. different solution?
        return self.Gamma.shape[1]

    @property
    def precision_mask(self):
        return self._sampler.precision_mask

    def temporary_set_galerkin_manually(self, V):

        Gamma, alpha = self._querry_point.construct_querry_weak_galerkin(V)
        Gamma = torch.tensor(Gamma, dtype=torch.double, device=self.device)
        alpha = torch.tensor(alpha, dtype= torch.double, device=self.device)

        self.Gamma = Gamma
        self.GammaTransposed = Gamma.t()
        self.alpha = alpha
        self.precision = (-1)*torch.ones(self.m, dtype=torch.double, device=self.device)

    def add_galerkin_sampler(self, sampler):
        pass

    def add_flux_constraint(self):
        raise NotImplementedError



class QuerryEnsemble(object):

    def __init__(self, querries, dtype, device):

        self._querries = querries
        self._dtype = dtype
        self._device = device

    def __len__(self):
        return len(self._querries)

    @property
    def N(self):
        return len(self)

    @property
    def m(self):
        # total number of 'pieces of information'
        return sum([querry.m for querry in self._querries])

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def precision_mask(self):
        # assumes that they are identical
        return self._querries[0].precision_mask

    def resample(self, ForceResample = False):
        for q in self:
            q.resample(ForceResample=ForceResample)

    @property
    def dim_out(self):
        return self._querries[0].dim_out

    def __getitem__(self, item):
        return self._querries[item]

    def __iter__(self):
        yield from self._querries

    @classmethod
    def FromQuerryPointEnsemble(cls, QuerryPointEnsemble, physics, CGR, flux, N_gaussian, N_rbf, l_rbf = None, *, dtype=None, device=None):

        assert isinstance(physics, dict)
        W = physics['W']

        if W is None:
            raise NotImplementedError('need to provide W (as numpy array)')
        assert W.shape[0] > W.shape[1]
        assert isinstance(W, np.ndarray)

        assert dtype is not None
        assert device is not None

        querries = list()

        if flux:
            # assemble necessary quantities to derive test functions for flux
            fluxconstr =  FluxConstraintReducedOrderModel(physics)
            fluxconstr.create_measures()

        for qp in QuerryPointEnsemble:

            samplers = list()

            if CGR:
                samplers.append(CoarseGrainedResidualSampler(qp=qp, W=W))

            if flux:
                samplers.append(FluxConstrainSampler(qp, fluxconstr))

            if N_gaussian > 0:
               samplers.append(GaussianSketchingSampler(qp=qp, N_aux=N_gaussian))

            if N_rbf > 0:
                assert l_rbf is not None
                samplers.append(RadialBasisFunctionSampler(qp=qp, l = l_rbf, N_aux=N_rbf))

            if len(samplers) == 1:
                sampler = samplers[0]
            else:
                sampler = ConcatenatedSamplers(samplers)

            querries.append(LinearQuerry(qp, sampler, dtype=dtype, device=device))

        return cls(querries, dtype=dtype, device=device)




class BaseVirtualObservable(object):


    def __init__(self, querry_point, dtype, device):

        assert isinstance(querry_point, QuerryPoint)
        self._querry_point = querry_point
        self._dtype = dtype
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def querry_point(self):
        return self._querry_point

    @property
    def m(self):
        raise NotImplementedError

    @property
    def d_y(self):
        return self._querry_point.dim_out

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def vars(self):
        raise NotImplementedError

    def resample(self):
        raise NotImplementedError

    def update_precision(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError


class VirtualObservable(BaseVirtualObservable):


    def __init__(self, querry, querry_point, dtype, device):

        super().__init__(querry_point, dtype, device)

        assert isinstance(querry, LinearQuerry)

        self._querry = querry

        self._mean = None
        self._vars = None

        self._vo_variances = None


    @property
    def querry(self):
        return self._querry

    @property
    def mean(self):
        return self._mean

    @property
    def vars(self):
        return self._vars

    @property
    def m(self):
        return self._querry.m

    @property
    def vo_variances(self):
        return self._vo_variances

    @vo_variances.setter
    def vo_variances(self, value):
        assert value.dtype == torch.double
        assert value.device == self.device
        self._vo_variances = value

    def resample(self, ForceResample = False):
        self._querry.resample(ForceResample = ForceResample)

    @torch.no_grad()
    def update(self, g, prec, iteration, *, ForceUpdate = False):

        if not ForceUpdate:
            raise RuntimeError

        g = g.to(dtype=torch.double)
        prec = prec.to(dtype=torch.double)

        # dirty solution
        self._Gamma = self._querry.Gamma.t()
        self._GammaTransposed = self.querry.GammaTransposed.t()
        self._alpha = self.querry.alpha

        cov = 1/prec
        Lambda = torch.einsum('im, m, sm -> is', [self._GammaTransposed, cov, self._GammaTransposed])   # checked this; seems to be okay
        Lambda += torch.diag(self.vo_variances)
        L = torch.cholesky(Lambda)
        LambdaInv = torch.cholesky_inverse(L)

        solvec = LambdaInv @ (self._GammaTransposed @ g - self._alpha)
        mean = g - torch.einsum('i, mi, m -> i', [cov, self._GammaTransposed, solvec])

        A = self._GammaTransposed * cov
        postcov_diag_subtractor = torch.einsum('si, sm, mi -> i', [A, LambdaInv, A])

        self._mean = mean
        self._vars = cov - postcov_diag_subtractor


class EnergyVirtualObservable(BaseVirtualObservable):


    def __init__(self, querry_point, num_iterations_per_update, stochastic_subspace = None, sampler = None, l = 0.1, dtype = None, device = None):

        if dtype is None or device is None:
            raise ValueError('need to provide dtype and device')

        super().__init__(querry_point, dtype=dtype, device=device)

        self._stochastic_subspace = stochastic_subspace

        self._num_iterations_per_update = num_iterations_per_update

        if sampler is None:
            if stochastic_subspace is None:
                raise ValueError
            sampler = RadialBasisFunctionSampler(self._querry_point, l=l, N_aux=stochastic_subspace)

        self._sampler = sampler
        self._temperature = 1
        self._temperature_schedule = None

        self._mean = None
        self._vars = None

        self._mean_np = None
        self._vars_np = None

        self._K_diag_np = self._querry_point.K.diagonal()
        self._forced_temperature = None

    @property
    def temperature(self):
        if self._forced_temperature is None:
            return self._temperature
        else:
            return self._forced_temperature

    def force_temperature(self, value):
        self._forced_temperature = value

    @property
    def mean(self):
        return torch.tensor(self._mean_np, dtype=torch.double, device=self.device)

    @property
    def vars(self):
        return torch.tensor(self._vars_np, dtype=torch.double, device=self.device)

    @property
    def m(self):
        return 1

    def resample(self, ForceResample = False):
        # nothing to be done
        pass

    def set_temperature(self, temperature):

        assert temperature >= 0
        self._temperature = temperature

    def _init(self):

        if self._mean_np is None:
            self._mean_np = np.zeros(self._querry_point.dim_out)

    def set_temperature_schedule(self, type, T_init, T_final, num_steps):

        assert type.lower() in ['linear', 'exponential']

        if type.lower() == 'linear':
            self._temperature_schedule = LinearTemperatureSchedule(T_init, T_final, num_steps)
        elif type.lower() == 'exponential':
            self._temperature_schedule = ExponentialTemperatureSchedule(T_init, T_final, num_steps)
        else:
            raise Exception

    def set_linear_temperature_schedule(self, T_init = 1, T_final = 0.0001, num_steps = None):

        if num_steps is None:
            raise ValueError

        self._temperature_schedule = LinearTemperatureSchedule(T_init, T_final, num_steps)

    def update_precision(self, iteration):

        if self._forced_temperature is not None:
            return

        if self._temperature_schedule is None:
            raise RuntimeError

        self._temperature = self._temperature_schedule.get_temperature(iteration)


    @torch.no_grad()
    def update(self, g, prec, iteration,  *, ForceUpdate = False):

        if not ForceUpdate:
            raise RuntimeError

        inv_temperature = 1/self.temperature

        self._vars_np = 1/(prec.detach().cpu().numpy() + inv_temperature * self._K_diag_np)

        self._init()

        A = np.diag(prec.detach().cpu().numpy()) + inv_temperature * self._querry_point.K
        b = inv_temperature * self._querry_point.f + prec.detach().cpu().numpy() *g.detach().cpu().numpy()

        for n in range(self._num_iterations_per_update):

            V = self._sampler.sample_V()
            M = np.array(V.T @ A @ V)
            self._mean_np = self._mean_np - V @ np.linalg.solve(M, V.T @ np.array(A @ self._mean_np - b).flatten())

    def __repr__(self):

        s = 'Energy virtual Observable | Current temperature = {}'.format(self._temperature)
        return s


class BaseVirtualObservablesEnsemble(object):

    def __init__(self, QuerryPointEnsemble, virtual_observables, dtype, device):

        self._QuerryPointEnsemble = QuerryPointEnsemble
        self._dtype, self._device = dtype, device
        self._virtual_observables = virtual_observables

        m_target = self._virtual_observables[0].m
        for vo in virtual_observables:
            assert vo.m == m_target

        self._mean = None
        self._vars = None

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def X(self):
        return self._QuerryPointEnsemble.X

    def __getitem__(self, item):
        return self._virtual_observables[item]

    def __iter__(self):
        yield from self._virtual_observables

    def __len__(self):
        return len(self._virtual_observables)

    def flush_cache(self):

        self._vars = None
        self._mean = None


    @property
    def mean(self):

        if self._mean is None:

            mean = torch.zeros(self.N, self.dim_out, device=self.device, dtype=self.dtype)

            for n in range(self.N):
                val = self._virtual_observables[n].mean
                assert val.dtype == torch.double
                assert val.device == self.device
                mean[n,:] = val
            self._mean = mean

        return self._mean.detach()

    @property
    def vars(self):

        if self._vars is None:
            vars = torch.zeros(self.N, self.dim_out, device=self.device, dtype=self.dtype)

            for n in range(self.N):
                val = self._virtual_observables[n].vars
                assert val.dtype == torch.double
                assert val.device == self.device
                vars[n,:] = val

            self._vars = vars

        return self._vars.detach()

    @property
    def logsigma(self):
        return 0.5*torch.log(self.vars)


    @property
    def M(self):
        return sum((a.m for a in self))

    @property
    def m(self):
        return self._virtual_observables[0].m

    @property
    def dim_out(self):
        return self[0].d_y

    @property
    def N(self):
        return len(self)

    def update(self, G, PREC, iteration, writer = None):

        self.update_vo_precision(iteration, writer)

        for n, virtual_observable in enumerate(self._virtual_observables):
            virtual_observable.update(G[n,:], PREC[n,:], iteration, ForceUpdate = True)

        self.flush_cache()

    def update_vo_precision(self, iteration):
        raise NotImplementedError

    def resample(self, ForceResample = False):
        for vo in self._virtual_observables:
            vo.resample(ForceResample=ForceResample)


class VirtualObservablesEnsemble(BaseVirtualObservablesEnsemble):

    def __init__(self, QuerryPointEnsemble, QuerryEnsemble, dtype, device):

        virtual_observables = list()
        for querry, querry_point in zip(QuerryEnsemble, QuerryPointEnsemble):
            virtual_observables.append(VirtualObservable(querry, querry_point, dtype=dtype, device=device))

        super().__init__(QuerryPointEnsemble, virtual_observables, dtype=dtype, device=device)
        self._QuerryEnsemble = QuerryEnsemble

        self._alpha_0 = 1e-6
        self._beta_0 = 1e-6

        self._prec_alpha = 0.5*self.N + self._alpha_0
        self._prec_beta = torch.ones(self.m, dtype=torch.double, device=self.device)

        self._infinite_precision_mask = None
        self._learnable_precision_indeces = None
        self._constant_precision = None
        self._mean_vo_variances = None

        self._mean_vo_variances = self._get_mean_vo_variances()
        self._set_member_variance_values(self._mean_vo_variances)

        self._precision_initialized = False

    @property
    def m_free(self):

        raise NotImplementedError

    @property
    def fixed_precision(self):

        if self._constant_precision is None:
            self._constant_precision = np.all(self.infinite_precision_mask.detach().cpu().numpy())
        return self._constant_precision

    @property
    def learnable_precision_indeces(self):
        if self._learnable_precision_indeces is None:
            self._learnable_precision_indeces = torch.tensor(np.where(np.invert(self.infinite_precision_mask.detach().cpu().numpy()))[0], dtype=torch.double, device=self.device)

    @property
    def infinite_precision_mask(self):

        if self._infinite_precision_mask is None:
            self._infinite_precision_mask = torch.tensor(self._QuerryEnsemble[0].precision_mask < 0, dtype=torch.bool, device=self.device)

        return self._infinite_precision_mask

    def _get_mean_vo_variances(self):

        mean_vars = self._prec_beta / (self._prec_alpha + 1)
        mean_vars[self.infinite_precision_mask] = 0
        return mean_vars

    def _set_member_variance_values(self, mean_vo_vars):

        for vo in self:
            vo.vo_variances = mean_vo_vars

    @torch.no_grad()
    def update_vo_precision(self, iteration, writer = None):

        if not self._precision_initialized:
            self._precision_initialized = True
            return

        if self[0].mean is None or self[0].vars is None:
            raise RuntimeError

        if not self.fixed_precision:

            beta = torch.zeros(self.m, dtype=torch.double, device=self.device)

            for ii, vo in enumerate(self):
                Gamma = vo.querry.Gamma
                mean = vo.mean
                alpha = vo.querry.alpha
                vars = vo.vars
                beta = beta + (Gamma @ mean - alpha)**2 + (Gamma**2 @ vars)

            self._prec_beta = 0.5*beta + self._beta_0

            self._mean_vo_variances = self._get_mean_vo_variances()
            self._set_member_variance_values(self._mean_vo_variances)

            if writer is not None:
                writer.add_scalar('Monitor/Mean_VO_variances', torch.mean(self._mean_vo_variances), global_step=iteration)


class EnergyVirtualObservablesEnsemble(BaseVirtualObservablesEnsemble):


    def __init__(self, QuerryPointEnsemble, num_iterations_per_update, sampler, dtype, device):

        virtual_observables = list()

        for qp in QuerryPointEnsemble:
            virtual_observables.append(EnergyVirtualObservable(qp, num_iterations_per_update, sampler= sampler, dtype=dtype, device=device))


        super().__init__(QuerryPointEnsemble, virtual_observables, dtype=dtype, device=device)

    def force_temperature(self, value):
        for vo in self:
            vo.force_temperature(value)

    def set_temperature(self, *args, **kwargs):
        for vo in self:
            vo.set_temperature(*args, **kwargs)

    def set_temperature_schedule(self, type, **kwargs):
        for vo in self:
            vo.set_temperature_schedule(type, **kwargs)

    def set_linear_temperature_schedule(self, *args, **kwargs):

        for vo in self:
            vo.set_linear_temperature_schedule(*args, **kwargs)

    def update_vo_precision(self, iteration, writer = None):

        for vo in self._virtual_observables:
            vo.update_precision(iteration)

        if writer is not None:
            writer.add_scalar('Monitoring/Temperature', self._virtual_observables[0].temperature, global_step = iteration)


class TemperatureSchedule(object):

    def __init__(self):
        pass

    def get_temperature(self, iteration):
        raise NotImplementedError



class LinearTemperatureSchedule(TemperatureSchedule):

    def __init__(self, T_init, T_final, num_steps):

        super().__init__()

        assert num_steps > 1
        assert T_final < T_init
        self._T_init = T_init
        self._T_final = T_final
        self._num_steps = num_steps

    def get_temperature(self, iteration):

        if iteration > self._num_steps:
            raise RuntimeError

        frac = iteration / (self._num_steps-1)
        return self._T_init + frac*(self._T_final - self._T_init)


class ExponentialTemperatureSchedule(TemperatureSchedule):


    def __init__(self, T_init, T_final, num_steps):

        super().__init__()
        assert num_steps > 1
        assert T_final < T_init
        self._T_init = T_init
        self._T_final = T_final
        self._num_steps = num_steps
        self._lmbda = - np.log(T_final / T_init)

    def get_temperature(self, iteration):

        if iteration > self._num_steps:
            raise RuntimeError

        t = iteration / (self._num_steps - 1)

        return self._T_init * np.exp(-self._lmbda * t)


