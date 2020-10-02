import numpy as np
import torch
from lamp.optimization import LearningScheduleWrapper
from bottleneck.components import PredictionEnsemble, Analysis
from bottleneck.VirtualObservables import QuerryPointEnsemble, QuerryEnsemble, VirtualObservablesEnsemble, EnergyVirtualObservablesEnsemble
from torch.utils.tensorboard import SummaryWriter
from bottleneck.VirtualObservables import RadialBasisFunctionSampler
from physics.BoundaryConditions import BoundaryConditionEnsemble
import time
import os
from factories.model import ModelFactory
from factories.data import DataFactory
from utils.time import Timer
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from fawkes.Plotting import PlotFunction2D
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


class TrainerParameters(object):

    def __init__(self):

        self.data = dict()
        self.scheduler = dict()
        self.trainer = dict()
        self.optimizer = dict()
        self.margs = dict()
        self.dargs = dict()

        self._dtype = None
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._recover_dtype_device()
        return self._device

    @property
    def dtype(self):
        if self.dtype is None:
            self._recover_dtype_device()
        return self._dtype

    def _recover_dtype_device(self):

        try:
            mf = ModelFactory.FromIdentifier(self.identifier)
        except AttributeError:
            raise RuntimeError('TrainerParameters cannot provide dtype or device, since identifier has not yet been set')
        self._dtype, self._device = mf.dtype, mf.device




class Trainer(object):

    def __init__(self, mf, df, folder=None, comment='', debug = False):

        self._mf = mf
        self._df = df

        self._folder = folder
        self._dl = None
        self._dlu = None

        self.model = None
        self.physics = None
        self.encoder = None
        self.discriminative_model = None
        self.datasets = None

        self._optimizer = None
        self._scheduler_wrapper = None
        self._PE = None
        self._analyis = None

        self._dtpye = None
        self._device = None

        self._global_runtime = 0

        self._global_iteration_counter = 0

        physics, model, discriminative_model, encoder, dtype, device = self._mf.setup()
        model.writer = SummaryWriter(comment=comment)

        self.model = model
        self.encoder = encoder
        self.physics = physics
        self.discriminative_model = discriminative_model
        self._dtype = dtype
        self._device = device

        self._config = None

        self._monitor = dict()
        self._monitor['elbo'] = list()
        self._monitor['elbo_iter'] = list()
        self._monitor['lr'] = list()
        self._monitor['lr_iter'] = list()

        self.debug = debug

        self._armortized_bs = None

        self._vo_is_initialized = False
        self._finalized = False



        if self._folder is not None:
            if self._folder[-1] != '/':
                self._folder += '/'
            os.makedirs(self._folder, exist_ok=True)


    def info(self):

        if self.model.encoder is None:
            assert self._armortized_bs is None
        else:
            assert self._armortized_bs is not None

        n_unsupervised = 0
        try:
            n_unsupervised = self.model.datasets['unsupervised'].N
        except:
            pass
        n_vo = 0
        try:
            n_vo = self.model.datasets['vo'].N
        except:
            pass

        print("============ MODEL INFO ==============")
        print("N_unsupervised: {}".format(n_unsupervised))
        print("N_supervised: {}".format(self.model.datasets['supervised'].N))
        print("N_vo: {}".format(n_vo))
        armorization = self.model.encoder is not None
        print("Armortization: {} ".format(armorization))
        print("Device: {}".format(self.model.device))
        print("Dtype: {}".format(self.model.dtype))
        print("========================================")


    @property
    def mf(self):
        return self._mf

    @property
    def dl(self):
        return self._dl

    @property
    def dlu(self):
        return self._dlu

    def setup_config(self, **kwargs):

        # config can be (and is) overwritten later
        self._config = dict()
        self._config['lr_init'] = None
        self._config['normalize'] = False
        self._config['l2_penalty'] = None
        self._config['l1_penalty'] = None

        self._config['N_PE_updates'] = 3
        self._config['N_PE_updates_final'] = 100
        self._config['N_monte_carlo_analysis'] = 64
        self._config['N_monte_carlo_analysis_final'] = 128
        self._config['N_monitor_interval'] = 500
        self._config['N_tensorboard_logging_interval'] = 1

        self._config['N_vo_update_interval'] = 250
        self._config['N_vo_holdoff']= 100
        self._config['N_monte_carlo_vo'] = 128

        self._config['MonitorTraining'] = True

        for key, value in kwargs.items():
            if key not in self.config:
                raise KeyError('Could not set > {} < in local config in Trainer'.format(key))
            self._config[key] = value

    @property
    def config(self):

        if self.debug:

            config = self._config.copy()
            config['N_monitor_interval'] = 5
            config['N_PE_updates'] = 1
            config['N_PE_updates_final'] = 5
            config['N_monte_carlo_analysis'] =  8
            config['N_monte_carlo_analysis_final'] = 16
            config['N_monte_carlo_vo'] = 16
            config['N_tensorboard_logging_interval'] = 1

            return config
        else:
            return self._config

    def get(self, configkey):

        try:
            value = self.config[configkey]
        except KeyError:
            raise KeyError('Could not retrieve > {} < from local config in Trainer'.format(configkey))

        return value


    @classmethod
    def FromIdentifier(cls, identifier, margs = None, dargs = None, *args, **kwargs):

        mf = ModelFactory.FromIdentifier(identifier)

        if margs is not None:
            for key, val in margs.items():
                mf.set(key, val)

        # deprecated: this is actually not used and handled externally.
        df = None

        return cls(mf=mf, df=df, *args, **kwargs)

    @property
    def device(self):
        return self._device

    def reset(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return self._dtype

    @property
    def scheduler_wrapper(self):
        return self._scheduler_wrapper

    def setup(self, scheduler_wrapper, **kwargs):

        if self._config is None:
            raise RuntimeError('Config has not yet been setup')

        if scheduler_wrapper is None:
            scheduler_wrapper = LearningScheduleWrapper.Dummy()

        self._optimizer = torch.optim.Adam(params = self.model.parameters(), lr=self.get('lr_init'))

        self._scheduler_wrapper = scheduler_wrapper
        self._PE = PredictionEnsemble(self.model, self.datasets['validation'], self._scheduler_wrapper, self.get('lr_init'), writer=self.model.writer)
        self._analysis = Analysis.FromPredictionEnsemble(self._PE)
        self._scheduler_wrapper.register_optimizer(self._optimizer, 'training')
        self._analysis_training = Analysis(self.model.q_z['supervised'], self.model, self.model.datasets['supervised'])
        self.model.tensorboard_logging_interval = self.config['N_tensorboard_logging_interval']



    def tinfo(self, N = None):

        if self.gn == 0:
            return

        avg = self._global_runtime/self.gn
        print("{} iterations in {} seconds : that makes on average {} seconds per iteration".format(self.gn, self._global_runtime, avg))
        if avg is not None:
            print("Will require (approx) {} for {} iterations".format(avg*N, N))

    @property
    def gn(self):

        return self._global_iteration_counter

    def _step(self, *args, **kwargs):

        self._global_iteration_counter += 1


    def set_data_from_datasets(self, dl, dlu, datasets, Nu, Ns, Nvo, VO=None, vo_spec = None, armortized_bs = None):

        assert 'validation' in datasets and len(datasets['validation']) > 0
        assert Nu is not None
        assert Ns is not None
        assert Nvo is not None
        assert Nu >= 0
        assert Ns >= 0
        assert Nvo >= 0

        self._dl = dl
        self._dlu = dlu

        assert 'supervised' in datasets
        if Nu > 0:
            assert 'unsupervised' in datasets and datasets['unsupervised']
        if Nvo > 0:
            assert 'vo' in datasets and datasets['vo']


        if 'supervised' in datasets:
            datasets['supervised'].restrict(Ns)

        if 'vo' in datasets:
            datasets['vo'].restrict(Nvo)

        if 'unsupervised' in datasets:
            datasets['unsupervised'].restrict(Nu)

        if Nvo > 0:
            assert 'vo' in datasets
            if VO is None:
                assert isinstance(vo_spec, dict)


                QPE = QuerryPointEnsemble.FromDataSet(datasets['vo'], self.physics['fom'])

                if vo_spec['type'].lower() == 'energy':

                    assert vo_spec['l_rbf'] is not None
                    assert vo_spec['N_rbf'] is not None

                    sampler = RadialBasisFunctionSampler(qp=QPE[0], l=vo_spec['l_rbf'], N_aux=vo_spec['N_rbf'])

                    VO = EnergyVirtualObservablesEnsemble(QPE, vo_spec['energy_num_iterations_per_update'], sampler=sampler, dtype=self.dtype, device=self.device)
                    VO.set_temperature_schedule('exponential', T_init=vo_spec['T_init'], T_final=vo_spec['T_final'], num_steps=vo_spec['T_iterations'])

                elif vo_spec['type'].lower() == 'constrain':


                    QE = QuerryEnsemble.FromQuerryPointEnsemble(QPE, self.physics, vo_spec['CGR'], vo_spec['flux'],
                                                                vo_spec['N_gaussian'], vo_spec['N_rbf'],
                                                                vo_spec['l_rbf'], dtype=self.dtype, device=self.device)

                    VO = VirtualObservablesEnsemble(QPE, QE, dtype=self.dtype, device=self.device)
                else:
                    raise ValueError(
                        'Type: {} not known as specification.'.format(
                            vo_spec['type']))

            else:
                raise NotImplementedError('Cannot restrict a virtual observable ensemble')

        create_unsupervised_qZ = True
        if Nu is not None and Nu > 0:

            if armortized_bs is not None:
                if self._optimizer is not None:
                    raise RuntimeError('Optimizer has already been created without encoder')
                create_unsupervised_qZ = False
                self.model.encoder = self.encoder

            datasets['unsupervised'].restrict(Nu)

        self._armortized_bs = armortized_bs
        self.model.register_datasets(datasets, VO, create_unsupervised_variational_approximation=create_unsupervised_qZ)
        self.datasets = datasets


    def results(self, analysis = None):

        if analysis is None:
            analysis = self._analysis

        to_fetch = ['relerr_y', 'r2_y', 'logscore_y']
        results = dict()

        for fetch in to_fetch:
            results[fetch] = analysis.data[fetch].final()


        results['runtime'] = self._global_runtime

        return results

    def use_vo(self):
        return 'vo' in self.datasets and self.datasets['vo']

    def update_vo(self):

        if self.use_vo():
            update_vo = self.gn >= self.get('N_vo_holdoff') and (np.mod(self.gn, self.get('N_vo_update_interval')) == 0 or not self._vo_is_initialized) and self.datasets['vo']
        else:
            update_vo = False
        return update_vo



    def run(self, N, restart = False, verbose = True, uverbose = False, callback = None):

        if self._finalized:
            raise RuntimeError('Cannot run trainer which has already been finalized')

        if verbose:
            print("Starting Trainer - RUN")

        timer = Timer(N)
        t_start = time.time()
        for n in tqdm(range(N)):

            self._optimizer.zero_grad()

            if self.update_vo():
                self.model.update_virtual_observables(self.get('N_monte_carlo_vo'), return_mean_stddev=False, step=self.gn)
                self._vo_is_initialized = True

            elbo = self.model.elbo(step=self.gn, vo_holdoff = self.gn < self.get('N_vo_holdoff'), armortized_bs = self._armortized_bs, normalize=self.get('normalize'),
                                   l1_penalty = self.get('l1_penalty'), l2_penalty=self.get('l2_penalty'))

            J = -elbo
            J.backward()

            self._optimizer.step()

            self._PE.update(self.get('N_PE_updates'), step=self.gn)

            if np.mod(n, self.get('N_monitor_interval')) == 0 and n > 0:

                self.model.record(self.gn)
                self._monitor['elbo_iter'].append(self.gn)
                self._monitor['elbo'].append(elbo.item())
                self._monitor['lr'].append(self._optimizer.param_groups[0]['lr'])
                self._monitor['lr_iter'].append(self.gn)
                self._analysis.eval_all_y(self.get('N_monte_carlo_analysis'), self.gn)


                if self.get('MonitorTraining'):

                    self._analysis_training.eval_all_y(self.get('N_monte_carlo_analysis'), self.gn)

                    if self.model.encoder is not None:

                        analysis_encoder = Analysis.FromEncoder(self.model, self.datasets['validation'])
                        logscore_y, r2_y, relerr_y = analysis_encoder.eval_all_y(
                            self.get('N_monte_carlo_analysis_final'))
                        self.model.writer.add_scalar('validation_encoder/logscore_y', logscore_y, global_step=self.gn)
                        self.model.writer.add_scalar('validation_encoder/r2_y', r2_y, global_step = self.gn)
                        self.model.writer.add_scalar('validation_encoder/relerr_y', relerr_y, global_step=self.gn)


                if verbose:
                    print("Step: {} / {} || ELBO= {}  || LogScore(y): {}  || RRT: {}  ".format(n, N, elbo.item(), self._analysis.data['logscore_y'].final(),  timer.RRT(step=n)))
            elif uverbose:
                 print("Step: {} / {} || RRT: {} ".format(n, N, timer.RRT(step=n)))


            self._step(n, N)
            self._scheduler_wrapper.step('training', metric=elbo)

            if callback is not None:
                callback(n, self.gn)

        for nl in range(self.gn, self.gn + self.get('N_PE_updates_final')):
            self._PE.update(self.get('N_PE_updates'), step=nl)

        self._analysis.eval_all_y(self.get('N_monte_carlo_analysis_final'), self.gn + self.get('N_PE_updates_final'))

        self._global_runtime += time.time() - t_start

    def finalize(self):

        try:
            results = self.results()
            hpdict = {'dummy' : 0}
            self.model.writer.add_hparams(hparam_dict= hpdict, metric_dict =  results)
        except AttributeError:
            # not implemented for older pytorch versions, apparently
            pass

        try:
            self.model.writer.flush()
        except AttributeError:
            pass

        self.model.writer.close()
        self._finalized = True


    def plot_elbo(self, figsize):

        plt.figure(figsize = figsize)

        plt.plot(self._monitor['elbo_iter'], self._monitor['elbo'], '-o')
        plt.grid()
        plt.xlabel('Iterations')
        plt.ylabel('ELBO')
        plt.title('ELBO')

    def plot_predictive_logscore(self, figsize):

        plt.figure(figsize = figsize)

        plt.plot(self._analysis.data['logscore_y'].iteration, self._analysis.data['logscore_y'].value, '-o')
        plt.grid()
        plt.xlabel('# Iteration')
        plt.ylabel('Logscore')
        plt.title('Predictive Logscore (validation)')


def Plot2D(trainer , indeces = None):

    if indeces is not None:
        assert len(indeces) == 3
    else:
        indeces = [0,7,8]


    azim = 240
    elev = 0
    width = 10
    height = 16

    analysis = trainer._analysis
    physics = trainer.physics

    Y_val = trainer.datasets['validation'].get('Y')
    BCE_val = trainer.datasets['validation'].get('BCE')

    def modify_ax(ax):

        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.zaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.zaxis.set_major_locator(MultipleLocator(0.1))

    def shift_axes_horizontally(ax, val):
        pos1 = ax.get_position()
        pos2 = [pos1.x0 + val, pos1.y0, pos1.width, pos1.height]
        ax.set_position(pos2)

    fig = plt.figure(figsize=(width, height))
    gs1 = gridspec.GridSpec(4, 2)
    gs1.update(wspace=0.025, hspace=0.0)  # set the spacing between axes.

    pos = 0
    for i, ind in enumerate(indeces):

        y_true = Y_val[ind, :].detach().cpu().numpy().flatten()
        Y_sample = analysis.sample_predictive_y(1024, ind)
        y_mean = torch.mean(Y_sample, 0).detach().cpu().numpy().flatten()

        y_mean_f = physics['fom'].scatter_restricted_solution(y_mean, bc=BCE_val[ind], ReturnFunction=True)
        y_true_f = physics['fom'].scatter_restricted_solution(y_true, bc=BCE_val[ind], ReturnFunction=True)

        ax1 = plt.subplot(gs1[pos], projection='3d')
        plt.axis('on')
        PlotFunction2D(y_mean_f, ax=ax1, fig=fig)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.view_init(azim=azim, elev=elev)
        modify_ax(ax1)
        pos += 1
        plt.xlabel(r'$s_1$', labelpad=-12)
        plt.ylabel(r'$s_2$', labelpad=-12)

        if i == 0:
            plt.title('Mean Prediction')

        ax1 = plt.subplot(gs1[pos], projection='3d')
        plt.axis('on')
        PlotFunction2D(y_true_f, ax=ax1, fig=fig)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.view_init(azim=azim, elev=elev)
        pos += 1
        if i == 0:
            plt.title('Reference')
        plt.xlabel(r'$s_1$', labelpad=-12)
        plt.ylabel(r'$s_2$', labelpad=-12)
        modify_ax(ax1)
        shift_axes_horizontally(ax1, -0.08)


def CreateTrainer(params, dl, dlu):

    return CreateTrainerFromPermutation(params, permutation=torch.arange(dl.N), permutation_u=torch.arange(dlu.N))


def CreateTrainerFromPermutation(params, permutation = None, permutation_u = None, dl=None, dlu=None, datasets=None, BCE_encoding = None):


    trainer = Trainer.FromIdentifier(params.identifier, params.margs, params.dargs, folder=params.folder, comment = params.comment, debug=params.debug)

    if BCE_encoding is not None:
        BCE = BoundaryConditionEnsemble.FromEncoding(BCE_encoding, V_fom=trainer.physics['fom'].V, V_rom=trainer.physics['rom'].V, model_factory=trainer.physics['fom'].factory)
    else:
        BCE = None


    if dl is None or dlu is None or datasets is None:
        if not (dl is None and dlu is None and datasets is None):
            raise Exception('Either pass all required quantities, or none')
        dl, dlu, datasets = CreateDataSetsFromPermutation(params.identifier, permutation, permutation_u,
                                                          params.data['N_val'], params.data['N_u_max'], params.data['N_s_max'],
                                                          params.data['N_vo_max'], trainer.physics, BCE, trainer.dtype, trainer.device)

    scheduler_wrapper = LearningScheduleWrapper.MultiStepLR(params.scheduler['milestones'], factor=params.scheduler['factor'])

    trainer.set_data_from_datasets(dl, dlu, datasets, params.data['N_u'], params.data['N_s'], params.data['N_vo'], VO=None, vo_spec=params.data['vo_spec'],
                               armortized_bs=params.data['armortized_bs'])
    trainer.setup_config(**params.trainer)
    trainer.setup(scheduler_wrapper=scheduler_wrapper)

    assert trainer.datasets['supervised'].get('X').shape[0] == params.data['N_s']
    assert trainer.datasets['supervised'].get('Y').shape[0] == params.data['N_s']

    if params.data['N_u'] > 0:
        if params.data['armortized_bs'] is not None:
            assert trainer.datasets['unsupervised'].get('X', params.data['armortized_bs']).shape[0] == params.data['armortized_bs']
            assert trainer.datasets['unsupervised'].get('X').shape[0] == params.data['N_u']
        else:
            assert trainer.datasets['unsupervised'].get('X').shape[0] == params.data['N_u']

    return trainer


def CreateDataSetsFromPermutation(identifier, permutation, permutation_u, N_val, N_u_max, N_s_max, N_vo_max, physics, BCE, dtype, device):

    df = DataFactory.FromIdentifier(identifier)
    dl, dlu = df.setup()
    dl.assemble(physics, BCE=BCE)

    assert len(dl) == len(permutation)
    assert len(dlu) == len(permutation_u)

    partition = dict()
    partition['supervised'] = N_s_max


    if N_vo_max > 0:
        partition['vo'] = N_vo_max

    partition['validation'] = N_val
    dl.randomized_partition(partition, identifier='default', ForceOverwrite=False, permutation=permutation)

    datasets = dl.construct_dataset_dictionary(identifier = 'default', dtype=dtype, device=device)

    if N_u_max > 0:
        partition_aux = dict()
        partition_aux['unsupervised'] = N_u_max
        dlu.randomized_partition(partition_aux, identifier='default', ForceOverwrite=False, permutation=permutation_u)
        datasets_aux = dlu.construct_dataset_dictionary(identifier = 'default', dtype=dtype, device=device)
        datasets['unsupervised'] = datasets_aux['unsupervised']

    return dl, dlu, datasets
