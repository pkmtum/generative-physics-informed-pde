from physics.RandomField import NormalRandomFieldSampler
from utils.data import DataLoader
from utils.strings import ensure_file_extension
import os

# ugly fix
DATAPATH = 'cdata/'

class DataFactory(object):

    def __init__(self, config = None):

        self.config  = config
        self._path = None
        self._forced_setup = False

    @property
    def path(self):
        return self._check_path(DATAPATH)

    def _check_path(self, path):
        if path[-1] != '/':
            raise ValueError('path must end with a backslash | path= {}'.format(path))
        return path

    @property
    def identifier(self):
        if self._identifier is not None:
            return self._identifier
        else:
            return type(self).__name__

    @classmethod
    def FromIdentifier(cls, identifier, *args, **kwargs):

        # note that this implies capitalization has to match
        classname = identifier
        try:
            factory_class = globals()[classname]
        except KeyError:
            raise KeyError('DataFactory cannot provide factory for specified identifier {}'.format(identifier))

        return factory_class(*args, **kwargs)

    @classmethod
    def FromRandomFieldSampler(cls, rfs, N, N_unsupervised):
        raise NotImplementedError

    def _create_dataloader(self, N, identifier, extension):

        file = self.path + identifier
        file = ensure_file_extension(file, extension)

        if os.path.exists(file) and not self._forced_setup:
            dataloader = DataLoader.FromFile(file)
        else:
            print("Could not find {} to load dataset (or forced); creating from sampler... ".format(file))
            dataloader = DataLoader.FromSampler(self._rfs, N)
            dataloader.save(file)

        return dataloader

    def _create_dataloaders(self, rfs, N, N_unsupervised, identifier):

        dataloader = self._create_dataloader(N, identifier, extension = '.pt')
        dataloader_unsupervised = self._create_dataloader(N_unsupervised, identifier, extension='.ptu')
        dataloader_unsupervised.lock_physics_assembly()

        return dataloader, dataloader_unsupervised


    def setup(self):
        return self._create_dataloaders(self._rfs, self._N, self._N_unsupervised, self.identifier)

    def force_setup(self):
        self._forced_setup = True
        return self.setup()


class highres(DataFactory):

    def __init__(self):

        super(highres, self).__init__()

        self._N = 2*1024
        self._N_unsupervised = 2048*10
        self._rfs = NormalRandomFieldSampler.FromImage(64, 64, 0.4, 0.80, 0.04, Truncation='adaptive')
        self._identifier = None

class highres32(DataFactory):

    def __init__(self):

        super(highres32, self).__init__()

        self._N = 1024
        self._N_unsupervised = 2048*10
        self._rfs = NormalRandomFieldSampler.FromImage(32, 32, 0.4, 0.80, 0.15, Truncation=None)
        self._identifier = None

