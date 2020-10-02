import numpy as np
import torch
from utils.time import StopWatch
from lamp.data import CustomTensorDataset
from physics.BoundaryConditions import BoundaryConditionEnsemble
from bottleneck.utils import DiscontinuousGalerkinPixelConverter

class DataLoader(object):

    def __init__(self, X, X_DG = None,  Y = None, BCE = None, F_ROM_BC = None, hash = None):

        if X.dtype != torch.double:
            raise ValueError

        if BCE is not None:
            if len(BCE) != X.shape[0]:
                raise ValueError

        if X_DG is not None:
            assert X.shape[0] == X_DG.shape[0]

        if Y is not None:
            assert Y.shape[0] == X.shape[0]

        if F_ROM_BC is not None:
            assert F_ROM_BC.shape[0] == X.shape[0]

        if X.device != torch.device('cpu'):
            raise ValueError

        self._X = X
        self._BCE = BCE
        self._X_DG = X_DG
        self._Y =  Y
        self._F_ROM_BC = F_ROM_BC

        self._permutation = dict()
        self._assigned_chunks = dict()
        self._state_indicator = dict()

        self._dependent_datasets = list()

        self._hash = hash

        self._lock_physics_assembly = False

    def lock_physics_assembly(self):
        self._lock_physics_assembly = True

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash(self._X)
        return self._hash

    @property
    def N(self):
        return self._X.shape[0]

    def register_dataset(self, dataset):
        self._dependent_datasets.append(dataset)

    def __len__(self):
        return self._X.shape[0]

    def assemble_BCE(self, physics):

        self._BCE = BoundaryConditionEnsemble.FromFactory(physics['fom'].factory, self.N)
        self._BCE.register_function_space('rom', physics['rom'].V)
        self._BCE.register_function_space('fom', physics['fom'].V)

    def assemble(self, physics, BCE = None):

        if self._lock_physics_assembly:
            raise RuntimeError

        if self._X.dim() != 3:
            raise ValueError


        if self._BCE is None:
            if BCE is not None:
                assert isinstance(BCE, BoundaryConditionEnsemble)
                assert BCE.check_if_registered('fom')
                assert BCE.check_if_registered('rom')
                self._BCE = BCE
            else:
                self.assemble_BCE(physics)

        self._Y = torch.zeros(len(self), physics['fom'].dim_out, dtype=torch.double, device=torch.device('cpu'))

        stopper = StopWatch(start=True)

        self.assemble_DG(physics)

        for n in range(len(self)):

            matprop = np.exp(self._X_DG[n, :].detach().numpy().flatten())
            self._Y[n,:] = torch.tensor(physics['fom'].solve(x=matprop, bc=self._BCE[n]), dtype=torch.double, device=torch.device('cpu'))

        self._F_ROM_BC = torch.tensor(self._BCE.FULL_F_WITH_APPLIED_BC('rom'), dtype=torch.double, device=torch.device('cpu'))

        stopper.stop()

    def assemble_DG(self, physics):

        Vc = physics['fom'].Vc

        assert Vc.dim() == 2*np.prod(np.array(self.X.shape)[1:]) == Vc.mesh().num_cells()

        self._X_DG = torch.zeros(len(self), Vc.dim(), dtype=torch.double, device=torch.device('cpu'))

        converter = DiscontinuousGalerkinPixelConverter(Vc)
        converter._assemble()

        for batch_ids in torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(len(self))),
                                                      batch_size=128,
                                                      drop_last=False):
            self._X_DG[batch_ids, :] = converter.ImageToFunctionBatchedFast(self._X[batch_ids, :])

    @property
    def X(self):
        return self._X

    @property
    def X_DG(self):
        if self._X_DG is None:
            raise RuntimeError('Assembly has not yet been called on this dataset')
        return self._X_DG

    @property
    def Y(self):
        if self._Y is None:
            raise RuntimeError('Assembly has not yet been called on this dataset')
        return self._Y

    @property
    def F_ROM_BC(self):
        if self._F_ROM_BC is None:
            raise RuntimeError('Assembly has not yet been called on this dataset')
        return self._F_ROM_BC

    @property
    def BCE(self):
        return self._BCE

    def reset_partition(self, identifier = None):

        if identifier is not None:
            del self._permutation[identifier]
            del self._assigned_chunks[identifier]
            del self._state_indicator[identifier]
        else:
            self._permutation = dict()
            self._assigned_chunks = dict()
            self._state_indicator = dict()

        for dataset in self._dependent_datasets:
            dataset.trigger_update()

    def construct_unsupervised_pytorch_dataloader(self, N_data, randomized_data, batch_size, randomized_batches, dtype, device):

        if not randomized_data:
            X = self.X[0:N_data].to(dtype=dtype, device=device)
        else:
            X = self.X[torch.randperm(self.N)][0:N_data].to(dtype=dtype, device=device)


        torch_dataset = CustomTensorDataset(X)
        pt_dataloader = torch.utils.data.dataloader.DataLoader(torch_dataset, batch_size=batch_size, shuffle=randomized_batches)

        return pt_dataloader

    @torch.no_grad()
    def ascending_partition(self, chunks, identifier = 'default', ForceOverwrite = False):
        permutation = torch.arange(self.N, dtype=torch.long)
        return self.randomized_partition(chunks, identifier=identifier, ForceOverwrite=ForceOverwrite, permutation=permutation)

    @torch.no_grad()
    def randomized_partition(self, chunks , identifier = 'default', *, ForceOverwrite = False, permutation = None):

        if identifier in self._permutation.keys() and not ForceOverwrite:
            raise RuntimeError

        if not chunks:
            raise ValueError

        N_total = sum(chunks.values())

        if N_total > self.N:
            raise ValueError

        if permutation is None:
            permutation = torch.randperm(self.N)
        elif isinstance(permutation, torch.Tensor) and permutation.dtype == torch.long:
            pass
        elif isinstance(permutation, np.ndarray):
            permutation = torch.tensor(permutation, dtype=torch.long)
        else:
            raise Exception
        self._permutation[identifier] = permutation
        self._assigned_chunks[identifier] = dict()
        self._state_indicator[identifier] = 0

        ptr = self._state_indicator[identifier]
        for chunk_label, chunk_size in chunks.items():
            self._assigned_chunks[identifier][chunk_label] = [torch.arange(ptr, ptr+chunk_size, dtype=torch.long)]
            ptr += chunk_size

        self._state_indicator[identifier] = ptr
        self._check_chunks(identifier)

    def _check_chunks(self, identifier):

        ids = torch.cat([torch.cat(subchunks) for subchunks in self._assigned_chunks[identifier].values()])
        unique_ids, counts = torch.unique(ids, sorted=True, return_counts=True)

        assert torch.all(counts == 1)
        assert unique_ids.min() == 0
        assert unique_ids.max() < self.N


    @torch.no_grad()
    def grow_partition(self, chunks_growth, identifier='default', SpecifyIncremental=True):

        if identifier not in self._assigned_chunks.keys():
            raise ValueError('The identifier {} is unknown'.format(identifier))

        for key in chunks_growth.keys():
            if key not in self._assigned_chunks[identifier].keys():
                raise ValueError('The chunk label {} is unknown'.format(key))

        if not chunks_growth:
            raise ValueError('Was supplied an empty dictionary')

        if not SpecifyIncremental:
            for chunk_label, chunk_size in chunks_growth.items():
                N_used = sum([a.numel() for a in self._assigned_chunks[identifier][chunk_label]])
                if N_used >= chunks_growth[chunk_label]:
                    raise ValueError
                chunks_growth[chunk_label] -= N_used

        N_available = self.N - self._state_indicator[identifier]
        N_requested = sum([chunk_size for chunk_size in chunks_growth.values()])

        if N_requested > N_available:
            raise ValueError

        ptr = self._state_indicator[identifier]
        for chunk_label, chunk_size in chunks_growth.items():
            self._assigned_chunks[identifier][chunk_label] +=  [torch.arange(ptr, ptr+chunk_size, dtype=torch.long)]
            ptr += chunk_size

        self._state_indicator[identifier] = ptr
        self._check_chunks(identifier)

        for dataset in self._dependent_datasets:
            dataset.trigger_update()


    def construct_dataset_dictionary(self, *, identifier=None, dtype, device):

        if identifier is None:
            if not self._permutation:
                raise RuntimeError

            datasets = dict()
            for identifier in self._permutation.keys():
                datasets[identifier] = dict()
                for label in self._assigned_chunks[identifier].keys():
                    datasets[identifier][label] = DataSet(self, label=label, identifier=identifier, dtype=dtype, device=device)
        else:
            if not identifier in self._permutation:
                raise KeyError

            datasets = dict()
            for label in self._assigned_chunks[identifier].keys():
                datasets[label] = DataSet(self, label=label, identifier=identifier, dtype=dtype, device=device)

        return datasets



    def save(self, path):


        if len(path.split('.')) == 1:
            raise ValueError

        torch.save({'X' : self.X, 'hash' : self.hash}, path)

    @classmethod
    def FromFile(cls, path, cutoff = None):

        if cutoff is not None:
            raise DeprecationWarning

        if len(path.split('.')) == 1:
            raise ValueError

        state = torch.load(path, map_location=torch.device('cpu'))

        X = state['X']
        hash = state['hash']

        if cutoff is not None:
            X = X[:cutoff,]


        return cls(X=X, hash=hash)

    @classmethod
    def FromSampler(cls, sampler, N):

        sh = sampler.sample().shape
        py = sh[0]
        px = sh[1]

        X = torch.zeros(N, py, px, dtype=torch.double, device=torch.device('cpu'))

        for n in range(N):

            X[n,:] = torch.tensor(sampler.sample(), dtype=torch.double, device=torch.device('cpu'))

        return cls(X=X)

    def __repr__(self):

        s = 'DataLoader with {} random field realizations ({},{}) [Assembled = {}]'.format(
            self._X.shape[0], self._X.shape[1], self._X.shape[2], self._X_DG is not None)
        return s


class DataSet(object):


    def __init__(self, dataloader, label, identifier='default', *, dtype, device):

        self._dataloader = dataloader
        self.identifier = identifier
        self.label = label

        self._dataloader.register_dataset(self)

        self._cached_indeces = None
        self._cache = dict()

        self._dtype = dtype
        self._device = device

        self._N_target = None

    @property
    def indeces(self):

        if self._cached_indeces is None:
            subset_lin = torch.cat(self._dataloader._assigned_chunks[self.identifier][self.label])
            self._cached_indeces = self._dataloader._permutation[self.identifier][subset_lin].tolist()

        return self._cached_indeces

    def __len__(self):

        if self._N_target is None:
            return len(self.indeces)
        else:
            return self._N_target

    def scramble(self):

        self._dataloader._permutation[self.identifier] = torch.arange(self._dataloader.N)
        self.trigger_update()


    def grow_in_size(self, N, incremental=False):

        if incremental:
            N_add = N
        else:
            if N <= len(self):
                raise ValueError
            N_add = N - self.N

        chunks_growth = {self.label : N_add}
        self._dataloader.grow_partition(chunks_growth, identifier=self.identifier, SpecifyIncremental=True)
        self.trigger_update()


    @property
    def N(self):
        return len(self)

    def restrict(self, N_target):

        assert isinstance(N_target, int)

        if N_target > self.N_max:
            raise ValueError

        if N_target == self._N_target:
            return

        if N_target == self.N_max:
            self._N_target = None
        else:
            self._N_target = N_target

        self.trigger_update()

    @property
    def N_max(self):
        return len(self.indeces)

    def trigger_update(self):
        #
        self._cached_indeces = None
        self._cache = dict()

    def get(self, idf, random_subset = None):

        validkeys = {'X', 'X_DG', 'Y', 'F_ROM_BC', 'BCE'}

        if not idf in validkeys:
            raise ValueError

        if not idf in self._cache:

            if self.N > 0:
                Q = getattr(self._dataloader, idf)[self.indeces]
                if idf in {'X', 'Y', 'F_ROM_BC'}:
                    Q = Q.to(dtype=self._dtype, device=self._device)

                if self._N_target is not None:
                    Q = Q[0:self._N_target]

            else:
                Q = None

            self._cache[idf] = Q

        if random_subset is None:
            return self._cache[idf]
        else:
            perm = torch.randperm(self.N, dtype=torch.long, device=self._device)
            return self._cache[idf][perm[0:random_subset],]


    def __repr__(self):
        s = 'Virtual dataset with {} datapoints | {} | {}'.format(self.N, self.label, self.identifier)
        return s









