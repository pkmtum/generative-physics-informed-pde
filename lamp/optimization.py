import torch
import numpy as np


class LearningScheduleWrapper(object):

    def __init__(self, create_scheduler, disable = False):

        self._schedulers = dict()
        self._optimizers = dict()
        self._disable = disable
        self._counter = dict()

        self._create_scheduler = create_scheduler

    def lock(self):
        self._disable = True

    def unlock(self):
        self._disable = False

    @classmethod
    def stepLR(cls, step_size, factor=0.1):

        def f(optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma=factor, last_epoch = -1)

        return cls(f)

    @classmethod
    def ReduceLROnPlateau(cls, patience, threshold=1e-3, factor=0.1, min_lr=1e-3, verbose = True, mode='max'):

        assert factor < 1
        def f(optimizer):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode=mode, patience=patience,
                                                              threshold=threshold, factor=factor, min_lr=min_lr, verbose=verbose)

        return cls(f)

    @classmethod
    def MultiStepLR(cls, milestones, factor, last_epoch=-1):

        assert factor < 1
        def f(optimizer):
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=factor, last_epoch=last_epoch)

        return cls(f)

    @classmethod
    def Dummy(cls):

        def f(optimizer):
            # exciting!
            return None

        wrapper = cls(f)
        wrapper.lock()
        return wrapper

    def set_learning_rate_manually(self, id, lr):

        for param_group in self._optimizers[id].param_groups:
            param_group['lr'] = lr

    def register_optimizer(self, optimizer, id):

        self._schedulers[id] = self._create_scheduler(optimizer)
        self._optimizers[id] = optimizer
        self._counter[id] = 0

    def step(self, id, N=None, N_max=None, interval=1, metric=None):

        if interval is None:
            # legacy fix
            interval = 1

        if id not in self._optimizers:
            raise ValueError('LearningScheduleWrapper does not have "{}" optimizer registered'.format(id))

        self._counter[id] += 1

        if self._disable:
            return

        if np.mod(self._counter[id], interval) != 0:
            return

        if isinstance(self._schedulers[id], torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError('ReduceLROnPlateau requires metric')
            self._schedulers[id].step(metric)
        else:
            self._schedulers[id].step()


