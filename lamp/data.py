import torch
import torch.utils.data

class BatchedOverSampler(torch.utils.data.Sampler):

    def __init__(self, batch_size, num_batches, num_data):
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._num_data = num_data

        self._weights = torch.ones(self._num_data) / self._num_data

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for n in range(self._num_batches):
            yield torch.multinomial(self._weights, num_samples=self._batch_size, replacement=True)


class CustomTensorDataset(torch.utils.data.Dataset):


    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):

        T = tuple(tensor[index] for tensor in self.tensors)

        if len(T) == 1:
            T = T[0]
        return T

    def __len__(self):
        return self.tensors[0].size(0)