import torch
import numpy as np
from inspect import isfunction

def coefficient_of_determination(y_pred, y, global_average = False):

    assert y_pred.shape == y.shape

    if y_pred.dim() > 2:
        batch_size = y_pred.shape[0]
        y_pred = y_pred.view(batch_size, -1)
        y = y.view(batch_size, -1)

    if global_average:
        e = torch.sum((y - y_pred) ** 2) / torch.sum((y - y.mean()) ** 2)
        return 1 - e.item()
    else:
        assert y_pred.shape[0] > 0
        e = torch.sum((y - y_pred) ** 2, 0) / torch.sum((y - y.mean(0)) ** 2, 0)
        return (1 - e).mean().item()


def get_default_dtype():

    return torch.float32

def get_default_device():

    return torch.device('cpu')


def sparse_matrix_batched_vector_multiplication(matrix, vector_batch):

    # code from https://github.com/pytorch/pytorch/issues/14489
    batch_size = vector_batch.shape[0]
    vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)
    return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1)


def architecture_from_linear_decay(dim_in, dim_out, num_hidden_layers, append_input_output_dim_to_architecture = False):

    architecture = list(np.linspace(dim_in, dim_out, num_hidden_layers + 2).astype(int))

    if append_input_output_dim_to_architecture:
        architecture = [dim_in] + architecture + [dim_out]

    return architecture

def get_device(device):


    if isinstance(device, torch.device):
        return device

    if device.lower() == 'cpu':
        return torch.device('cpu')
    elif device.lower() in ['gpu', 'cuda']:
        return torch.device('cuda:0')
    else:
        raise ValueError('device not recognized in get_device()')


def get_dtype(dtype):

    # takes string and returns torch.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        if dtype.lower() == 'float32':
            return torch.float32
        elif dtype.lower() in ['float64', 'double']:
            return torch.float64
        else:
            raise ValueError('Supplied invalid string as dtype. Must be either float32, float64 or double')
    else:
        raise ValueError('Supplied invalid and unknown dtype argument.')

def get_activation_function(activ, module=True, force_string = False):

    if force_string and not isinstance(activ, str):
        raise ValueError('An actual activation function instead of an ID string was passed, with force_string set to true')

    if isinstance(activ, torch.nn.Module):
        return activ
    elif isfunction(activ):
        return activ
    elif isinstance(activ, str):
        if activ.lower() == 'relu':
            if module:
                return torch.nn.ReLU()
            else:
                return torch.nn.functional.ReLU
        else:
            raise ValueError('Supplied invalid string for activation function (does not match any known preset)')
    else:
        raise ValueError('Supplied invalid and unknwon argument for activation function')
