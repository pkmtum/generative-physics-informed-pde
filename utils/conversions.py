import torch

def Convert_ScipySparse_PyTorchSparse(A, dtype, device=None):

    A = A.tocoo()

    A_rows = torch.tensor(A.row, dtype=torch.long)
    A_cols = torch.tensor(A.col, dtype=torch.long)
    A_data = torch.tensor(A.data, dtype=dtype)

    indices = torch.cat((A_rows.unsqueeze(0), A_cols.unsqueeze(0)), 0)

    return torch.sparse_coo_tensor(indices, A_data, dtype=dtype, device=device)