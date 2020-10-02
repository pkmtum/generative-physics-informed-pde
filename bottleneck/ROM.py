import torch
import numpy as np
import fenics as df

class ROM(object):


    def __init__(self, physics, M, dtype, device):

        self.M = M
        self.dtype = dtype
        self.device = device

        self._bc_dofs = torch.tensor(physics.constrained_dofs.copy(), dtype=torch.long, device=device)
        self._free_dofs = torch.tensor(physics.free_dofs.copy(), dtype=torch.long, device=device)

    @property
    def physics(self):
        raise DeprecationWarning

    @property
    def V_dim(self):
        return self.M.shape[0]

    @property
    def Vc_dim(self):
        return self.M.shape[2]

    @property
    def dim_in(self):
        return self.Vc_dim

    @property
    def dim_out(self):
        return self.V_dim

    @classmethod
    def FromPhysics(cls, physics, dtype=torch.double, device=torch.device("cpu")):

        V = physics.V
        Vc = physics.Vc

        if V.mesh().num_cells() > 290:
            raise Exception('ROM exceeds intended maximum size')

        M = np.zeros((V.dim(), V.dim(), Vc.dim()))

        m = df.Function(Vc)
        for i in range(Vc.dim()):
            m.vector()[:] = 0
            m.vector()[i] = 1
            a2 = df.derivative(physics.a, physics.alpha, m)
            M[:, :, i] = df.assemble(a2).array().copy()

        M = torch.tensor(M, dtype=dtype, device=device)

        return cls(physics, M, dtype=dtype, device=device)

    def _solve_eqs(self, A, B):

        y_sol, _ = torch.solve(B,A)
        return y_sol


    def __call__(self, X, F=None, ReturnStiffness=False):

        # F is assumed to already have been modified correctly for the B.C.
        if X.dim() < 2:
            x = X.unsqueeze(0)

        if F.dim() < 3:
            F = F.unsqueeze(2)

        trunc = 1e-12
        if (X <= trunc).any().item():
            raise ValueError('At least one of the conductivity values supplied to the ROM was smaller than {}'.format(trunc))

        K = self.GetStiffness(X, DirichletBC=True)

        if F is None:
            raise DeprecationWarning
        else:
            y_rom =  self._solve_eqs(K.permute(2, 0, 1), F).squeeze(2)

        if ReturnStiffness:
            return y_rom, K
        else:
            return y_rom


    def GetStiffness(self, x, DirichletBC=True):

        K_batched = torch.matmul(self.M, x.t())

        if DirichletBC:
            # slow and clumsy
            K_batched[self._bc_dofs] = 0
            K_batched[self._bc_dofs, self._bc_dofs] = 1

        return K_batched


    def __repr__(self):
        return "This is the ROM \n Maps: {} -> {}".format(self.Vc_dim, self.V_dim)










