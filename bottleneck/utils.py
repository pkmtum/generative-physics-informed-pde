import numpy as np
import fenics as df
import torch

class DiscontinuousGalerkinPixelConverter(object):

    def __init__(self, V):

        self.V = V
        self.mesh = V.mesh()

        self._X = None
        self._Y = None

        self._dx = None
        self._dy = None

        self._Nx = None
        self._Ny = None

        #
        self.Interpolator = None
        self.ReverseInterpolator = None

    @property
    def px(self):

        if self._Nx is None:
            self._assemble()

        return self._Nx - 1

    @property
    def py(self):

        if self._Ny is None:
            self._assemble()

        return self._Ny - 1

    def _assemble(self):

        coordinates = np.array(np.zeros(self.mesh.num_vertices()), dtype=[('x', float), ('y', float)])
        for i, vertex in enumerate(df.vertices(self.mesh)):
            coordinates['x'][i] = vertex.x(0)
            coordinates['y'][i] = vertex.x(1)

        self._Ny = len(np.unique(coordinates['y']))
        self._Nx = len(np.unique(coordinates['x']))

        coordinates = np.sort(coordinates, order=['y', 'x'])

        X = coordinates['x'].reshape(self._Ny, self._Nx)
        Y = coordinates['y'].reshape(self._Ny, self._Nx)

        self._X = np.flipud(X)
        self._Y = np.flipud(Y)

        # check if uniform mesh
        T = np.diff(X, axis=1);
        self._dx = T[0, 0]
        assert (np.all(np.abs(T - self._dx) < 1e-12))
        T = np.diff(Y, axis=0);
        self._dy = T[0, 0]
        assert (np.all(np.abs(T - self._dy) < 1e-12))

        Interpolator = np.zeros(((self._Ny - 1) * (self._Nx - 1), self.V.dim()))

        for i, cell in enumerate(df.cells(self.mesh)):

            x = cell.midpoint().x()
            y = cell.midpoint().y()

            cx = int(x // self._dx)
            cy = int(y // self._dy)

            cy = (self._Ny - 2) - cy
            pixel_id = cy * (self._Ny - 1) + cx

            Interpolator[pixel_id, i] = 0.5


        ReverseInterpolator = np.zeros((self.V.dim(), (self._Ny - 1) * (self._Nx - 1)))

        for i, row in enumerate(Interpolator):
            ind = np.where(row)[0]
            ReverseInterpolator[ind[0], i] = 1
            ReverseInterpolator[ind[1], i] = 1

        self.Interpolator = Interpolator
        self.ReverseInterpolator = ReverseInterpolator

        #
        a, b = np.where(self.Interpolator != 0)
        self._DofToPixelPermutator = torch.tensor(b, dtype=torch.long)

        a, b = self.ReverseInterpolator.nonzero()
        self._PixelToDofPermutator = torch.tensor(b, dtype=torch.long)

    @property
    def Interpolator_Dofs_To_Image(self):
        pass

    @property
    def Interpolator_Image_To_Dofs(self):
        pass

    def FunctionToImage(self, x, reshape=False):

        y = self.Interpolator @ x
        if reshape:
            y = y.reshape((self._Ny - 1, self._Nx - 1))
        return y

    def FunctionToImageBatchedFast(self, X):

        if X.dim() == 1:
            X = X.view(1,-1)

        batch_size = X.shape[0]
        X_perm = X[:, self._DofToPixelPermutator]
        Images_flattened = 0.5*(X_perm[:,0::2] + X_perm[:,1::2])
        Images = Images_flattened.view(batch_size, self.py, self.px)

        return Images

    def ImageToFunctionBatchedFast(self, Images):

        batch_size = Images.shape[0]
        flattened_Images = Images.view(batch_size, -1)
        X = flattened_Images[:,self._PixelToDofPermutator]
        return X

    def ConstructPixelatorModule(self, device):

        class Pixelator(torch.nn.Module):

            def __init__(self, DofToPixelPermutator, Nx, Ny, dimV):

                super(Pixelator, self).__init__()
                self._DofToPixelPermutator = DofToPixelPermutator
                self._Nx = Nx
                self._Ny = Ny
                self._dimV = dimV

            @property
            def px(self):
                return self._Nx - 1

            @ property
            def py(self):
                return self._Ny - 1

            def forward(self, X):

                if X.dim() == 1:
                    # deal with batch_size 1
                    X = X.view(1,-1)

                batch_size = X.shape[0]
                X_perm = X[:, self._DofToPixelPermutator]
                Images_flattened = 0.5*(X_perm[:,0::2] + X_perm[:,1::2])
                Images = Images_flattened.view(batch_size, self.py, self.px)

                return Images

            def __repr__(self):
                return 'Transforms function space objects (batch_dim x V.dim()) into images of size  (batch_dim x py px)'

        return Pixelator(self._DofToPixelPermutator.to(device=device), self._Nx, self._Ny, self.V.dim()).to(device=device)


    def ConstructInversePixelatorModule(self, device):


        class InversePixelator(torch.nn.Module):

            def __init__(self, PixelToDofPermutator, Nx, Ny, dimV):

                super(InversePixelator, self).__init__()
                self._PixelToDofPermutator = PixelToDofPermutator
                self._Nx = Nx
                self._Ny = Ny
                self._dimV = dimV

            @property
            def px(self):
                return self._Nx - 1

            @ property
            def py(self):
                return self._Ny - 1

            def forward(self, Images):

                batch_size = Images.shape[0]
                flattened_Images = Images.view(batch_size, -1)
                X = flattened_Images[:, self._PixelToDofPermutator]
                return X

                return Images

            def __repr__(self):
                return 'Transforms (batch_dim x py px) images into (batch_dim x V.dim()) function space objects'

        return InversePixelator(self._PixelToDofPermutator.to(device=device), self._Nx, self._Ny, self.V.dim()).to(device=device)

    def ImageToFunction(self, y):

        if y.ndim == 2:
            y = y.flatten(order='C')  # default order: C style

        return self.ReverseInterpolator @ y


def reparametrize(mean, logsigma):

    std = torch.exp(logsigma)
    return mean + std * torch.randn_like(std)


def relative_error(y, y_true):

    return (torch.norm(y - y_true) / torch.norm(y_true)).item()


def relative_error_batched(Y_mean, Y_true):
    return torch.mean((torch.sqrt(torch.sum((Y_mean - Y_true) ** 2, 1)) / torch.sqrt(torch.sum(Y_true ** 2, 1)))).item()


def DiagonalGaussianLogLikelihood(target, mean, logvars, target_logvars=None, reduce=torch.sum):

    if target_logvars is None:
        sigma = logvars.mul(0.5).exp_()
        part1 = logvars
        part2 = ((target - mean) / sigma) ** 2
        log2pi = 1.8378770664093453
        L = -0.5 * (part1 + part2 + log2pi)
        if reduce is not None:
            L = reduce(L)
        return L
    else:
        raise DeprecationWarning


def UnitGaussianKullbackLeiblerDivergence(mean: torch.Tensor, logvars: torch.Tensor) -> torch.Tensor:

    return -0.5 * torch.sum(1 + logvars - mean.pow(2) - logvars.exp())

def OptimizeEffectiveProperties(dataset, g, num_iterations = 300, lr=1e-2, verbose  = True, y_preprocessor = None):

    Y = dataset.get('Y')

    if y_preprocessor is not None:
        Y = y_preprocessor(Y)

    F_ROM_BC = dataset.get('F_ROM_BC')
    logX_effprop = torch.zeros(Y.shape[0], g.dim_effective_property, dtype=Y.dtype, device=Y.device, requires_grad=True)

    optimizer = torch.optim.Adam([logX_effprop], lr=lr)
    loss = torch.nn.modules.loss.MSELoss()

    objective = list()
    relerr_list = list()
    for n in range(num_iterations):
        if np.mod(n, 100) == 0 and n > 0:
            with torch.no_grad():
                myrelerr = relative_error_batched(Y_predict, Y)
                relerr_list.append(myrelerr)
            if verbose:
                print("Iteration {} || RelErr : {}".format(n, myrelerr))
        Y_predict = g.forward_mean(logX_effprop, F_ROM_BC)
        if y_preprocessor is not None:
            Y_predict = y_preprocessor(Y_predict)

        J = loss(Y_predict, Y)
        optimizer.zero_grad()
        J.backward()
        objective.append(J.item())
        optimizer.step()

    return logX_effprop, Y_predict, objective, relerr_list


def ReducedOrderModelSolve(dataset, physics, W):

    assert isinstance(W, np.ndarray)
    assert W.shape[0] > W.shape[1]

    X_DG = dataset.get('X_DG')
    X = X_DG.detach().cpu().numpy()
    BCE = dataset.get('BCE')

    N = X.shape[0]

    Y_rom = np.zeros((N, physics.dim_out))

    for n in range(N):
        K, f = physics.assemble_system(np.exp(X[n,:]), BCE[n], only_free_dofs=True)
        K_rom = W.T @ K @ W
        f_rom = W.T @ f
        y_rom = np.linalg.solve(K_rom, f_rom)
        y_rom = W @ y_rom
        Y_rom[n,:] = y_rom

    Y = dataset.get('Y')
    dtype = Y.dtype
    device = Y.device
    Y_rom = torch.tensor(Y_rom, dtype=dtype, device=device)

    return Y_rom