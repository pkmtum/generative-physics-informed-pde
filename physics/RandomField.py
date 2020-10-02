import numpy as np
import dolfin as df
from dolfin import Cell
from dolfin import Function
from dolfin import plot
import logging
import warnings


logger = logging.getLogger("RandomField")


class NormalRandomFieldSampler(object):

    def __init__(self, V=None, mean=None, stddev=None, corrlength=None, *, X=None, Truncation=None, py=None, px=None):

        if mean is None or stddev is None or corrlength is None:
            raise Exception(
                'You need to supply mean, standard deviation and correlation length to random field constructor')

        if stddev <= 0:
            raise ValueError

        if corrlength <= 0:
            raise ValueError

        if V is not None and X is not None:
            raise ValueError

        # redundant ...
        if (
                        V is not None and (V.ufl_element().family() != 'Discontinuous Lagrange' or V.ufl_element().degree() != 0 or V.num_sub_spaces() != 0 or V.dim() != V.mesh().num_cells())):
            raise TypeError

        if X is not None and X.shape[1] > 3:
            raise Exception


        self._X = X

        self._V = V

        if self.dim_out > 8192:
            raise RuntimeError

        self._corrlength = corrlength

        self._mean = mean

        self._stddev = stddev

        self._L = None

        self._truncation = Truncation

        self._eigvals = None

        self._py = py
        self._px = px

    @classmethod
    def FromImage(cls, py, px, mean, stddev, corrlength, Truncation = None, ly=1, lx=1):

        pixelwidth_x = lx/px
        pixelwidth_y = ly/py

        x = np.linspace(0 + 0.5 * pixelwidth_x, lx - 0.5 * pixelwidth_x, px)
        y = np.linspace(0 + 0.5 * pixelwidth_x, ly - 0.5 * pixelwidth_y, py)
        X,Y = np.meshgrid(x,y)

        X = np.hstack([X.flatten().reshape(-1, 1), Y.flatten().reshape(-1, 1)])

        return cls(mean=mean, stddev=stddev, X=X, corrlength=corrlength, Truncation=Truncation, py=py, px=px)


    @property
    def dim_out(self):
        if self._V is not None:
            return self._V.dim()
        else:
            return self._X.shape[0]

    @property
    def dim_in(self):
        if self._L is None:
            self._assemble()
        return self._L.shape[1]

    @property
    def tdim(self):

        if self._X is None:
            self._assemble()

        return self._X.shape[1]


    def _sample(self, gamma=None):

        if self._L is None:
            self._assemble()

        if gamma is None:
            gamma = np.random.normal(0, 1, self.dim_in)

        sample = self._mean + self._L.dot(gamma)

        if self._py is not None and self._px is not None:
            sample = sample.reshape(self._py, self._px)
        return sample


    def sample(self, gamma=None, batch_size=None):

        if gamma is not None and batch_size is not None:
            raise ValueError

        if batch_size is None:
            return self._sample(gamma=gamma)
        else:

            if self._px is None and self._py is None:

                X = np.zeros((batch_size, self.dim_out))
                for n in range(batch_size):
                    X[n,:] = self._sample()
            else:
                X = np.zeros((batch_size, self._py, self._px))
                for n in range(batch_size):
                    X[n,:,:] = self._sample()

            return X


    def plot(self, theta=None, title = None):

        if theta is None:
            theta = np.random.normal(0, 1, self.dim_in)

        f = Function(self._V)
        sample = self.sample(theta)
        f.vector()[:] = sample

        if title is not None:
            plot(f, title=title)
        else:
            plot(f)

        return sample

    def subspace(self, frac=0.9999):

        if self._L is None:
            self._assemble()

        if self._L.shape[0] == self._L.shape[1]:
            raise Exception

        return self._L


    def _assemble(self):

        if self._V is not None:
            self._X = ExtractPoints(self._V)

        C = np.zeros((self.dim_out, self.dim_out))

        for i, row in enumerate(self._X):

            r_squared = np.sum(np.square((row - self._X)), 1)
            C[i][:] = (self._stddev**2) * np.exp((-0.5*r_squared) / (self._corrlength**2))

        C = C + 1e-12*np.eye(C.shape[0])

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.flip(eigvals, 0)
        eigvecs = np.fliplr(eigvecs)
        self._eigvals = eigvals.copy()

        if self._truncation is not None:

            truncation = self._truncation

            if isinstance(truncation, str):
                if truncation.lower() == 'adaptive':
                    truncation = 0.999

            if isinstance(truncation, float):
                assert 0.9 < truncation < 0.9999
                variance_explained = np.cumsum(self._eigvals)/np.sum(self._eigvals)
                truncation = np.argmax(variance_explained > 0.999)

            if isinstance(truncation, int):
                if truncation >= self.dim_out or truncation < 1:
                    raise ValueError


            eigvals_clipped = eigvals[0:truncation]
            eigvecs_clipped = eigvecs[:, 0:truncation]

            Gamma = np.diag(np.sqrt(eigvals_clipped))
            V = eigvecs_clipped
            self._L = V.dot(Gamma)

        else:

            # Simple Cholesky decomposition
            self._L = np.linalg.cholesky(C)

    @staticmethod
    def ConvertLogMeanStd(mean, std):

        if mean <= 0 or std <= 0:
            raise ValueError

        mu = np.log(mean) - 0.5 * np.log((std / mean) ** 2 + 1)
        sigma = np.sqrt(np.log((std / mean) ** 2 + 1))
        return mu, sigma

def ExtractPoints(structure):

    if isinstance(structure, df.Mesh):
        mesh = structure
        dim = mesh.num_cells()
        dofmap = None
    elif isinstance(structure, df.FunctionSpace):
        ufl = structure.ufl_element()
        if ufl.degree() != 0 or ufl.family() != 'Discontinuous Lagrange':
            raise ValueError
        mesh = structure.mesh()
        dim = structure.dim()
        dofmap = structure.dofmap()
    else:
        raise ValueError

    Cells = [Cell(mesh, i) for i in range(mesh.num_cells())]

    topological_dim = mesh.geometry().dim()

    X = np.zeros((dim, topological_dim))

    for cell in Cells:

        if dofmap is not None:
            index = dofmap.cell_dofs(cell.index())
        else:
            index = cell.index()

        X[index, :] = cell.midpoint().array()[0:topological_dim]

    return X



