from fawkes.utils import AssembleBasisFunctionMatrix
from dolfin import Point
from numpy import reshape
import numpy as np
from numpy import hstack
from numpy import meshgrid
import scipy
from numpy import zeros


class Probe(object):


    def __init__(self, f, points, shape=None, f_global=None):


        if not points.flags['C_CONTIGUOUS']:
            raise Exception('Require C-contiguous array')

        self._H = None
        self._points = points
        self._f = f
        self._f_global = f_global
        self._shape = shape

    @property
    def dim(self):
        return self.numpoints

    @property
    def numpoints(self):
        return self._points.shape[0]

    @property
    def spacedim(self):

        if self._f_global:
            return self._f_global.function_space().mesh().topology().dim()
        else:
            return self._f.function_space().mesh().topology().dim()

    @classmethod
    def FromLine(cls, f, x, y=None, z=None, *, f_global=None):

        # stack arrays
        p = np.column_stack([c for c in [x, y, z] if c is not None])

        if not p.flags['C_CONTIGUOUS']:
            raise Exception('Require C-contiguous array')

        # We do not need to set the shape?
        return cls(f, p, f_global=f_global)

    @classmethod
    def FromNodes(cls, f, *, f_global=None, mesh=None):

        if mesh is None:
            mesh = f.function_space().mesh()

        p = mesh.coordinates().copy()

        if not p.flags['C_CONTIGUOUS']:
            raise Exception('Require C-contiguous array')

        return cls(f, p, f_global=f_global)

    @classmethod
    def Grid_2D(cls, f, x, y, *, f_global=None):

        if x.ndim == 1 and y.ndim == 1:
            x, y = meshgrid(x, y)

        if not x.shape == y.shape:
            raise Exception('Grid faulty')

        x_flat = reshape(x, (x.size, 1))
        y_flat = reshape(y, (y.size, 1))

        p = hstack((x_flat, y_flat))

        return cls(f, p, shape=x.shape, f_global=f_global)


    def EvaluateBatch(self, X, mask=None,  Confidence=None):

        if self._H is None:
            self.Assemble()

        Y = np.zeros((self.numpoints, X.shape[1]))

        if mask is not None:
            if self._f_global is None:
                Full_x = self._f.vector().get_local()
            else:
                Full_x = self._f_global.vector().get_local()

        for i,column in enumerate(X.T):
            if mask is not None:
                Full_x[mask] = column
                Y[:,i] = self._H.dot(Full_x)
            else:
                Y[:,i] = self.Evaluate(x=column)

        meanY = np.mean(Y,1, dtype=np.float64)
        varY = np.var(Y,1, dtype=np.float64)

        if Confidence is not None:
            conf = scipy.stats.mstats.mquantiles(Y, prob=Confidence, axis=1)
            return meanY, varY, conf
        else:
            return meanY, varY


        return meanY, varY


    def Assemble(self):

        if self._f_global is None:
            self._H = AssembleBasisFunctionMatrix(self._f.function_space(), self._points, ReturnType='scipy')
        else:
            self._H = AssembleBasisFunctionMatrix(self._f.function_space(), self._points, ReturnType='scipy', Global_V = self._f_global.function_space())

    def Evaluate(self, format = 'LIST', x=None, VectorComponent=None, TensorComponent=None, TensorSymmetry=True, mask=None):

        if self.spacedim != 2:
            raise Exception('Assuming dim=2')

        if self._H is None:
            self.Assemble()

        if x is None:
            if self._f_global is None:
                x = self._f.vector().get_local()
            else:
                x = self._f_global.vector().get_local()

        if mask is not None:
            x = x[mask]

        y = self._H.dot(x)

        if VectorComponent is not None and TensorComponent is not None:
            raise Exception('Either vector or tensor')

        if VectorComponent is not None:
            y = y[VectorComponent::self.spacedim]
        elif TensorComponent is not None:
            if TensorSymmetry:
                y = y[TensorComponent::3]
            else:
                y = y[TensorComponent::4]

        if self._shape is not None and format == 'GRID':
            return reshape(y, self._shape)

        return y

    def EvaluateReference(self, VectorComponent=None, TensorComponent=None):

        y = zeros(self._points.shape[0])

        if VectorComponent is None and TensorComponent is None:
            for i, row in enumerate(self._points):
                y[i] = self._f(Point(row))
        elif VectorComponent is not None:
            for i, row in enumerate(self._points):
                y[i] = self._f(Point(row))[VectorComponent]
        elif TensorComponent is not None:
            for i, row in enumerate(self._points):
                y[i] = self._f(Point(row))[TensorComponent]

        return y

    @property
    def BasisFunctionMatrix(self):
        return self._H
