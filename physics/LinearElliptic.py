import numpy as np
import dolfin as df
from scipy.sparse import linalg as slinalg
from physics.LinearEllipticFactories import GetFactory
from fawkes.utils import ConvertFenicsBackendToScipyCSRSparse


class LinearEllipticPhysics(object):

    def __init__(self, identifier, physics_id, mesh, V=None, Vc=None):

        self.identifier = identifier
        self.factory, _ = GetFactory(physics_id)
        self.a, self.set_alpha, self.alpha, self.V, self.Vc = self.factory(mesh, V, Vc)
        self.mesh = mesh
        self._form_compiler_parameters = {"optimize": True}

        # cache
        self._constrained_dofs = None
        self._free_dofs = None

    @property
    def dx(self):
        return self.factory.dx

    @property
    def ds(self):
        return self.factory.ds

    @property
    def free_dofs(self):
        if self._free_dofs is None:
            self._constrained_dofs, self._free_dofs = self.factory.constrained_and_free_dofs(self.V)
        return self._free_dofs

    @property
    def tdim(self):
        return self.mesh.topology().dim()

    @property
    def constrained_dofs(self):
        if self._constrained_dofs is None:
            self._constrained_dofs, self._free_dofs = self.factory.constrained_and_free_dofs(self.V)
        return self._constrained_dofs

    @property
    def neumann_boundary_dofs(self):

        if self._neumann_dofs is None:

            boundary = df.CompiledSubDomain("on_boundary")
            bc = df.DirichletBC(self.V, df.Constant(1), boundary)
            all_boundary_dofs = set(bc.get_boundary_values().keys())
            self._neumann_dofs = np.array(list(all_boundary_dofs - set(self.constrained_dofs)), dtype=int)

        return self._neumann_dofs

    @property
    def neumann_boundary_values(self):
        raise NotImplementedError

    @property
    def dim_in(self):
        return self.Vc.dim()

    @property
    def dim_out(self):
        return self.free_dofs.size

    @property
    def dim_out_all(self):
        return self.V.dim()

    def set_x(self, x):

        if np.any(x <= 0):
            raise ValueError('Trying to set negative or zero material values')

        self.set_alpha(x)

    def transfer(self, mesh):
        return LinearEllipticPhysics(self.id, mesh)


    def solve(self, x, bc, only_free_dofs=True, ReturnType='numpy'):

        self.set_x(x)
        u = df.Function(self.V)
        L = bc.neumann_boundary_condition.compile_form(self.V)
        df.solve(self.a == L, u, bc.dirichlet_boundary_condition.transfer(self.V), form_compiler_parameters=self._form_compiler_parameters)

        if ReturnType.lower() == 'fenics':
            return u
        elif ReturnType.lower() == 'numpy':
            u = u.vector().get_local()
            if only_free_dofs:
                return u[bc.free_dofs(self.identifier)]
            else:
                return u
        else:
            raise NotImplementedError

    def scatter_restricted_solution(self, y, bc, ReturnFunction = False):

        constrained_dofs = bc.constrained_dofs(identifier='fom')
        constrained_dofs_values = bc.constrained_dofs_values(identifier='fom')
        free_dofs = bc.free_dofs(identifier='fom')

        y_ = np.zeros(self.dim_out_all)
        y_[constrained_dofs] = constrained_dofs_values
        y_[free_dofs] = y

        if ReturnFunction:
            f = df.Function(self.V)
            f.vector()[:] = y_
            return f
        else:
            return y_

    def solve_direct(self, x, bc, only_free_dofs = True):

        self.set_x(x)

        K, f = self.assemble_system(x, bc)
        y_sub = slinalg.spsolve(K, f)

        if only_free_dofs:
            return y_sub
        else:
            y = np.zeros(self.V.dim())
            y[bc.constrained_dofs(self.identifier)] = bc.constrained_dofs_values(self.identifier)
            y[bc.free_dofs(self.identifier)] = y_sub
            return y



    def assemble_system(self, x, bc, *, only_free_dofs = True):

        if not only_free_dofs != 'scipy_csr':
            # legacy check
            raise TypeError

        self.set_x(x)
        K = df.assemble(self.a)
        f = bc.assemble_vanilla_force_vector(self.identifier)

        if not only_free_dofs:
            return ConvertFenicsBackendToScipyCSRSparse(K), f

        constrained_dofs = bc.constrained_dofs(self.identifier)
        free_dofs = bc.free_dofs(self.identifier)
        constrained_dofs_values = bc.constrained_dofs_values(self.identifier)

        Ks = ConvertFenicsBackendToScipyCSRSparse(K)
        K_coupling = Ks[free_dofs,:][:, constrained_dofs]
        f_effective = f[free_dofs] - K_coupling.dot(constrained_dofs_values)
        Ks = Ks[free_dofs][:, free_dofs]

        return Ks, f_effective

    def assemble_system_fenics(self, x, bc, *, only_free_dofs = True):

        self.set_x(x)
        L = bc.neumann_boundary_condition.compile_form()

        K, f = df.assemble_system(self.a, L, bcs = bc.dirichlet_boundary_condition.transfer(self.V))

        if not only_free_dofs:
            return ConvertFenicsBackendToScipyCSRSparse(K), f
        else:
            raise NotImplementedError


