import numpy as np
import fenics as df
from fawkes.BoundaryConditions import DirichletBoundaryCondition, DirichletSpecification
from fawkes.BoundaryConditions import NeumannBoundaryCondition, NeumannSpecification
from physics.BoundaryConditions import GeneralBoundaryCondition


def GetFactory(id):

    # returns factory and topological dimension
    if id.lower() == 'ND':
        return LinearElliptic_2D_Factory_ND(), 2
    elif id.lower() == 'NDP'.lower():
        return LinearElliptic_2D_Factory_NDP(), 2
    else:
        raise NotImplementedError


def SetupUnitMeshHelper(mesh, V=None, Vc=None):

    if V is None:
        V = df.FunctionSpace(mesh,'CG',1)
    if Vc is None:
        Vc = df.FunctionSpace(mesh,'DG',0)

    boundaries = dict()
    boundaries['left'] = df.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
    boundaries['bottom'] = df.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
    boundaries['top'] = df.CompiledSubDomain("near(x[1], 1.0) && on_boundary")
    boundaries['right'] = df.CompiledSubDomain("near(x[0], 1.0) && on_boundary")

    boundarymarkers = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
    boundarymarkers.set_all(0)
    domainmarkers = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)


    boundaries['left'].mark(boundarymarkers,1)
    boundaries['bottom'].mark(boundarymarkers,2)
    boundaries['right'].mark(boundarymarkers, 3)
    boundaries['top'].mark(boundarymarkers,4)

    ds = df.Measure('ds', domain=mesh, subdomain_data=boundarymarkers)
    dx = df.Measure('dx', domain=mesh, subdomain_data=domainmarkers)

    return boundaries, boundarymarkers, domainmarkers, dx, ds, V, Vc


class LinearEllipticFactory(object):

    def __init__(self):

        self._mesh = None
        self.compliant_dirichlet_boundary = True
        self._constant_boundary_condition = None

    @property
    def constant_boundary_condition(self):
        if self._constant_boundary_condition is None:
            raise NotImplementedError
        return self._constant_boundary_condition

    @property
    def tdim(self):
        if self._mesh is None:
            raise RuntimeError
        return self._mesh.topology().dim()

    def sample_boundary_condition(self, dbc=None, nbc=None):

        if dbc is None:
            dbc = self.sample_dirichlet_boundary_condition()

        if nbc is None:
            nbc = self.sample_neumann_boundary_condition()

        gbc = GeneralBoundaryCondition(dirichlet_boundary_condition=dbc, neumann_boundary_condition=nbc, set_alpha=self._set_alpha)
        return gbc

    def sample_neumann_boundary_condition(self):
        raise NotImplementedError

    def sample_dirichlet_boundary_condition(self):
        raise NotImplementedError

    def constrained_and_free_dofs(self, V):
        if self.compliant_dirichlet_boundary:
            dbc = self.sample_dirichlet_boundary_condition()
            constrained_dofs, _, free_dofs = dbc.extract(V, ReturnFreeDofs = True)
            return constrained_dofs, free_dofs
        else:
            raise Exception


    def constrained_dofs(self, V):
        if self.compliant_dirichlet_boundary:
            dbc = self.sample_dirichlet_boundary_condition()
            constrained_dofs, _, _ = dbc.extract(V, ReturnFreeDofs = False)
            return constrained_dofs
        else:
            raise Exception

    def free_dofs(self, V):
        if self.compliant_dirichlet_boundary:
            dbc = self.sample_dirichlet_boundary_condition()
            _, _, free_dofs = dbc.extract(V, ReturnFreeDofs = True)
            return free_dofs
        else:
            raise Exception

    def reconstruct_dirichlet(self, DirichletEncoding):
        if self._constant_boundary_condition:
            return self.sample_dirichlet_boundary_condition()
        else:
            raise NotImplementedError

    def reconstruct_neumann(self, NeumannEncoding):
        if self._constant_boundary_condition:
            return self.sample_neumann_boundary_condition()
        else:
            raise NotImplementedError


class LinearElliptic_2D_Factory_ND(LinearEllipticFactory):

    def __init__(self):
        super(LinearElliptic_2D_Factory_ND, self).__init__()
        self._constant_boundary_condition = True

        self._v = None
        self._dx = None
        self._ds = None
        self._boundaries = None
        self._dx_markers = None
        self._ds_markers = None
        self._set_alpha = None
        self.compliant_boundaries = True


    def __call__(self, mesh, V=None, Vc=None):

        boundaries, boundarymarkers, domainmarkers, dx, ds, V, Vc = SetupUnitMeshHelper(mesh, V, Vc)

        self._ds_markers = boundarymarkers
        self._dx_markers = domainmarkers

        self._dx = dx
        self._ds = ds
        self._boundaries = boundaries
        self._subdomains = self._boundaries

        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        alpha = df.Function(Vc, name='conductivity')
        alpha.vector()[:] = 1

        def set_alpha(x):
            alpha.vector()[:] = x

        self._set_alpha = set_alpha
        a = alpha * df.inner(df.grad(u), df.grad(v)) * dx

        return a, set_alpha, alpha, V, Vc


    def sample_neumann_boundary_condition(self):

        c = 0
        fe = df.Constant(c)
        ns = NeumannSpecification('dx', expression=fe, subdomain=None)
        nbc = NeumannBoundaryCondition([ns])
        return nbc

    def sample_dirichlet_boundary_condition(self):

        u0 = df.Constant(0)
        u1 = df.Constant(1)
        bcs_specs = [DirichletSpecification(u1, self._boundaries['right']), DirichletSpecification(u0, self._boundaries['left'])]
        bcs_wrapper = DirichletBoundaryCondition(bcs_specs)
        return bcs_wrapper


class LinearElliptic_2D_Factory_NDP(LinearEllipticFactory):

    def __init__(self):

        super(LinearElliptic_2D_Factory_NDP, self).__init__()

        self._constant_boundary_condition = False
        self._v = None
        self._dx = None
        self._ds = None
        self._boundaries = None
        self._dx_markers = None
        self._ds_markers = None
        self._set_alpha = None
        self.compliant_boundaries = True


    def __call__(self, mesh, V=None, Vc=None):

        boundaries, boundarymarkers, domainmarkers, dx, ds, V, Vc = SetupUnitMeshHelper(mesh, V, Vc)
        self._ds_markers = boundarymarkers
        self._dx_markers = domainmarkers
        self._dx = dx
        self._ds = ds
        self._boundaries = boundaries
        self._subdomains = self._boundaries

        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        alpha = df.Function(Vc, name='conductivity')
        alpha.vector()[:] = 1

        def set_alpha(x):
            alpha.vector()[:] = x

        self._set_alpha = set_alpha

        a = alpha * df.inner(df.grad(u), df.grad(v)) * dx

        return a, set_alpha, alpha, V, Vc

    def sample_neumann_boundary_condition(self, nbe = None):

        c = 0
        fe = df.Constant(c)

        if nbe is not None:
            assert nbe.type == 'NPD'

        ns = NeumannSpecification('dx', expression=fe, subdomain=None)
        nbc = NeumannBoundaryCondition([ns], encoding_type = 'NDP', encoding_data = dict())
        return nbc

    def reconstruct_neumann(self, NeumannEncoding):

        return self.sample_neumann_boundary_condition()

    def sample_dirichlet_boundary_condition(self, dbe = None):

        low_left = -0.5
        high_left = 0.5
        low_right = -0.5
        high_right = 0.5

        if dbe is None:

            u0 = np.random.uniform(low=low_left,high=high_left)
            u1 = np.random.uniform(low=low_left,high=high_left)
            u2 = np.random.uniform(low=low_right,high=high_right)
            u3 = np.random.uniform(low=low_right,high=high_right)

            dbe_data = dict()
            dbe_data['u0'] = u0
            dbe_data['u1'] = u1
            dbe_data['u2'] = u2
            dbe_data['u3'] = u3

        else:
            #
            dbe_data = dbe.data
            assert dbe.type == 'NDP'

        u0 = df.Constant(dbe_data['u0'])
        u1 = df.Constant(dbe_data['u1'])
        u2 = df.Constant(dbe_data['u2'])
        u3 = df.Constant(dbe_data['u3'])

        leftBoundary = df.Expression('u0*(1-x[1]) + u1*x[1]' , u0=u0, u1=u1, degree=1)
        rightBoundary = df.Expression('u2*(1-x[1]) + u3*x[1]' , u2=u2, u3=u3, degree=1)


        bcs_specs = [DirichletSpecification(rightBoundary, self._boundaries['right']), DirichletSpecification(leftBoundary, self._boundaries['left'])]

        # this is quite awkward in order to avoid circular dependencies
        if dbe is None:
            bcs_wrapper = DirichletBoundaryCondition(bcs_specs, encoding_data = dbe_data, encoding_type = 'NDP')
        else:
            bcs_wrapper = DirichletBoundaryCondition(bcs_specs, encoding=dbe)

        return bcs_wrapper

    def reconstruct_dirichlet(self, DirichletEncoding):

        return self.sample_dirichlet_boundary_condition(dbe = DirichletEncoding)

