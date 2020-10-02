import numpy as np
import dolfin as df
from fawkes.utils import AssembleBasisFunctionMatrix
from physics.LinearElliptic import LinearEllipticPhysics


class FluxForm(object):

    def __init__(self, u, alpha):

        self._u = u
        self._alpha = alpha
        self._normal = df.FacetNormal(u.function_space().mesh())
        self._form = None
        self._side = '+'

    def _append(self, lform):

        if self._form is None:
            self._form = lform
        else:
            self._form = self._form + lform

    def append_ds(self, ds):

        self._append(df.dot(self._alpha*df.grad(self._u), self._normal)*ds)


    def append_dS(self, dS):

        self._append(df.dot(self._alpha(self._side) * df.grad(self._u)(self._side), self._normal(self._side))*dS)


    def append_dx(self, dx):

        self._append(df.Constant(0.0)*dx)

    def assemble_derivative(self, x):

        self._alpha.vector()[:] = x
        return df.assemble(df.derivative(self._form, self._u)).get_local()

class FluxConstraintReducedOrderModel(object):


    def __init__(self, physics, bc = None):

        # warnings.warn('Assumes that coarse and fine mesh are compliant. User is responsible for this.')
        self._physics = physics
        self._V = self._physics['fom'].V
        self._Vc = self._physics['fom'].Vc
        self._mesh_coarse = physics['rom'].mesh
        self._mesh_fine = physics['fom'].mesh

        self._flux_forms = list()

        if bc is None:
            bc = physics['fom'].factory.sample_boundary_condition()
            bc.register_function_space('fom', physics['fom'].V)
            bc.register_function_space('rom', physics['rom'].V)

        self._rom_exterior_facets = bc.dirichlet_boundary_condition.mark_facets(physics['rom'].mesh)

        self.Gamma = np.zeros((self._V.dim(), self._mesh_coarse.num_cells()))

        self._initialized = False

    @property
    def initialized(self):
        return self._initialized

    @property
    def tdim(self):
        return self._mesh_coarse.topology().dim()

    @property
    def N(self):
        return self._mesh_coarse.num_cells()


    def _assemble(self, x):

        Gamma = np.zeros((self._V.dim(), self.N))

        if not len(self._flux_forms) == self.N:
            self.create_measures()

        for n, form in enumerate(self._flux_forms):
            Gamma[:,n] = form.assemble_derivative(x).copy()

        return Gamma

    def assemble_reduced(self, x, bc):

        Gamma = self._assemble(x)
        return self._create_reduced_system(Gamma, bc)


    def create_measures(self):

        for rom_cell_counter, rom_cell in enumerate(df.cells(self._mesh_coarse)):

            form = FluxForm(df.Function(self._V), df.Function(self._Vc))
            facetfct = df.MeshFunction('size_t', self._mesh_fine, self._mesh_fine.topology().dim() - 1)
            facetfct.set_all(0)

            for local_facet_id, rom_facet in enumerate(df.facets(rom_cell)):
                for fom_facet in df.facets(self._mesh_fine):

                    mp = fom_facet.midpoint()

                    p0 = df.Vertex(self._mesh_coarse, rom_facet.entities(0)[0])
                    p1 = df.Vertex(self._mesh_coarse, rom_facet.entities(0)[1])
                    p0 = df.Point(np.array([p0.x(0), p0.x(1)]))
                    p1 = df.Point(np.array([p1.x(0), p1.x(1)]))

                    eps = mp.distance(p0) + mp.distance(p1) - p0.distance(p1)

                    if eps < 1e-12:

                        facetfct.set_value(fom_facet.index(), local_facet_id+1)

                if self._rom_exterior_facets[rom_facet.index()]:
                    form.append_ds(df.Measure('ds', domain=self._mesh_fine, subdomain_data=facetfct, subdomain_id= local_facet_id+1))
                else:
                    form.append_dS(df.Measure('dS', domain=self._mesh_fine, subdomain_data=facetfct, subdomain_id = local_facet_id+1))

            cellfct = df.MeshFunction('size_t', self._mesh_fine, self._mesh_fine.topology().dim())
            cellfct.set_all(0)

            for fom_cell in df.cells(self._mesh_fine):
                if rom_cell.contains(fom_cell.midpoint()):
                    cellfct.set_value(fom_cell.index(), 1)

            form.append_dx(df.Measure('dx', domain=self._mesh_fine, subdomain_data=cellfct, subdomain_id = 1))
            self._flux_forms.append(form)


        self._initialized = True



    def _create_reduced_system(self, Gamma, bc):

        constrained_dofs_values = bc.constrained_dofs_values('fom')
        constrained_dofs = bc.constrained_dofs('fom')
        free_dofs = bc.free_dofs('fom')
        Gamma_reduced = np.zeros((self._physics['fom'].dim_out, self._mesh_coarse.num_cells()))
        alpha_reduced = np.zeros(self._mesh_coarse.num_cells())

        for n in range(self.N):
            Gamma_reduced[:,n] = Gamma[free_dofs, n]
            alpha_reduced[n] = np.dot(self.Gamma[constrained_dofs, n], constrained_dofs_values)

        # dirty fix
        alpha_reduced = alpha_reduced * (-1)

        return Gamma_reduced.T, alpha_reduced



class QOI(object):

    # todo: move this class somewhere else (it is misplaced here)
    def __init__(self, physics , mx = 0.5, my = 0.5, L = None):


        assert isinstance(physics, LinearEllipticPhysics)

        self._V = physics.V
        self._mx = mx
        self._my = my
        self._L = L
        self._physics = physics

        self._dof = None
        self._functional = None

        self._assemble()


    def _assemble(self):

        if self._L is None:
            pp = np.array([self._mx, self._my]).reshape(1, -1)
            self._functional = AssembleBasisFunctionMatrix(self._V, pp).toarray().flatten()
        else:
            dom = SquareSubdomain(self._L, mx=self._mx, my=self._my)
            markers_dom = df.MeshFunction('size_t', self._physics.mesh, self._physics.mesh.topology().dim())
            markers_dom.set_all(0)
            dom.mark(markers_dom, 1)
            self._mesh_fct = markers_dom

            dx = df.Measure('dx', domain=self._physics.mesh, subdomain_id = 1, subdomain_data=markers_dom)
            ff = df.Function(self._physics.V)
            ff.vector()[:] = 1
            qoi = ff*dx

            self._functional = df.assemble(df.derivative(qoi, ff)).get_local()

    def _complete(self, Y, BC):

        assert isinstance(Y, np.ndarray)
        assert Y.ndim == 2

        # augment prediction with strongly enforced boundary conditions
        Y_full = np.zeros((Y.shape[0], self._V.dim()))
        for n in range(Y.shape[0]):
            Y_full[n,:] = self._physics.scatter_restricted_solution(Y[n,:].copy().flatten(), BC[n], ReturnFunction=False)
        return Y_full


    def extract(self, Y, BC = None):

        if BC is not None:
            Y = self._complete(Y,BC)
        else:
            assert isinstance(Y, np.ndarray)
            assert Y.ndim == 2
            assert Y.shape[1]  == self._V.dim()

        if self._dof is not None:
            return Y[:, self._dof]
        elif self._functional is not None:
            return (Y @ self._functional).flatten()



class SquareSubdomain(df.SubDomain):

    # todo: move this class somewhere else (it is misplaced here)
    def __init__(self, L, mx, my):

        super(SquareSubdomain, self).__init__()
        self._L = L
        self._mx = mx
        self._my = my

    def inside(self, x, on_boundary):

        L = self._L
        dy = np.abs(x[1] - self._my)
        dx = np.abs(x[0] - self._mx)
        if dx <= L and dy <= L:
            return True
        else:
            return False