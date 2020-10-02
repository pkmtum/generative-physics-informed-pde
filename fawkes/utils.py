from scipy.sparse import csr_matrix
import fenics as df
import numpy as np
from numpy import arange
from petsc4py import PETSc
import warnings


def refine(mesh, N=1):

    for i in range(N):
        mesh = df.refine(mesh)

    return mesh


def ExtractDirichletValues(bcs):

    if not isinstance(bcs, list):
        bcs = [bcs]

    dofs = np.array([dof for bc in bcs for dof in bc.get_boundary_values().keys()], dtype=np.int)
    vals = np.array([val for bc in bcs for val in bc.get_boundary_values().values()], dtype=np.double)

    dofs, index = np.unique(dofs, return_index=True)
    values = vals[index]

    return dofs, values


def ConvertFenicsBackendToScipyCSRSparse(A):

        A_mat = df.as_backend_type(A).mat()
        return csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
        
        
def ConvertFenicsBackendToScipyCOOSparse(A):

        return ConvertFenicsBackendToScipyCSRSparse(A).tocoo()


def ConstructDiscontinuousGalerkinScalarFunctionSpaceMapping(V):

    if (V.ufl_element().family() != 'Discontinuous Lagrange' or V.ufl_element().degree() != 0 or V.num_sub_spaces() != 0):
        raise TypeError('This mapping can only be applied to zero order discontinuous scalar Galerkin function spaces')

    mesh = V.mesh()
    Cells = [df.Cell(mesh, i) for i in range(mesh.num_cells())]

    DG0_Scalar_Map = np.zeros(V.dim(), dtype=int)
    dmap = V.dofmap()

    for cell in Cells:
        cellid = cell.index()
        DG0_Scalar_Map[cellid] = dmap.cell_dofs(cellid)

    return DG0_Scalar_Map


def AssembleDGOverlap(Vc, Vf):

    W = AssembleMeshOverlapMatrix(Vc, Vf).toarray()

    overlap = [list() for i in range(Vc.dim())]

    for i in range(Vc.dim()):
        overlap[i] = np.argwhere(W[i, :].flatten()).flatten()

    return overlap

def AssembleMeshOverlapMatrix(Vc, Vf):

    # assumes that Vc and Vf are conforming function spaces
    meshc = Vc.mesh()
    meshf = Vf.mesh()

    if Vc.dim() != meshc.num_cells() or Vf.dim() != meshf.num_cells():
        raise TypeError('One of supplied function spaces not of type DG0')

    if meshc.num_cells() > meshf.num_cells():
        raise ValueError

    tree = meshc.bounding_box_tree()

    reversemap = np.zeros(meshf.num_cells(), dtype=np.int)
    areafraction = np.zeros(meshf.num_cells())
    coarse_area_by_id = np.zeros(meshc.num_cells())

    dofmapc = Vc.dofmap()
    dofmapf = Vf.dofmap()

    for cell in df.cells(meshc):
        coarse_area_by_id[cell.index()] = cell.volume()

    for cell in df.cells(meshf):
        cell_id = tree.compute_first_entity_collision(cell.midpoint())
        coarse_dof = dofmapc.cell_dofs(cell_id)
        fine_dof = dofmapf.cell_dofs(cell.index())
        reversemap[fine_dof] = coarse_dof
        areafraction[fine_dof] = cell.volume() / coarse_area_by_id[cell_id]

    _W = np.zeros((Vc.dim(), Vf.dim()))

    #  Issue: slow and not sparse
    for i in range(Vc.dim()):
        finedofs = np.where(reversemap == i)[0]
        _W[i, finedofs] = areafraction[finedofs]

    W = csr_matrix(_W); del(_W);
    return W




def AssembleBasisFunctionMatrix(V, points, *, Global_V=None, ReturnType='scipy'):

    if points.ndim != 2:
        raise Exception

    if points.shape[1] > 3:
        raise Exception

    dim = V.num_sub_spaces()
    if dim == 0:
        dim = 1

    mesh = V.mesh()
    element = V.element()
    nbasis = element.space_dimension()
    nrows = len(points)

    if Global_V is None:
        ncols = V.dim()
    else:
        ncols = Global_V.dim()

    comm_self = df.MPI.comm_self

    if dim > 1:
        mat = PETSc.Mat().createAIJ(size=(nrows * dim, ncols), comm=comm_self)
        V_subspaces = V.split()
    else:
        mat = PETSc.Mat().createAIJ(size=(nrows, ncols), comm=comm_self, )

    mat.setUp()
    mat.assemblyBegin()

    dmap = V.dofmap()
    tree = mesh.bounding_box_tree()

    for row, pt in enumerate(points):

        cell_id = tree.compute_first_entity_collision(df.Point(*pt))

        if cell_id >= mesh.num_cells():
            raise Exception('No collision with mesh for requested point')

        cell = df.Cell(mesh, cell_id)
        vertex_coordinates = cell.get_vertex_coordinates()
        vertex_coordinates = np.array(vertex_coordinates, dtype=np.float64)
        cell_orientation = cell.orientation()
        basis_values = element.evaluate_basis_all(pt, vertex_coordinates, cell_orientation)
        assert basis_values.size == dim*nbasis

        if dim > 1:

            col_indices = dmap.cell_dofs(cell_id)

            for ii, vsub in enumerate(V_subspaces):
                row_indices = [dim * row + ii]
                submap = arange(ii, dim * nbasis, dim)
                mat.setValues(row_indices, col_indices, basis_values[submap], PETSc.InsertMode.INSERT_VALUES)
        else:

            row_indices = [row]
            col_indices = dmap.cell_dofs(cell_id)
            mat.setValues(row_indices, col_indices, basis_values, PETSc.InsertMode.INSERT_VALUES)

    mat.assemblyEnd()

    if ReturnType == 'scipy':
        r = mat.getValuesCSR()
        I = csr_matrix((r[2], r[1], r[0]), shape=(nrows * dim, ncols))
    elif ReturnType.lower() == 'csr_data':
        r = mat.getValuesCSR()
        return r[2], r[1], r[0]
    elif ReturnType == 'fenics':
        I = df.PETScMatrix(mat)
    else:
        raise Exception('return type not recognized')

    return I
