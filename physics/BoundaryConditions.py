import numpy as np
import dolfin as df
import torch
import warnings
from fawkes.BoundaryConditions import BoundaryEncoding, BoundaryEncodingEnsemble

class BoundaryConditionEnsemble(object):

    def __init__(self, general_boundary_conditions):

        self._general_boundary_conditions = general_boundary_conditions

        for gbc in self._general_boundary_conditions:
            gbc.register_ensemble(self)

        self._V = dict()
        self._constrained_dofs = dict()
        self._free_dofs = dict()
        self._constrained_dofs_values = dict()
        self._F = dict()

    def __len__(self):
        return len(self._general_boundary_conditions)

    def __iter__(self):

        yield from self._general_boundary_conditions

    def _parse_identifier(self, identifier):
        return identifier.lower()

    def encode(self):

        encodings = list()
        for gbc in self._general_boundary_conditions:
            encodings.append(gbc.encode())

        return BoundaryEncodingEnsemble(encodings)

    @classmethod
    def FromEncoding(cls, encoding, model_factory, V_rom, V_fom):
    
    
        assert V_rom.dim() < V_fom.dim()

        general_boundary_conditions = list()
        for enc in encoding:
            general_boundary_conditions.append(GeneralBoundaryCondition.FromEncoding(enc, model_factory))

        BCE = BoundaryConditionEnsemble(general_boundary_conditions)

        BCE.register_function_space('rom', V_rom)
        BCE.register_function_space('fom', V_fom)

        return BCE


    def __getitem__(self, key):

        if isinstance(key, torch.Tensor):
            raise NotImplementedError

        if isinstance(key, list):
            return [self._general_boundary_conditions[mykey] for mykey in key]
        else:
            return self._general_boundary_conditions[key]


    def register_function_space(self, identifier, V):

        assert isinstance(identifier, str) and isinstance(V, df.FunctionSpace)
        identifier = self._parse_identifier(identifier)

        if identifier in self._V.keys():
            warnings.warn('Function space with this identifier ({}) has already been registered. Skipping ...')
            return

        self._V[identifier] = V
        self._cache_function_space(identifier, V)

    def check_if_registered(self, identifier):
        return identifier in self._V

    def delete_function_spaces(self):

        self._V = dict()
        self._constrained_dofs = dict()
        self._free_dofs = dict()
        self._constrained_dofs_values = dict()
        self._F = dict()

        for gbc in self._general_boundary_conditions:
            gbc.delete_function_spaces()

    @classmethod
    def FromFactory(cls, factory, N):

        try:
            bcs = [factory.sample_boundary_condition() for n in range(N)]
        except AttributeError:
            print("The factory does not seem to have a sampling function for boundary conditions")
            raise

        return cls(bcs)

    def _cache_function_space(self, identifier, V):

        identifier = self._parse_identifier(identifier)
        constrained_dofs, _, free_dofs = self._general_boundary_conditions[0].dirichlet_boundary_condition.extract(V, ReturnFreeDofs=True)
        self._constrained_dofs[identifier] = constrained_dofs
        self._free_dofs[identifier] = free_dofs

        self._constrained_dofs_values[identifier] = np.zeros((len(self), constrained_dofs.size))

        for n, gbc in enumerate(self._general_boundary_conditions):
            gbc.register_function_space(identifier, V)
            self._constrained_dofs_values[identifier][n,:] = gbc.constrained_dofs_values(identifier)


    def constrained_dofs(self, identifier):
        return self._constrained_dofs[self._parse_identifier(identifier)]

    def num_constrained_dofs(self, identifier):
        return self.constrained_dofs[self._parse_identifier(identifier)].size

    def free_dofs(self, identifier):
        return self._free_dofs[self._parse_identifier(identifier)]

    def constrained_dofs_values(self, identifier):
        return self._constrained_dofs_values[self._parse_identifier(identifier)]

    def FULL_F_WITH_APPLIED_BC(self, identifier):

        identifier = self._parse_identifier(identifier)
        F = self._F.get(identifier)

        if F is None:

            F = np.zeros((len(self), self._V[identifier].dim()))

            for n, gbc in enumerate(self._general_boundary_conditions):
                F[n,:] = gbc.assemble_vanilla_force_vector(identifier)
                F[n,gbc.constrained_dofs(identifier)] = gbc.constrained_dofs_values(identifier)

            self._F[self._parse_identifier(identifier)] = F

        return F

class GeneralBoundaryCondition(object):

    def __init__(self, dirichlet_boundary_condition, neumann_boundary_condition, set_alpha = None, ensemble = None):

        self.dirichlet_boundary_condition = dirichlet_boundary_condition
        self._ensemble = ensemble
        self.neumann_boundary_condition = neumann_boundary_condition

        self._V = dict()
        self._constrained_dofs = dict()
        self._constrained_dofs_values = dict()
        self._free_dofs = dict()
        self._f = dict()

    @classmethod
    def FromEncoding(cls, encoding, model_factory):

        nbc = encoding.neumann_encoding.reconstruct(model_factory)
        dbc = encoding.dirichlet_encoding.reconstruct(model_factory)
        return cls(dirichlet_boundary_condition=dbc, neumann_boundary_condition=nbc, set_alpha=None)

    def encode(self):

        dbe = self.dirichlet_boundary_condition.encode()
        nbe = self.neumann_boundary_condition.encode()
        return BoundaryEncoding(dirichlet_encoding=dbe, neumann_encoding=nbe)

    @property
    def _set_alpha(self):
        raise NotImplementedError

    @_set_alpha.setter
    def _set_alpha(self, value):
        raise NotImplementedError

    def register_ensemble(self, ensemble):

        self._ensemble = ensemble

    def _parse_identifier(self, identifier):
        return identifier.lower()

    def register_function_space(self, identifier, V):

        assert isinstance(identifier, str) and isinstance(V, df.FunctionSpace)
        identifier = self._parse_identifier(identifier)

        if identifier in self._V.keys():
            warnings.warn('Function space with this identifier ({}) has already been registered. Skipping ...')
            return

        identifier = self._parse_identifier(identifier)
        self._V[identifier] = V
        self._cache_function_space(identifier, V)

    def _cache_function_space(self, identifier, V):

        identifier = self._parse_identifier(identifier)

        if self._ensemble is not None:
            _ , constrained_dofs_values = self.dirichlet_boundary_condition.extract(V, ReturnFreeDofs=False)
            self._constrained_dofs_values[identifier] = constrained_dofs_values
        else:
            constrained_dofs , constrained_dofs_values, free_dofs = self.dirichlet_boundary_condition.extract(V, ReturnFreeDofs=True)
            self._constrained_dofs_values[identifier] = constrained_dofs_values
            self._free_dofs[identifier] = free_dofs
            self._constrained_dofs[identifier] = constrained_dofs

    def delete_function_spaces(self):
        raise NotImplementedError

    def constrained_dofs(self, identifier):

        identifier = self._parse_identifier(identifier)
        value = self._constrained_dofs.get(identifier)

        if value is not None:
            return value
        elif value is None and self._ensemble is not None:
            return self._ensemble.constrained_dofs(identifier)
        elif value is None and self._ensemble is None:
            raise ValueError

    def free_dofs(self, identifier):

        identifier = self._parse_identifier(identifier)
        value = self._free_dofs.get(identifier)

        if value is not None:
            return value
        elif value is None and self._ensemble is not None:
            return self._ensemble.free_dofs(identifier)
        elif value is None and self._ensemble is None:
            raise ValueError

    def constrained_dofs_values(self, identifier):

        identifier = self._parse_identifier(identifier)
        return self._constrained_dofs_values[identifier]

    def assemble_vanilla_force_vector(self, identifier):

        identifier = self._parse_identifier(identifier)
        return self.neumann_boundary_condition.assemble_flux(self._V[identifier])




