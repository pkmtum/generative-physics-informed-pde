import torch
import fenics as df
from physics.LinearElliptic import LinearEllipticPhysics
from bottleneck.Decoder import CNNDecoder
from bottleneck.Encoder import CNNEncoder
from fawkes.utils import refine
from bottleneck.components import EffectivePropertyMap, ReducedOrderModelOperator
from bottleneck.generative import GenerativeModel
from bottleneck.components import PhysicsResolutionInterpolator


def fetch_dtype_device(dtype, device):

    if dtype.lower() == 'float32':
        dtype = torch.float32
    elif dtype.lower () == 'float64' or dtype.lower() == 'double':
        dtype = torch.double
    else:
        raise ValueError('dtype option not recognized. options are: float32, float64 (double)')

    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif device.lower() == 'cuda' or device.lower() == 'cuda:0' or device.lower() == 'gpu':
        device = torch.device('cuda:0')
    elif device.lower() == 'best':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(">>> Setting GPU as device for computation <<<")
        else:
            device = torch.device('cpu')
            print(">>> Setting CPU as device for computation <<<")
    else:
        raise ValueError('device option not recognized. options are: cpu, cuda, best')

    return dtype, device


class ModelFactory(object):

    def __init__(self, *args, **kwargs):

        self.params = dict()
        self.params['independent_X'] = True
        self.params['ptype'] = None
        self.params['dim_latent'] = None
        self.params['binary_field'] = False
        self.params['dtype'] = None
        self.params['device'] = None
        self.params['nx_rom'] = None
        self.params['ny_rom'] = None
        self.params['eff_property_map_hidden_layers'] = None
        self.params['num_refines'] = None
        self.params['make_scalar_effective_property_map'] = False

        self._identifier = None
        self._custom_params_set_flag = False

        self._dtype = None
        self._device = None

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype, self._device = fetch_dtype_device(self.params['dtype'], self.params['device'])
        return self._dtype

    @property
    def device(self):
        if self._device is None:
            self._dtype, self._device = fetch_dtype_device(self.params['dtype'], self.params['device'])
        return self._device

    @classmethod
    def FromIdentifier(cls, identifier, *args, **kwargs):

        classname = identifier
        factory_class = globals()[classname]
        return factory_class(*args, **kwargs)

    @property
    def identifier(self):
        if self._identifier is not None:
            return self._identifier
        return type(self).__name__

    def set(self, *args):

        if len(args) == 1 and isinstance(args[0], dict):
            lparams = args[0]
            for key, val in lparams.items():
                if key not in self.params:
                    raise KeyError
                self.params[key] = val

            self._custom_params_set_flag = True

        elif len(args) == 2 and isinstance(args[0], str):
            key = args[0]
            val = args[1]
            if key not in self.params:
                raise KeyError
            self.params[key] = val
        else:
            raise ValueError

    def _setup(self, **kwargs):

        if not self._custom_params_set_flag:
            raise RuntimeError

        for key in kwargs:
            if key not in self.params:
                raise KeyError

        if kwargs:
            raise RuntimeError

        def gp(key):

            value = kwargs.get(key, None)
            if value is None:
                value = self.params[key]

            if value is None:
                raise ValueError
            return value

        dtype, device = fetch_dtype_device(gp('dtype'), gp('device'))

        nx_fom = gp('nx_rom')*(2**gp('num_refines'))
        ny_fom = gp('ny_rom')*(2**gp('num_refines'))
        mesh_rom = df.UnitSquareMesh(df.MPI.comm_self, gp('nx_rom'), gp('ny_rom'))
        mesh_fom = refine(mesh_rom, gp('num_refines'))

        ########## Physics #########
        physics = dict()
        physics['fom'] = LinearEllipticPhysics('fom', gp('ptype'), mesh_fom)
        physics['rom'] = LinearEllipticPhysics('rom', gp('ptype'), mesh_rom)
        dh = PhysicsResolutionInterpolator(physics, dtype=torch.double)
        physics['W'] = dh._W.detach().cpu().numpy().T.copy()

        return physics, gp, ny_fom, nx_fom, dtype, device


    def _closure(self, physics, encoder, decoder, getparams, dtype, device, make_scalar_eff_prop = False):

        encoder = encoder.to(dtype=dtype, device=device)
        decoder = decoder.to(dtype=dtype, device=device)
        f = decoder

        if getparams('make_scalar_effective_property_map'):
            raise DeprecationWarning
        else:
            g = ReducedOrderModelOperator.FromPhysics(physics, dtype=dtype, device=device)
            gp = EffectivePropertyMap(f.dim_latent, g.dim_effective_property,
                                      num_hidden_layers=getparams('eff_property_map_hidden_layers'), independent_X=getparams('independent_X'), dtype=dtype,
                                      device=device)

        model = GenerativeModel(f=f, g=g, gp=gp, dtype=dtype, device=device)

        discriminative_model = model.extract_discriminative_model(FromLatentEncoding=False, duplicate=True, encoder=encoder)

        return physics, model, discriminative_model, encoder, dtype, device

    @property
    def physics(self):

        # wasteful, but there we go
        results = self.setup()
        return results[0]

class highres(ModelFactory):

    def __init__(self, **kwargs):

        super(highres, self).__init__(**kwargs)

        self.params['ptype'] = 'ND'
        self.params['dim_latent'] = 64
        self.params['binary_field'] = False
        self.params['dtype'] = 'float32'
        self.params['device'] = 'best'
        self.params['nx_rom'] = 8
        self.params['ny_rom'] = 8
        self.params['eff_property_map_hidden_layers'] = 0
        self.params['num_refines'] = 3
        self.params['droprate'] = 0.2

        self._identifier = 'highres'
        self.set(kwargs)

    def setup(self, *args, **kwargs):

        physics, gp, ny_fom, nx_fom, dtype, device = self._setup(*args, **kwargs)

        latent_img_size = (8, 8)
        latent_img_features = 1
        init_features_decoder = 6
        init_features_encoder = 6
        blocks = [1, 2, 1]
        growth_rate = 4
        upsample = 'nearest'

        target_img_size = gp('nx_rom') * (2 ** gp('num_refines'))

        decoder = CNNDecoder(target_img_size, gp('dim_latent'), latent_img_size, latent_img_features,
                             init_features_decoder,
                             blocks, gp('binary_field'), growth_rate, drop_rate=gp('droprate'), upsample=upsample,
                             force_single_output=False)
        encoder = CNNEncoder(target_img_size, gp('dim_latent'), blocks, growth_rate, init_features_encoder,
                             drop_rate=gp('droprate'))

        return self._closure(physics, encoder, decoder, gp, dtype, device)

class highres32(ModelFactory):

    def __init__(self, **kwargs):

        super(highres32, self).__init__(**kwargs)

        self.params['ptype'] = 'NDP'
        self.params['dim_latent'] = 16
        self.params['binary_field'] = False
        self.params['dtype'] = 'float32'
        self.params['device'] = 'best'
        self.params['nx_rom'] = 4
        self.params['ny_rom'] = 4
        self.params['eff_property_map_hidden_layers'] = 0
        self.params['num_refines'] = 3
        self.params['droprate'] = 0
        self.params['homoscedastic'] = False

        self._identifier = 'highres32'
        self.set(kwargs)

    def setup(self, *args, **kwargs):

        physics, gp, ny_fom, nx_fom, dtype, device = self._setup(*args, **kwargs)

        latent_img_size = (8,8)
        latent_img_features = 1
        init_features_decoder = 4
        init_features_encoder = 4
        blocks = [1,1]
        growth_rate = 4
        upsample = 'nearest'

        target_img_size = gp('nx_rom') * (2 ** gp('num_refines'))

        decoder = CNNDecoder(target_img_size, gp('dim_latent'), latent_img_size, latent_img_features,
                             init_features_decoder,
                             blocks, gp('binary_field'), growth_rate, drop_rate=gp('droprate'), upsample=upsample,
                             force_single_output=False, homoscedastic=gp('homoscedastic'))
        encoder = CNNEncoder(target_img_size, gp('dim_latent'), blocks, growth_rate, init_features_encoder,
                             drop_rate=gp('droprate'))

        return self._closure(physics, encoder, decoder, gp, dtype, device)

