import dolfin as df
import math

class RadialBasisFunction(df.UserExpression):

    # slow and old
    def __init__(self, r0, l, **kwargs):
        self.r0 = r0
        self.l = l
        super().__init__(**kwargs)


    def eval_cell(self, values, x, ufc_cell):
        raise NotImplementedError


    def eval(self, values, x):

        T = (x[0] - self.r0[0]) ** 2 + (x[1] - self.r0[1]) ** 2
        values[0] = math.exp((-T / self.l ** 2))

    def value_shape(self):
        return ()


def FastRadialBasisFunction(element):

    # new and improved. r0 and l are placeholders to be changed
    r0 = df.Constant((0.5, 0.5))
    l = df.Constant(0.15)
    return df.Expression(' exp(-(pow((x[0] - r0[0]),2) + pow((x[1] - r0[1]),2))/ pow(l,2))', r0=r0, l=l, element=element), r0, l



