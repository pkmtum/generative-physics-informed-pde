import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D


def PlotObj(obj, fig, ax, nn, flat =None, *args, **kwargs):

    X,Y = np.meshgrid(np.linspace(0,1,nn), np.linspace(0,1,nn))
    xx = X.flatten()
    yy = Y.flatten()
    zz = np.zeros(xx.shape)

    for n in range(xx.size):
        zz[n] = obj(df.Point([xx[n], yy[n]]))

    X = xx.reshape(X.shape)
    Y = yy.reshape(X.shape)
    Z = zz.reshape(X.shape)

    if flat is None:
        ls = LightSource(azdeg=0, altdeg=65)
        colors = ls.shade(Z, plt.cm.magma)
        urf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                                       antialiased=True, facecolors = colors, *args, **kwargs) #  facecolors=colors
    else:
        urf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                                           antialiased=True, color='b', *args, **kwargs)
    ax.view_init(5, 90)

def PlotFunction2D(f, f_lower = None, f_upper = None, nn=60, fig = None, ax = None):

    if fig is None:
        fig = plt.figure(figsize=(12, 12))

    if ax is None:
        ax = fig.gca(projection='3d')

    PlotObj(f, fig, ax, nn, alpha=1, flat=None)
    if f_upper is not None:
        PlotObj(f_upper, fig, ax, nn, alpha=0.3, flat=True)
    if f_lower is not None:
        PlotObj(f_lower, fig, ax, nn, alpha=0.3, flat=True)

    return fig, ax

