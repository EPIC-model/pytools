from tools.netcdf.field_dataset import FieldDataset
import matplotlib as mpl
import numpy as np
import scipy
from tools.geometry import Plane

def profile_plot(ax,
                 dset: FieldDataset,
                 step: int,
                 axis: str,
                 field: str,
                 measure: str = 'mean',
                 **kwargs):
    """
    Generate a profile plot.

    Parameters
    ----------
    ax: plotting axis
    dset: field dataset
    step: time frame of dataset
    axis: 'x', 'y' or 'z'
    field: name of field data
    measure: 'mean' or 'rms'
    """

    if not isinstance(dset, FieldDataset):
        raise TypeError("Dataset must be of type 'FieldDataset'")

    if not dset.is_open():
        raise RuntimeError("Dataset is closed.")

    if axis not in ('x', 'y', 'z'):
        raise ValueError("Argument 'axis' must be 'x', 'y' or 'z'.")

    if axis == 'x':
        copy_periodic = {'x'}
        _axes = (1, 2)
    elif axis == 'y':
        copy_periodic = {'y'}
        _axes = (0, 2)
    elif axis == 'z':
        copy_periodic = {}
        _axes = (0, 1)

    _data = dset.get_data(varname=field, step=step, copy_periodic=copy_periodic)
    _axis = dset.get_axis(varname=axis, copy_periodic=copy_periodic)

    lab = ''
    if measure == 'mean':
        _profile = np.mean(a=_data, axis=_axes)
        lab = 'mean of '
    elif measure == 'rms':
        lab = 'rms of '
        _profile = np.sqrt(np.mean(a=_data**2, axis=_axes))
    else:
        raise ValueError("Only 'mean' or 'rms' allowed.")

    if axis == 'z':
        ax.plot(_profile, _axis, **kwargs)
        ax.set_xlabel(lab + dset.get_label(field))
        ax.set_ylabel(axis)
    else:
        ax.plot(_axis, _profile, **kwargs)
        ax.set_xlabel(axis)
        ax.set_ylabel(lab + dset.get_label(field))


def slice_plot(ax: mpl.axes._axes.Axes,
               dset: FieldDataset,
               step: int,
               plane: Plane,
               field: str,
               **kwargs):
    """
    Generate a slice plot.

    Parameters
    ----------
    ax: plotting axis
    dset: field dataset
    step: time frame of dataset
    plane: where to take the slice
    field: name of field
    """

    if not isinstance(dset, FieldDataset):
        raise TypeError("Dataset must be of type 'FieldDataset'")

    if not dset.is_open():
        raise RuntimeError("Dataset is closed.")

    _method = kwargs.pop('interpolation', 'linear')
    _colorbar = kwargs.pop('colorbar', False)

    _copy_periodic = {'x', 'y'}

    _data = dset.get_data(varname=field, step=step, copy_periodic=_copy_periodic)

    _x = dset.get_axis(varname='x', copy_periodic=_copy_periodic)
    _y = dset.get_axis(varname='y', copy_periodic=_copy_periodic)
    _z = dset.get_axis(varname='z')

    interp = scipy.interpolate.RegularGridInterpolator(points=(_x, _y, _z),
                                                       values=_data,
                                                       method=_method)

    if plane.orientation == 'xy':
        _pg, _qg = dset.get_meshgrid(coords={'x', 'y'}, copy_periodic=_copy_periodic)
        _rg = plane.height * np.ones(_pg.size)
        pts = np.column_stack([_pg.ravel(), _qg.ravel(), _rg])
        xlabel = 'x'
        ylabel = 'y'
    elif plane.orientation == 'xz':
        _pg, _qg = dset.get_meshgrid(coords={'x', 'z'}, copy_periodic=_copy_periodic)
        _rg = plane.height * np.ones(_pg.size)
        pts = np.column_stack([_pg.ravel(), _rg, _qg.ravel()])
        xlabel = 'x'
        ylabel = 'z'
    elif plane.orientation == 'yz':
        _pg, _qg = dset.get_meshgrid(coords={'y', 'z'}, copy_periodic=_copy_periodic)
        _rg = plane.height * np.ones(_pg.size)
        pts = np.column_stack([_rg, _pg.ravel(), _qg.ravel()])
        xlabel = 'y'
        ylabel = 'z'
    else:
        raise RuntimeError("Only xy-, xz- or yz-plane supported.")

    _values = interp(pts).reshape(_pg.shape)

    pc = ax.pcolormesh(_pg, _qg, _values, shading='gouraud', **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if _colorbar:
        _cbar = ax.cax.colorbar(pc)
        _cbar = _cbar.set_label(dset.get_label(field))

    #xmin = _pg.min()
    #xmax = _qg.max()
    #ymin = _pg.min()
    #ymax = _qg.max()
    #ax.imshow(X=_values.transpose(),
              #interpolation='bilinear',
              #interpolation_stage='data',
              #origin='lower',
              #extent=[xmin, xmax, ymin, ymax],
              #aspect='auto')


