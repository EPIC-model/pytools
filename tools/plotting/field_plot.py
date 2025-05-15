from tools.netcdf.field_dataset import FieldDataset
import matplotlib as mpl
import numpy as np

def profile_plot(ax: mpl.axes._axes.Axes,
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
