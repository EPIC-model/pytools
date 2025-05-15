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

    _data = dset.get_data(name=field, step=step, copy_periodic=copy_periodic)
    _axis = dset.get_axis(name=axis, copy_periodic=copy_periodic)

    xlab = ''
    if measure == 'mean':
        _mean = np.mean(a=_data, axis=_axes)
        xlab = 'mean of '

    if axis == 'z':
        ax.plot(_mean, _axis, **kwargs)
        ax.set_xlabel(axis)
        ax.set_ylabel(xlab + dset.get_label(field))
    else:
        ax.plot(_axis, _mean, **kwargs)
        ax.set_xlabel(xlab + dset.get_label(field))
        ax.set_ylabel(axis)
