from tools.netcdf.stat_dataset import StatDataset
import matplotlib as mpl

def line_plot(ax: mpl.axes._axes.Axes, dset: StatDataset, x: str, y: str, **kwargs):
    """
    Generate a line plot.

    x: name of variable on x-axis
    y: name of variable on y-axis
    """
    if not isinstance(dset, StatDataset):
        raise TypeError("Dataset must be of type 'StatDataset'")

    if not dset.is_open():
        raise RuntimeError("Dataset is closed.")

    xdata = dset.get_data(varname=x, step=-1)
    ydata = dset.get_data(varname=y, step=-1)
    ax.plot(xdata, ydata, **kwargs)
    ax.set_xlabel(dset.get_label(varname=x))
    ax.set_ylabel(dset.get_label(varname=y))
