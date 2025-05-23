import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl


def create_image_grid(nrows: int, ncols: int, figsize: tuple[float, float], dpi: int, **kwargs):
    """
    Create am ImageGrid. For further info on **kwargs check the reference.

    Reference
    ---------
    https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.axes_grid.ImageGrid.html
    """

    _rect = kwargs.pop('rect', 111)
    _axes_pad = kwargs.pop('axes_pad', 0.07)

    # ensure nrows_ncols is not provided
    # because we use arguments nrows and ncols instead
    _nrows_ncols = kwargs.pop('nrows_ncols', None)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid = ImageGrid(fig=fig,
                     rect=_rect,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=_axes_pad,
                     **kwargs)

    return fig, grid

def remove_axis_ticks(ax, axes: set[str] = {'x', 'y'}):
    """
    Remove ticks from a plotting axis.
    """
    _axes = axes.intersection({'x', 'y'})

    for _axis in _axes:
        ax.tick_params(axis=_axis,
                       which='both',
                       right=False,
                       left=False,
                       bottom=False,
                       top=False,
                       labelbottom=False,
                       labeltop=False,
                       labelright=False,
                       labelleft=False)
