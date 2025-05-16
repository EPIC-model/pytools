## Profile plot
Profile plots can be created from gridded data using
```python
from tools.plotting import profile_plot
```
where you can use either `mean` or `rms` (root-mean-square).

#### Example
```python
dset = tools.load_dataset(filename='sqg_28x128x32_fields.nc', verbose=False)
for step in [0, 2, 4, 6, 8, 10]:
    time = dset.get_axis(varname='t')
    time = round(float(time[step]))
    profile_plot(ax=ax,
                 dset=dset,
                 step=step,
                 axis='z',
                 field='x_vorticity',
                 measure='rms',
                 label=r't = ' + str(time))
dset.close()

ax.legend()
plt.tight_layout()

plt.savefig('rms_profile_xi.png', bbox_inches='tight')
plt.close()
```

## Slice plot
You can create slice plots from gridded data with
```python
from tools.plotting import slice_plot
```

#### Example
```python
import tools
from tools.plotting import create_image_grid
import matplotlib.pyplot as plt
from tools.plotting import slice_plot
from tools.geometry import PlaneXZ

fig, grid = create_image_grid(nrows=2,
                              ncols=3,
                              figsize=(14, 6),
                              dpi=400,
                              cbar_mode='each',
                              cbar_location='top',
                              cbar_pad=0.05,
                              axes_pad=(0.2, 0.45))

dset = tools.load_dataset(filename='sqg_28x128x32_fields.nc', verbose=False)

# define plane where to take slice
xz_plane = PlaneXZ(y=0.0)

steps = [0, 2, 4, 6, 8, 10]

for i, ax in enumerate(grid):
    ax.set_aspect(50)
    slice_plot(ax=ax,
               dset=dset,
               step=steps[i],
               plane=xz_plane,
               field='buoyancy',
               colorbar=True,
               cmap=plt.cm.seismic)

plt.savefig('image_grid.png', bbox_inches='tight')
plt.close()
```
