# Python tools for reading and plotting EPIC data

# How to load a dataset
A dataset can simply be loaded with the function `load_dataset`
```python
import tools
dset = tools.load_dataset(filename, verbose)
dset.close()
```
where `filename` is an EPIC output file and `verbose` is a boolean flag.
In this case `load_dataset` selects the correct dataset class. However, you
can also load a dataset class directly with
```python
import tools.netcdf as nc
dset = nc.FieldDataset(verbose)
dset.open(filename)
dset.close()
```
where other dataset classes are `ParcelDataset` and `StatDataset`.

# How to access data in a dataset
After loading a dataset, you can access data with
```python
dset.get_data(name, step)
```
where `name` is the data name, e.g. `z_vorticity`, and `step` is an integer. A list of
available data is printed when you type `print(dset)`. Note that `get_data` accepts
further options like `copy_periodic` for field datasets. For further info on datasets
check the corresponding classes (e.g. `help(dset)`).


# How to write an EPIC netCDF input file
You can either write a field (`FieldWriter`) or parcel (`ParcelWriter`) netCDF file. An example
template is given below.

```python
import tools.netcdf as nc

# or ncfile = nc.ParcelWriter()
ncfile = nc.FieldWriter()

ncfile.open('filename.nc')

"""
x_vorticity and y_vorticity are either parcel of field data
"""
ncfile.add_dataset('x_vorticity', x_vorticity, unit='1/s')
ncfile.add_dataset('y_vorticity', y_vorticity, unit='1/s')

# add physical quantities that are recognised by EPIC
ncfile.add_physical_quantity(..., ...)
ncfile.add_physical_quantity(..., ...)

# add parameters that are recognised by EPIC
ncfile.add_parameter(..., ...)
ncfile.add_parameter(..., ...)

"""
origin: lower left corner of domain, 3D array
extent: domain extent, 3D array
nx, ny, nz: number of grid cells
"""
ncfile.add_box(origin, extent, [nx, ny, nz])

ncfile.close()
```
