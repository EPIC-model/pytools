## How to write an EPIC netCDF input file
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
