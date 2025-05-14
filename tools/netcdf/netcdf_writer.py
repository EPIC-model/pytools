import abc
from .dataset import FileType
from datetime import datetime
from dateutil.tz import tzlocal
import netCDF4 as nc
import numpy as np
import time

class NetcdfWriter(abc.ABC):

    def __init__(self):
        self._ncfile = None

    def _open(self, filename: str, file_type: FileType):
        """
        Open a NetCDF file.
        """
        self._ncfile = nc.Dataset(filename, "w", format="NETCDF4")

        self._write_info(file_type)

        self._physical_quantities = {}

        self._parameters = {}

    def close(self):
        """
        Close file after adding all datasets.
        """
        self._write_group('physical_quantities', self._physical_quantities)
        self._write_group('parameters', self._parameters)
        self._ncfile.close()

    @abc.abstractmethod
    def add_dataset(self, name: str, values: np.ndarray, dtype: str = 'f8', **kwargs) -> None:
        """
        Add a single dataset to the file.
        """
        pass

    def add_physical_quantity(self, key: str, value : float | int | bool) -> None:
        self._physical_quantities[key] = value

    def add_parameter(self, key: str, value : float | int | bool) -> None:
        self._parameters[key] = value

    def _write_info(self, file_type):
        """
        9 March 2022
        https://stackoverflow.com/questions/32490629/getting-todays-date-in-yyyy-mm-dd-in-python
        https://docs.python.org/3/library/datetime.html
        https://stackoverflow.com/questions/415511/how-to-get-the-current-time-in-python
        https://stackoverflow.com/questions/35057968/get-system-local-timezone-in-python
        """
        self._ncfile.setncattr('file_type', file_type)
        self._ncfile.setncattr('creation_date', datetime.today().strftime('%Y/%m/%d'))
        self._ncfile.setncattr('creation_time', datetime.now().strftime('%H:%M:%S'))
        self._ncfile.setncattr('creation_zone', "UTC" + datetime.now(tzlocal()).strftime('%z'))

    def _write_group(self, group_name : str, params: dict):
        if not params == {}:
            pgrp = self._ncfile.createGroup(group_name)
            for key, val in params.items():
                pgrp.setncattr(key, val)

    def add_box(self, origin: np.ndarray, extent: np.ndarray, ncells: np.ndarray) -> None:
        """
        Add origin, extent and number of grid cells of the simulation box.

        Parameters
        ----------
        origin : np.array of floats (length 2 or 3)
            The origin of the domain (x, y, z).
        extent : np.array of floats (length 2 or 3)
            The extent of the box (x, y, z).
        ncells : np.array of floats (length 2 or 3)
            The number of cells per dimension (x, y, z).
        """
        origin = np.asarray(origin, dtype=np.float64)
        extent = np.asarray(extent, dtype=np.float64)
        ncells = np.asarray(ncells, dtype=np.int32)

        l = len(origin)

        if l < 2 or l > 3:
            raise RuntimeError("Array 'origin' must have length 2 or 3.")

        if not len(extent) == l:
            raise RuntimeError("Array 'extent' must have length 2 or 3.")

        if not len(ncells) == l:
            raise RuntimeError("Array 'ncells' must have length 2 or 3.")

        self._ncfile.setncattr(name="origin", value=origin)
        self._ncfile.setncattr(name="extent", value=extent)
        self._ncfile.setncattr(name="ncells", value=ncells)

    def _add_dataset_properties(self, var: nc._netCDF4.Variable, **kwargs) -> None:
        """
        Add a field dataset.

        Parameters
        ----------
        name:
            The field name.
        values:
            The field data.
        """
        unit = kwargs.pop('unit', '')
        if unit:
            var.units = unit

        standard_name = kwargs.pop('standard_name', '')
        if standard_name:
            var.standard_name = standard_name

        long_name = kwargs.pop('long_name', '')
        if long_name:
            var.long_name = long_name


class FieldWriter(NetcdfWriter):

    def __init__(self):
        super().__init__()
        self._dim_names_in_3d = ['x', 'y', 'z']
        self._dim_names_in_2d = ['x', 'z']

    def open(self, filename: str):
        """
        Open a field netCDF file.
        """
        super()._open(filename, file_type=FileType.FIELDS)
        self._ndims = 0

    def set_dim_names(self, names: list) -> None:
        """
        Set names of spatial dimensions.
        Default: ['x', 'y', 'z'] in 3D and ['x', 'z'] in 2D.
        """
        names = list(names)
        if len(names) == 2:
            self._dim_names_in_2d = names
        elif len(names) == 3:
            self._dim_names_in_3d = names

    def add_axis(self, axis: str, values: np.ndarray) -> None:
        """
        Add 1D array specifying a spatial or temporal axis.
        """
        if axis in self._dim_names_in_3d or axis == 't':
            var = self._ncfile.createVariable(varname=axis,
                                              datatype='f8',
                                              dimensions=(axis))
            var[:] = values[:]

    def add_dataset(self, name: str, values: np.ndarray, dtype: str = 'f8', **kwargs) -> None:

        values = np.asarray(values)

        ti = kwargs.pop('time_index', 0)

        if self._ndims == 0:
            shape = np.shape(values)

            # add dimensions
            self._ncfile.createDimension(dimname="t", size=None)
            if len(shape) == 2:
                self._ncfile.createDimension(dimname=self._dim_names_in_2d[1], size=shape[0])
                self._ncfile.createDimension(dimname=self._dim_names_in_2d[0], size=shape[1])
                self._ndims = 2
            elif len(shape) == 3:
                self._ncfile.createDimension(dimname=self._dim_names_in_3d[2], size=shape[0])
                self._ncfile.createDimension(dimname=self._dim_names_in_3d[1], size=shape[1])
                self._ncfile.createDimension(dimname=self._dim_names_in_3d[0], size=shape[2])
                self._ndims = 3
            else:
                RuntimeError("Shape must be of 2 or 3 dimensions")


        if self._ndims == 2:
            if not name in self._ncfile.variables.keys():
                var = self._ncfile.createVariable(varname=name,
                                                  datatype=dtype,
                                                  dimensions=('t',
                                                              self._dim_names_in_2d[1],
                                                              self._dim_names_in_2d[0]))
            else:
                var = self._ncfile.variables[name]
            var[ti, :, :] = values[:, :]
        else:
            if not name in self._ncfile.variables.keys():
                var = self._ncfile.createVariable(varname=name,
                                                  datatype=dtype,
                                                  dimensions=('t',
                                                              self._dim_names_in_3d[2],
                                                              self._dim_names_in_3d[1],
                                                              self._dim_names_in_3d[0]))
            else:
                var = self._ncfile.variables[name]
            var[ti, :, :, :] = values[:, :, :]

        self._add_dataset_properties(var)


class ParcelWriter(NetcdfWriter):

    def open(self, filename: str):
        """
        Open a parcel netCDF file.
        """
        super()._open(filename, file_type=FileType.PARCELS)
        self._nparcels = 0
        self.time = 0.0

    def add_dataset(self, name: str, values: np.ndarray, dtype: str = 'f8', **kwargs) -> None:
        shape = np.shape(values)
        if len(shape) > 1:
            RuntimeError("Shape must be of 1-dimensional.")

        if self._nparcels == 0:
            self._nparcels = len(values)
            # add dimension
            self._ncfile.createDimension(dimname="t", size=None)
            self._ncfile.createDimension(dimname="n_parcels", size=self._nparcels)

            time = self._ncfile.createVariable(varname='t',
                                               datatype=dtype,
                                               dimensions=('t'))
            time[0] = self.time

        var = self._ncfile.createVariable(varname=name,
                                          datatype=dtype,
                                          dimensions=('t', 'n_parcels'))
        var[0, :] = values[:]

        self._add_dataset_properties(var)
