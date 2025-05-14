import netCDF4 as nc
from .dataset import Dataset
import numpy as np


class FieldDataset(Dataset):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

        self._derived_fields = list()

    def open(self, filename: str):
        super().open(filename)
        self.is_three_dimensional = (self.extent.size == 3)
        if self._verbose:
            print("Field dataset is 3-dimensional:", self.is_three_dimensional)

        v = super().variables
        if {'x_velocity', 'y_velocity', 'z_velocity'}.issubset(v):
            self._derived_fields.append('kinetic_energy')
        if {'x_vorticity', 'y_vorticity', 'z_vorticity'}.issubset(v):
            self._derived_fields.append('vorticity_magnitude')
            self._derived_fields.append('enstrophy')
        if {'x_velocity', 'x_vorticity',
            'y_velocity', 'y_vorticity',
            'z_velocity', 'z_vorticity'}.issubset(v):
            self._derived_fields.append('helicity')
            self._derived_fields.append('cross_helicity_magnitude')

    @property
    def variables(self) -> list:
        return super().variables + self._derived_fields

    def get_size(self) -> int:
        """
        Get dataset size (i.e. number of saved time frames).
        """
        return self._nc_handle.dimensions['t'].size

    def get_axis(self, name: str, copy_periodic: bool = True) -> np.ndarray:
        """
        Get grid point values of the x-axis, y-axis or z-axis.
        """
        if name not in ['x', 'y', 'z']:
            raise ValueError("No axis called '" + name + "'.")
        axis = np.array(self._nc_handle.variables[name])
        if copy_periodic and name in ['x', 'y']:
            axis = np.append(axis, abs(axis[0]))
        return axis

    def get_meshgrid(self, copy_periodic: bool = True):
        """
        Return grid points in a mesh.
        """
        x = self.get_axis('x', copy_periodic)
        y = self.get_axis('y', copy_periodic)
        z = self.get_axis('z', copy_periodic)

        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
        assert np.all(xg[:, 0, 0] == x)
        assert np.all(yg[0, :, 0] == y)
        assert np.all(zg[0, 0, :] == z)

        return xg, yg, zg

    def get_data(self,
                 name: str,
                 step: int,
                 indices: np.ndarray = None,
                 copy_periodic: bool = True) -> np.ndarray:
        """
        Return field data.
        """

        if name in self._derived_fields:
            return self._get_derived_dataset(name, step, copy_periodic)

        self.check_data(name, step)

        if indices is not None:
            return np.array(self._nc_handle.variables[name][step, ...]).squeeze()[indices, ...]
        else:
            fdata = np.array(self._nc_handle.variables[name][step, ...]).squeeze()

            if copy_periodic:
                fdata = self._copy_periodic_layers(fdata)

            if self.is_three_dimensional:
                # change ordering from (z, y, x) to (x, y, z)
                fdata = np.transpose(fdata, axes=[2, 1, 0])
            else:
                fdata = np.transpose(fdata, axes=[1, 0])

            return fdata

    def _get_derived_dataset(self, name: str, step: int, copy_periodic: bool) -> np.ndarray:
        if name == 'vorticity_magnitude':
            x_vor = self.get_data(name='x_vorticity', step=step, copy_periodic=copy_periodic)
            y_vor = self.get_data(name='y_vorticity', step=step, copy_periodic=copy_periodic)
            z_vor = self.get_data(name='z_vorticity', step=step, copy_periodic=copy_periodic)
            return np.sqrt(x_vor ** 2 + y_vor ** 2 + z_vor ** 2)
        if name == 'helicity':
            u = self.get_data(name='x_velocity', step=step, copy_periodic=copy_periodic)
            v = self.get_data(name='y_velocity', step=step, copy_periodic=copy_periodic)
            w = self.get_data(name='z_velocity', step=step, copy_periodic=copy_periodic)
            xi = self.get_data(name='x_vorticity', step=step, copy_periodic=copy_periodic)
            eta = self.get_data(name='y_vorticity', step=step, copy_periodic=copy_periodic)
            zeta = self.get_data(name='z_vorticity', step=step, copy_periodic=copy_periodic)
            return u * xi + v * eta + w * zeta
        if name == 'cross_helicity_magnitude':
            u = self.get_data(name='x_velocity', step=step, copy_periodic=copy_periodic)
            nx, ny, nz = u.shape
            uvec = np.zeros((nx, ny, nz, 3))
            ovec = np.zeros((nx, ny, nz, 3))
            uvec[:, :, :, 0] = u
            uvec[:, :, :, 1] = self.get_data(name='y_velocity',
                                             step=step,
                                             copy_periodic=copy_periodic)
            uvec[:, :, :, 2] = self.get_data(name='z_velocity',
                                             step=step,
                                             copy_periodic=copy_periodic)
            ovec[:, :, :, 0] = self.get_data(name='x_vorticity',
                                             step=step,
                                             copy_periodic=copy_periodic)
            ovec[:, :, :, 1] = self.get_data(name='y_vorticity',
                                             step=step,
                                             copy_periodic=copy_periodic)
            ovec[:, :, :, 2] = self.get_data(name='z_vorticity',
                                             step=step,
                                             copy_periodic=copy_periodic)
            ch = np.cross(uvec, ovec)
            x_ch = ch[:, :, :, 0]
            y_ch = ch[:, :, :, 1]
            z_ch = ch[:, :, :, 2]
            return np.sqrt(x_ch ** 2 + y_ch  ** 2 + z_ch ** 2)

        if name == 'kinetic_energy':
            u = self.get_data(name='x_velocity', step=step, copy_periodic=copy_periodic)
            v = self.get_data(name='y_velocity', step=step, copy_periodic=copy_periodic)
            w = self.get_data(name='z_velocity', step=step, copy_periodic=copy_periodic)
            return 0.5 * (u ** 2 + v ** 2 + w ** 2)
        if name == 'enstrophy':
            xi = self.get_data(name='x_vorticity', step=step, copy_periodic=copy_periodic)
            eta = self.get_data(name='y_vorticity', step=step, copy_periodic=copy_periodic)
            zeta = self.get_data(name='z_vorticity', step=step, copy_periodic=copy_periodic)
            return 0.5 * (xi ** 2 + eta ** 2 + zeta ** 2)

    def _copy_periodic_layers(self, field: np.ndarray) -> np.ndarray:
        if self.is_three_dimensional:
            nz, ny, nx = field.shape
            field_copy = np.empty((nz, ny+1, nx+1))
            field_copy[:, 0:ny, 0:nx] = field.copy()
            field_copy[:, ny, :] = field_copy[:, 0, :]
            field_copy[:, :, nx] = field_copy[:, :, 0]
        else:
            nz, nx = field.shape
            field_copy = np.empty((nz, nx+1))
            field_copy[:, 0:nx] = field.copy()
            field_copy[:, nx] = field_copy[:, 0]
        return field_copy
