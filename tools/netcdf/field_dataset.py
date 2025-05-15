import netCDF4 as nc
from .dataset import Dataset
import numpy as np


class FieldDataset(Dataset):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

        self._derived_fields = {}

    def open(self, filename: str):
        super().open(filename)
        self.is_three_dimensional = (self.extent.size == 3)
        if self._verbose:
            print("Field dataset is 3-dimensional:", self.is_three_dimensional)

        v = super().variables
        if {'x_velocity', 'y_velocity', 'z_velocity'}.issubset(v) and 'kinetic_energy' not in v:
            self._derived_fields['kinetic_energy'] = self._kinetic_energy
        if {'x_vorticity', 'y_vorticity', 'z_vorticity'}.issubset(v):
            if 'vorticity_magnitude' not in v:
                self._derived_fields['vorticity_magnitude'] = self._vorticity_magnitude
            if 'enstrophy' not in v:
                self._derived_fields['enstrophy'] = self._enstrophy
        if {'x_velocity', 'x_vorticity',
            'y_velocity', 'y_vorticity',
            'z_velocity', 'z_vorticity'}.issubset(v):
            if 'helicity' not in v:
                self._derived_fields['helicity'] = self._helicity
            if 'cross_helicity_magnitude' not in v:
                self._derived_fields['cross_helicity_magnitude'] = self._cross_helicity_magnitude

    @property
    def variables(self) -> list:
        return super().variables + list(self._derived_fields.keys())

    def get_size(self) -> int:
        """
        Get dataset size (i.e. number of saved time frames).
        """
        return self._nc_handle.dimensions['t'].size

    def get_axis(self, name: str, copy_periodic: set[str]) -> np.ndarray:
        """
        Get grid point values of the x-axis, y-axis or z-axis.
        """
        if name not in ['x', 'y', 'z']:
            raise ValueError("No axis called '" + name + "'.")
        axis = np.array(self._nc_handle.variables[name])
        if name in copy_periodic and name in ['x', 'y']:
            axis = np.append(axis, abs(axis[0]))
        return axis

    def get_meshgrid(self, copy_periodic: set[str]):
        """
        Return grid points in a mesh.
        """
        x = self.get_axis('x', copy_periodic)
        y = self.get_axis('y', copy_periodic)
        z = self.get_axis('z')

        xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
        assert np.all(xg[:, 0, 0] == x)
        assert np.all(yg[0, :, 0] == y)
        assert np.all(zg[0, 0, :] == z)

        return xg, yg, zg

    def get_data(self,
                 name: str,
                 step: int,
                 copy_periodic: set[str]) -> np.ndarray:
        """
        Return field data.
        """
        self.check_data(name, step)

        if name in self._derived_fields.keys():
            return self._derived_fields[name](step, copy_periodic)
        else:
            data = np.array(self._nc_handle.variables[name][step, ...])

            if copy_periodic:
                data = self._copy_periodic_layers(copy_periodic, data)

            if self.is_three_dimensional:
                # change ordering from (z, y, x) to (x, y, z)
                data = np.transpose(data, axes=[2, 1, 0])
            else:
                data = np.transpose(data, axes=[1, 0])

            return data

    def _copy_x_periodic(self, field: np.ndarray) -> np.ndarray:
        if self.is_three_dimensional:
            nz, ny, nx = field.shape
            field_copy = np.empty((nz, ny, nx+1))
            field_copy[:, :, 0:nx] = field.copy()
            field_copy[:, :, nx] = field_copy[:, :, 0]
        else:
            nz, nx = field.shape
            field_copy = np.empty((nz, nx+1))
            field_copy[:, 0:nx] = field.copy()
            field_copy[:, nx] = field_copy[:, 0]
        return field_copy

    def _copy_y_periodic(self, field: np.ndarray) -> np.ndarray:
        if self.is_three_dimensional:
            nz, ny, nx = field.shape
            field_copy = np.empty((nz, ny+1, nx))
            field_copy[:, 0:ny, :] = field.copy()
            field_copy[:, ny, :] = field_copy[:, 0, :]
        return field_copy

    def _copy_periodic_layers(self, copy_periodic: set[str], field: np.ndarray) -> np.ndarray:
        if copy_periodic == {'x'}:
            return self._copy_x_periodic(field)
        elif copy_periodic == {'y'}:
            return self._copy_y_periodic(field)
        elif copy_periodic == {'x', 'y'}:
            if self.is_three_dimensional:
                nz, ny, nx = field.shape
                field_copy = np.empty((nz, ny, nx))
                field_copy[:, 0:ny, 0:nx] = field.copy()
                field_copy[:, ny, :] = field_copy[:, 0, :]
                field_copy[:, :, nx] = field_copy[:, :, 0]
            else:
                nz, nx = field.shape
                field_copy = np.empty((nz, nx+1))
                field_copy[:, 0:nx] = field.copy()
                field_copy[:, nx] = field_copy[:, 0]
            return field_copy
        return field

    def _vorticity_magnitude(self, step: int, copy_periodic: set[str]) -> np.ndarray:
        xi = self.get_data(name='x_vorticity', step=step, copy_periodic=copy_periodic)
        eta = self.get_data(name='y_vorticity', step=step, copy_periodic=copy_periodic)
        zeta = self.get_data(name='z_vorticity', step=step, copy_periodic=copy_periodic)
        return np.sqrt(xi ** 2 + eta ** 2 + zeta ** 2)

    def _helicity(self, step: int, copy_periodic: set[str]) -> np.ndarray:
        u = self.get_data(name='x_velocity', step=step, copy_periodic=copy_periodic)
        v = self.get_data(name='y_velocity', step=step, copy_periodic=copy_periodic)
        w = self.get_data(name='z_velocity', step=step, copy_periodic=copy_periodic)
        xi = self.get_data(name='x_vorticity', step=step, copy_periodic=copy_periodic)
        eta = self.get_data(name='y_vorticity', step=step, copy_periodic=copy_periodic)
        zeta = self.get_data(name='z_vorticity', step=step, copy_periodic=copy_periodic)
        return u * xi + v * eta + w * zeta

    def _cross_helicity_magnitude(self, step: int, copy_periodic: set[str]) -> np.ndarray:
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

    def _kinetic_energy(self, step: int, copy_periodic: set[str]) -> np.ndarray:
        u = self.get_data(name='x_velocity', step=step, copy_periodic=copy_periodic)
        v = self.get_data(name='y_velocity', step=step, copy_periodic=copy_periodic)
        w = self.get_data(name='z_velocity', step=step, copy_periodic=copy_periodic)
        return 0.5 * (u ** 2 + v ** 2 + w ** 2)

    def _enstrophy(self, step: int, copy_periodic: set[str]) -> np.ndarray:
        xi = self.get_data(name='x_vorticity', step=step, copy_periodic=copy_periodic)
        eta = self.get_data(name='y_vorticity', step=step, copy_periodic=copy_periodic)
        zeta = self.get_data(name='z_vorticity', step=step, copy_periodic=copy_periodic)
        return 0.5 * (xi ** 2 + eta ** 2 + zeta ** 2)
