import netCDF4 as nc
from .dataset import Dataset
import re
import numpy as np
import os

class ParcelDataset(Dataset):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def open(self, filename: str):
        super().open(filename)
        self.is_three_dimensional = (self.extent.size == 3)
        self._loaded_step = -1
        self._n_parcel_files = 0

        basename = os.path.basename(filename)
        # 14 Feb 2022
        # https://stackoverflow.com/questions/15340582/python-extract-pattern-matches
        p = re.compile(r"(.*)_(\d*)_parcels.nc")
        result = p.search(basename)
        self._basename = result.group(1)
        self._first_step = int(result.group(2).lstrip("0"))
        self._dirname = os.path.dirname(filename)
        if self._dirname == '':
            self._dirname = '.'
        for ff in os.listdir(self._dirname):
            if ff.startswith(self._basename) and ff.endswith('_parcels.nc'):
                self._n_parcel_files += 1
                result = p.match(ff)
                index = int(result.group(2).lstrip("0"))
                self._first_step = min(self._first_step, index)
        self._last_step = self._first_step + self._n_parcel_files - 1
        if self._verbose:
            print("First and last parcel file step:", self._first_step, self._last_step)

        for step in range(self._first_step, self._last_step+1):
            if not os.path.exists(self._get_filename(step)):
                raise RuntimeError("Parcel file numbers not consecutive.")

        self._load_step(self._first_step)


    def get_size(self) -> int:
        """
        Get the number of parcels of currently loaded step.
        """
        return self._nc_handle.dimensions['n_parcels'].size

    @property
    def first_step(self) -> int:
        """
        Return first valid step number.
        """
        return self._first_step

    @property
    def last_step(self) -> int:
        """
        Return last valid step number.
        """
        return self._last_step

    def get_data(self, varname: str, step: int, indices: np.ndarray = None) -> np.ndarray:
        """
        Get parcel attribute data.
        """
        super().check_data(varname, step)

        self._load_step(step)

        data = np.array(self._nc_handle.variables[varname]).squeeze()

        if indices is not None:
            return data[indices, ...]
        else:
            return data

    def _get_step_string(self, step: int) -> str:
        return str(step).zfill(10)

    def _get_filename(self, step: int) -> str:
        s = self._get_step_string(step)
        return os.path.join(self._dirname, self._basename + '_' + s + '_parcels.nc')

    def _load_step(self, step: int) -> None:
        """
        Load data of the next parcel file.
        """
        if step < self._first_step or step > self._last_step:
            raise RuntimeError("Step number outside bounds.")

        if self._loaded_step == step:
            return
        if self._verbose:
            print("Loading step", step)
        self._loaded_step = step
        self._nc_handle.close()
        filename = self._get_filename(step)
        self._nc_handle = nc.Dataset(filename, "r", format="NETCDF4")
