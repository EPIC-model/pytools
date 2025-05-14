import netCDF4 as nc
from .dataset import Dataset
import numpy as np


class StatDataset(Dataset):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)


    def get_size(self) -> int:
        """
        Get dataset size (i.e. number of saved time frames).
        """
        return self._nc_handle.dimensions['t'].size

    def get_data(self,
                 name: str,
                 step: int) -> np.ndarray | float:
        """
        Return statistics data.
        If step = -1, this function returns all the data.
        """
        super().check_data(name, step=max(0, step))

        data = np.array(self._nc_handle.variables[name])
        if step > -1:
            data = data[step]

        return data
