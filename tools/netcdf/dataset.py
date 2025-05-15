import abc
from enum import StrEnum
import netCDF4 as nc
import numpy as np
import os

class FileType(StrEnum):
    PARCELS = "parcels"
    FIELDS = "fields"
    FIELD_STATS = "field_stats"
    PARCEL_STATS = "parcel_stats"
    NONE = "none"


def check_file_type(filename: str) -> FileType:
    try:
        ncfile = nc.Dataset(filename, "r", format="NETCDF4")
        file_type = FileType(ncfile.getncattr('file_type'))
        ncfile.close()
    except:
        file_type = FileType.NONE
    return file_type


class Dataset(abc.ABC):

    def __init__(self, verbose: bool = False):
        """
        Abstract base class for netCDF datasets.
        """
        self._nc_handle = None
        self._file_type = FileType.NONE
        self._verbose = verbose
        self._filename = str()

    def open(self, filename: str) -> None:
        """
        Open a dataset.
        """
        if self._nc_handle is not None:
            if self.is_open():
                if self._verbose:
                    print("Dataset already open.")
                return

        if not os.path.exists(filename):
            raise IOError("File '" + filename + "' does not exist.")
        self._nc_handle = nc.Dataset(filename, "r", format="NETCDF4")
        self._file_type = FileType(self._nc_handle.getncattr('file_type'))
        self._filename = filename
        if self._verbose:
            print("Opened", self._filename)

    def is_open(self) -> bool:
        """
        Check if dataset is open.
        """
        if self._nc_handle is None:
            return False
        return self._nc_handle.isopen()

    def close(self) -> None:
        """
        Closes an open dataset.
        """
        if not self.is_open():
            return
        self._nc_handle.close()
        self._nc_handle = None
        self._file_type = FileType.NONE
        self._filename = str()
        if self._verbose:
            print("Closed", self._filename)

    @property
    def variables(self) -> list:
        """
        Return a list of available variables in a dataset.
        """
        return list(self._nc_handle.variables.keys())

    @property
    def extent(self) -> np.ndarray:
        """
        Get domain extent.
        """
        try:
            res = self._nc_handle.getncattr("extent")
        except:
            res = self._nc_handle['parameters'].getncattr('extent')
        return res

    @property
    def ncells(self) -> np.ndarray:
        """
        Get number of grid cells.
        """
        try:
            res = self._nc_handle.getncattr("ncells")
        except:
            res = self._nc_handle['parameters'].getncattr('ncells')
        return res

    @property
    def origin(self) -> np.ndarray:
        """
        Get domain origin (lower left corner).
        """
        try:
            res = self._nc_handle.getncattr("origin")
        except:
            res = self._nc_handle['parameters'].getncattr('origin')
        return res

    def get_label(self, varname) -> str:
        """
        Return variable label for plotting.
        """
        return  self._nc_handle.variables[varname].long_name

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get dataset size.
        This function must be overriden by derived classes.
        """
        pass

    @property
    def first_step(self) -> int:
        """
        Return first valid step number.
        """
        return 0

    @property
    def last_step(self) -> int:
        """
        Return last valid step number.
        """
        return self.get_size() - 1

    @property
    def file_type(self) -> FileType:
        """
        Get the file type of the open dataset.
        """
        return self._file_type

    @abc.abstractmethod
    def get_data(self, varname: str, step: int, **kwargs) -> np.ndarray:
        """
        This function must be overriden by derived classes.
        """
        pass

    def check_data(self, varname: str, step: int) -> None:
        """
        This function checks if data is available.
        """
        if not varname in self.variables:
            raise IOError("Dataset '" + varname + "' unknown.")

        if step < self.first_step or step > self.last_step:
            msg = "Dataset has only steps " + str(self.first_step) + " to " + str(self.last_step) + "."
            raise ValueError(msg)

    def info(self):
        """
        Print general information about dataset.

        18 Feb 2022
        https://stackoverflow.com/questions/8450472/how-to-print-a-string-at-a-fixed-width
        19 Feb 2022
        https://stackoverflow.com/questions/873327/pythons-most-efficient-way-to-choose-longest-string-in-list
        """
        if self._nc_handle.ncattrs():
            print("GLOBAL ATTRIBUTES:")
            print("-" * 18)
            l = len(max(self._nc_handle.ncattrs(), key=len))
            fmt = '{0: <' + str(l) + '}'
            for key in self._nc_handle.ncattrs():
                print(fmt.format(key), "\t", self._nc_handle.getncattr(key))

    def __str__(self):
        """
        Print information about the dataset.
        """
        print("DIMENSIONS:")
        print("-" * 11)
        for dim in self._nc_handle.dimensions:
            print("    ", dim, "=", self._nc_handle.dimensions[dim].size)

        print("VARIABLES:")
        print("-" * 10)
        for var in self.variables:
            print("    ", var)
        return ""
