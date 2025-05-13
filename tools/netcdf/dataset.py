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
        return self._nc_handle.isopen()

    def close(self) -> None:
        """
        Closes an open dataset.
        """
        if not self.is_open():
            return
        self._nc_handle.close()
        if self._verbose:
            print("Closed", self._filename)

    @property
    def extent(self) -> np.ndarray:
        """
        Get domain extent.
        """
        return self._nc_handle.getncattr("extent")

    @property
    def ncells(self) -> np.ndarray:
        """
        Get number of grid cells.
        """
        return  self._nc_handle.getncattr("ncells")

    @property
    def origin(self) -> np.ndarray:
        """
        Get domain origin (lower left corner).
        """
        return  self._nc_handle.getncattr("origin")

    @abc.abstractmethod
    def get_size(self) -> int:
        """
        Get dataset size (i.e. number of saved time frames).
        This function must be overriden by derived classes.
        """
        pass

    @property
    def file_type(self) -> FileType:
        """
        Get the file type of the open dataset.
        """
        return self._file_type

    @abc.abstractmethod
    def get_data(self, name: str, step: int, **kwargs) -> np.ndarray:
        """
        This function must be overriden by derived classes.
        """
        if not name in self._nc_handle.variables.keys():
            raise IOError("Dataset '" + name + "' unknown.")

        n_steps = self.get_size()
        if step > n_steps - 1:
            raise ValueError("Dataset has only steps 0 to " + str(n_steps - 1) + ".")

    def _has_global_attributes(self):
        return not self._nc_handle.ncattrs() == []

    # 18 Feb 2022
    # https://stackoverflow.com/questions/8450472/how-to-print-a-string-at-a-fixed-width
    # 19 Feb 2022
    # https://stackoverflow.com/questions/873327/pythons-most-efficient-way-to-choose-longest-string-in-list
    def __str__(self):
        """
        Print information about the dataset.
        """
        if self._has_global_attributes():
            print("=" * 80)
            # print global attributes
            print("GLOBAL ATTRIBUTES:")
            l = len(max(self._nc_handle.ncattrs(), key=len))
            fmt = '{0: <' + str(l) + '}'
            for key in self._nc_handle.ncattrs():
                print(fmt.format(key), "\t", self._nc_handle.getncattr(key))
            print("-" * 80)

        print("DIMENSIONS:")

        for dim in self._nc_handle.dimensions:
            print("    ", dim, "=", self._nc_handle.dimensions[dim].size)

        print("-" * 80)

        print("VARIABLES:")
        # get first variable name
        name = list(self._nc_handle.variables.keys())[0]

        if not self._nc_handle.variables[name].ncattrs() == []:
            # get length of longest attribute string
            l = len(max(self._nc_handle.variables[name].ncattrs(), key=len))
            fmt = '{0: <' + str(l) + '}'

        # print variables and their attributes
        for var in self._nc_handle.variables:
            print("    ", var)
            for attr in self._nc_handle.variables[var].ncattrs():
                print("\t", fmt.format(attr), "\t", self._nc_handle.variables[var].getncattr(attr))
        print("=" * 80)
        return ""
