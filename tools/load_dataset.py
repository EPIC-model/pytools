from . import netcdf as nc
from .netcdf.field_dataset import FieldDataset
from .netcdf.parcel_dataset import ParcelDataset
from .netcdf.stat_dataset import StatDataset
from .netcdf.dataset import check_file_type, FileType

def load_dataset(filename: str, verbose: bool = True) -> FieldDataset | ParcelDataset | StatDataset:
    file_type = check_file_type(filename)

    match file_type:
        case FileType.PARCELS:
            dset = ParcelDataset(verbose)
        case FileType.FIELDS:
            dset = FieldDataset(verbose)
        case FileType.FIELD_STATS:
            dset = StatDataset(verbose)
        case FileType.PARCEL_STATS:
            dset = StatDataset(verbose)
        case FileType.NONE:
            raise RuntimeError("No such file type.")

    dset.open(filename)
    return dset
