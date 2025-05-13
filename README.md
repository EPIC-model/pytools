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
