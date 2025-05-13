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

# How to access data in a dataset
After loading a dataset, you can access data with
```python
dset.get_data(name, step)
```
where `name` is the data name, e.g. `z_vorticity`, and `step` is an integer. A list of
available data is printed when you type `print(dset)`. Note that `get_data` accepts
further options like `copy_periodic` for field datasets. For further info on datasets
check the corresponding classes (e.g. `help(dset)`).
