# Python tools for reading and plotting EPIC data

## How to set up your Python environment
After downloading and installing [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main),
you can install all required Python packages into a new Python environment with either
```bash
conda config --add channels conda-forge
conda create --name pytools python=3.13.3
conda install conda-forge::numpy
conda install conda-forge::scipy
conda install conda-forge::matplotlib
conda install conda-forge::netcdf4
conda install conda-forge::colorcet
```
or
```
conda config --add channels conda-forge
conda create --name pytools python=3.13.3 --file requirements.txt
```
where `pytools` will be the name of the environment. It is recommended to use the latest Python version.
You may therefore need to change `python=3.13.3`. The Python environment is activated with
```bash
conda activate pytools
```
and deactivated with
```
conda deactivate
```

* [loading datasets](docs/datasets.md)
* [plotting](docs/plotting.md)
* [writing EPIC input files](docs/netcdf_writer.md)

