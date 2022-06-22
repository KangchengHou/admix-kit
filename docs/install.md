# Installation

```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit
pip install -r requirements.txt; pip install -e .
```

## Update to latest
To update to the latest version, run the following
```bash
# reinstalling these dependencies because these are constantly being updated
pip uninstall -y pgenlib
pip install -U git+https://github.com/bogdanlab/tinygwas.git#egg=tinygwas
pip install -U git+https://github.com/KangchengHou/dask-pgen.git#egg=dask-pgen
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit & pip install -e .
```

## Notes
1. Installation requires cmake version > 3.12. Use `cmake --version` to check your cmake version. Use `pip install cmake` to install the latest version.

