# Installation

```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit
pip install -r requirements.txt; pip install -e .
```

### Update to latest (admix-kit only)
```
# go to the path of admix-kit
cd /path/to/admix-kit
git pull
pip install -e .
```

### Update to latest (admix-kit and other dependencies)
```{note}
Try update "admix-kit only" described above first before updating other dependencies.
```

```bash
# reinstalling these dependencies because these are constantly being updated
pip uninstall -y pgenlib
pip install -U git+https://github.com/bogdanlab/tinygwas.git#egg=tinygwas
pip install -U git+https://github.com/KangchengHou/dask-pgen.git#egg=dask-pgen
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit & pip install -e .
```