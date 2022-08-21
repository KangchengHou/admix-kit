# Installation

## Install latest version
```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
pip install git+https://github.com/kangchenghou/admix-kit
```

## Hacking on `admix-kit`

```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit && pip install -e .
```


### Update to latest (admix-kit and other dependencies)

```{note}
The following will be rarely used.
```

```bash
# reinstalling these dependencies because these are constantly being updated
pip uninstall -y pgenlib
pip install -U git+https://github.com/bogdanlab/tinygwas.git#egg=tinygwas
pip install -U git+https://github.com/KangchengHou/dask-pgen.git#egg=dask-pgen
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit & pip install -e .
```