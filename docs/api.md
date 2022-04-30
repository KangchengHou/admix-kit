# Python API

We introduce the following main functions:

Import admix-tools as:
```python
import admix
```

## Data structures: `admix.Dataset`

```{eval-rst}
.. autosummary:: 
   :toctree: _api

   admix.Dataset
```

## Association testing: `admix.assoc`

```{eval-rst}
.. autosummary:: 
   :toctree: _api

   admix.assoc.marginal
```

## Simulation: `admix.simulate`

```{eval-rst}
.. autosummary::
   :toctree: _api

    admix.simulate.admix_geno
    admix.simulate.quant_pheno
    admix.simulate.binary_pheno
    admix.simulate.quant_pheno_1pop
    admix.simulate.quant_pheno_grm
```

## Data management: `admix.data`

```{eval-rst}
.. autosummary::
   :toctree: _api

   admix.data.allele_per_anc
   admix.data.af_per_anc
```

## Input and output: `admix.io`

```{eval-rst}
.. autosummary::
   :toctree: _api

   admix.io.read_dataset
   admix.io.read_lanc
   admix.io.read_rfmix
```

## External tools: `admix.tools`
    
```{eval-rst}
.. autosummary::
   :toctree: _api

    admix.tools.plink2.lift_over
```

## Plot: `admix.plot`

```{eval-rst}
.. autosummary::
   :toctree: _api

    admix.plot.lanc
    admix.plot.admixture
    admix.plot.manhattan
    admix.plot.qq
```