# Python API

Import admix-kit as:
```python
import admix
```

## Association testing

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

   admix.assoc.marginal
```

## Genetic correlation across local ancestry

```{note}
For genetic correlation analyis, because of involvement of multiple steps, it is recommended to use the command line interface `admix genet-cor` to perform the analysis. See [here](https://kangchenghou.github.io/admix-kit/cli/genet-cor.html) for details.
```


```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

   admix.cli.admix_grm
   admix.cli.admix_grm_merge
   admix.cli.genet_cor
```

## Polygenic scoring
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

   admix.data.calc_partial_pgs
```

## Data structures: `admix.Dataset`

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

   admix.Dataset
```

## Simulation: `admix.simulate`

```{eval-rst}
.. autosummary::
   :nosignatures:
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
   :nosignatures:
   :toctree: _api

   admix.data.allele_per_anc
   admix.data.af_per_anc
```

## Input and output: `admix.io`

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

   admix.io.read_dataset
   admix.io.read_lanc
   admix.io.read_rfmix
```

## External tools: `admix.tools`
    
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

    admix.tools.plink2.lift_over
```

## Plot: `admix.plot`

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: _api

    admix.plot.lanc
    admix.plot.admixture
    admix.plot.manhattan
    admix.plot.qq
```