.. module:: admix
.. automodule:: admix
   :noindex:

Import admix-tools as::

   import admix

Data structures: `Dataset`
--------------------------

.. module:: admix.dataset
.. currentmodule:: admix

.. autosummary:: 
   :toctree: _api

   dataset.Dataset


Association testing: `assoc`
----------------------------

.. module:: admix.assoc
.. currentmodule:: admix

.. autosummary:: 
   :toctree: _api

   assoc.marginal

Simulation: `simulate`
----------------------

.. module:: admix.simulate
.. currentmodule:: admix

.. autosummary::
   :toctree: _api

    simulate.admix_geno
    simulate.quant_pheno
    simulate.binary_pheno
    simulate.quant_pheno_1pop
    simulate.quant_pheno_grm

Data management: `data`
------------------------

.. module:: admix.data
.. currentmodule:: admix

.. autosummary::
   :toctree: _api

   data.allele_per_anc
   data.af_per_anc


Tools: `tools`
--------------
.. module:: admix.tools
.. currentmodule:: admix

.. autosummary::
   :toctree: _api

    tools.plink.run
    tools.plink2.run
   

Plot: `plot`
--------------
.. module:: admix.plot
.. currentmodule:: admix

.. autosummary::
   :toctree: _api

    plot.lanc
    plot.admixture
    plot.manhattan
    plot.qq