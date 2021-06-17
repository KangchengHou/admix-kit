"""
Data structures
===============

We introduce the central data stuctures used in this package.

These data will be of shape

#. genotype (``n_snp``, ``n_indiv``, ``n_haploid``)
#. local ancestry (``n_snp``, ``n_indiv``, ``n_haploid``)
#. information about SNPs (``n_snp``, ``n_snp_feature``)
#. information about individuals (``n_indiv``, ``n_indiv_feature``)

We make use of ``xarray`` to deal with all the data in an unified way, 
specifically, we will have an xarray.Dataset to store these. Below is an illustration 
of an example dataset.

Features about SNPs should end with "@snp", whereas features about individuals should
end with "@indiv" to differentiate the variables belonging to the SNP axis and variables 
that belonging to the individual axis.


.. note::
    In the future, we may implement a class called ``admix.Dataset`` for more convenient
    handling of these. But we will use the direct interface to xarray before our interface
    is stable enough.

    Some operations we aim to support: 
"""

import xarray as xr

# Genotype can be accessed using ``.geno``
# Local ancestry can be accessed using ``.lanc``
#
