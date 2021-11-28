Association testing in admixed population
-----------------------------------------


Basics of association testing
=============================
Suppose we are coping with :math:`y = X \beta + C \alpha + e` where 
:math:`y, x` are the phenotypes and genotypes at a specific SNP. :math:`C` are the 
covariates and :math:`\alpha` is the vector of coefficients of the covariates, while 
:math:`e` is the environmental noise.

GWAS aims to test :math:`\beta = 0` for each SNP. One can use ANOVA to perform an F-test.
An equivalent way is to project both sides onto the space orthogonal to :math:`C`. 
Specifically, let :math:`P = C(C^\top C)^{-1}C^\top`, we multiply :math:`I - P` to 
both sides of the equation and obtain :math:`(I - P) y = (I - P) X \beta + (I - P) e`. 
Now we can just test :math:`\beta` in a linear model with only :math:`(I - P)X` as the
variables and perform an overall F-test.


.. math::
    C^\dagger

See `this post <https://stats.stackexchange.com/questions/258461/proof-that-f-statistic-follows-f-distribution>`_.
for more details.

Example of conditioning on local ancestry loses power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Consider a two-way admixture and the allele frequencies in the two ancestral populations 
are :math:`p_1=0.01, p_2=0.99`. If this SNP is causal, we expect an admixture signal:
proportion of cases are differentiated for individuals with different local ancestries.
Also, an association testing without local ancestry correction will produce strong 
association signal. However, if we alternatively condition on the local ancestry, we 
will be testing the association within each local ancestry background 1,2 respectively.
Since there is little variation in each of the population, we will hardly observe 
association signal.

TODO: does it always lose power?

Example of local ancestry bring new information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
