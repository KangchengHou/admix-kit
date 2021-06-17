Association testing in admixed population
-----------------------------------------


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
