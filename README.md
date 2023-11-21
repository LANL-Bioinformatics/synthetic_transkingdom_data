# synthetic_transkingdom_data
Code to generate synthetic data from log-normal distribution of taxa including multiple kingdoms

This code was prepared to demonstrate the difficulties in network inference when using data from two paired data types (e.g. 16S for bacteria and ITS for fungi in a single sample). The basic functionality is to generate synthetic log-normal distributed "absolute" abundances of samples and simulate the sequencing process with those samples after (possibly) splitting them into two or more groups. 

Networks in the results folder were constructed using simple approaches (log-covariance and centered log ratio) for which code is provided, as well as SparCC (see https://doi.org/10.1371/journal.pcbi.1002687) using SparCC3 (https://github.com/JCSzamosi/SparCC3) as well as the Gaussian LASSO approach (see https://doi.org/10.1371/journal.pcbi.1004226) (exact code used unavailable at this time).

The jupyter notebook contains a tutorial for generating synthetic data.
