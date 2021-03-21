What is PENSA?
==============

PENSA is a collection of python methods for exploratory analysis and comparison of protein structural ensembles, e.g., from molecular dynamics simulations.

With PENSA, you can (currently):

- **compare structural ensembles** of proteins via the relative entropy of their features, statistical tests, or state-specific information and visualize deviations on a reference structure.
- project several ensembles on a **joint reduced representation** using principal component analysis (PCA) or time-lagged independent component analysis (tICA) and sort the structures along the obtained components.
- **cluster structures across ensembles** via k-means or regular-space clustering and write out the resulting clusters as trajectories.
- trace allosteric information flow through a protein using **state-specific information** analysis methods.

All functionality is available as a python package (installation see below). For the most common applications, example `python scripts <https://github.com/drorlab/pensa/tree/master/scripts>`_ are provided. 

To make yourself familiar with PENSA's functionality, see the `tutorial <https://github.com/drorlab/pensa/tree/master/tutorial>`_.

