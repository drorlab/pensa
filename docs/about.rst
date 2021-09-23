What is PENSA?
==============

PENSA is a collection of python methods for exploratory analysis and comparison of protein structural ensembles, e.g., from molecular dynamics simulations.

Functionality
*************

With PENSA, you can (currently):

- **compare structural ensembles** of proteins via the relative entropy of their features, statistical tests, or state-specific information and visualize deviations on a reference structure.
- project several ensembles on a **joint reduced representation** using principal component analysis (PCA) or time-lagged independent component analysis (tICA) and sort the structures along the obtained components.
- **cluster structures across ensembles** via k-means or regular-space clustering and write out the resulting clusters as trajectories.
- trace allosteric information flow through a protein using **state-specific information** analysis methods.

All functionality is available as a python package. For the most common applications, example `python scripts <https://github.com/drorlab/pensa/tree/master/scripts>`_ are provided. 

To make yourself familiar with PENSA's functionality, see the `tutorial <https://pensa.readthedocs.io/en/latest/tut-1-intro.html>`_.

Citation
********

If you publish about work for which PENSA was useful, please cite it accordingly.

The general citation, representing the "concept" of the software is the following:

    Martin Vögele, Neil Thomson, Sang Truong, Jasper McAvity. (2021). PENSA. Zenodo. http://doi.org/10.5281/zenodo.4362136

To get the citation and DOI for a particular version, see `Zenodo <https://zenodo.org/record/4362136>`_.
