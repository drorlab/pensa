What is PENSA?
==============


PENSA is a collection of python methods for exploratory analysis and comparison of biomolecular conformational ensembles, e.g., from molecular dynamics simulations.


Functionality
*************

With PENSA, you can (currently):

- **compare structural ensembles** of proteins via the relative entropy of their features, statistical tests, or state-specific information and visualize deviations on a reference structure.
- project several ensembles on a **joint reduced representation** using principal component analysis (PCA) or time-lagged independent component analysis (tICA) and sort the structures along the obtained components.
- **cluster structures across ensembles** via k-means or regular-space clustering and write out the resulting clusters as trajectories.
- trace allosteric information flow through a protein using **state-specific information** analysis methods.

Biomolecules can be featurized using backbone torsions, sidechain torsions, or arbitrary distances (e.g., between all backbone C-alpha atoms). 
We also provide density-based methods to featurize water and ion pockets as well as a featurizer for hydrogen bonds. 
The library is modular so you can easily write your own feature reader.

PENSA also includes trajectory processing tools based on MDAnalysis and plotting functions using Matplotlib.

All functionality is available as a python package. 


Citation
********

If you publish about work for which PENSA was useful, please cite our preprint:

    M. Vögele, N. J. Thomson, S. T. Truong, J. McAvity, U. Zachariae, R. O. Dror:
    *Systematic Analysis of Biomolecular Conformational Ensembles with PENSA*.
    `arXiv:2212.02714 [q-bio.BM] 2022 <https://arxiv.org/abs/2212.02714>`_.

The reference for the software implementation itself is the following:

    Martin Vögele, Neil Thomson, Sang Truong, Jasper McAvity: *PENSA*. Zenodo, 2024. http://doi.org/10.5281/zenodo.4362136

To get the citation and DOI for a particular version, see `Zenodo <https://zenodo.org/record/4362136>`_.
