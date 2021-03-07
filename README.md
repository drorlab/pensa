# PENSA - Protein Ensemble Analysis

![Package](https://github.com/drorlab/pensa/workflows/package/badge.svg)
[![Documentation
Status](https://readthedocs.org/projects/pensa/badge/?version=latest)](http://pensa.readthedocs.io/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pensa.svg)](https://badge.fury.io/py/pensa)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4362136.svg)](https://doi.org/10.5281/zenodo.4362136)

A collection of Python methods for exploratory analysis and comparison of protein structural ensembles, e.g., from molecular dynamics simulations.
All functionality is available as a Python package. For the most common applications, example [Python scripts](https://github.com/drorlab/pensa/tree/master/scripts) are provided. 

To get started, see the [documentation](https://pensa.readthedocs.io/en/latest/) and the [tutorials](https://github.com/drorlab/pensa/tree/master/tutorial).

## Functionality

With PENSA, you can (currently):
- Compare structural ensembles of proteins via the relative entropy of their features, statistical tests, or state-specific information and visualize deviations on a reference structure.
- Project several ensembles on a joint reduced representations using principal component analysis (PCA) or time-lagged independent component analysis (tICA) and sort the structures along the obtained components.
- Cluster structures via k-means or regular-space clustering and write out the resulting clusters as trajectories.

We also provide the first easily applicable implementation of state-specific information analysis methods.

Proteins are featurized via [PyEMMA](http://emma-project.org/latest/) using backbone torsions, sidechain torsions, or backbone C-alpha distances, making PENSA compatible to all functionality available in PyEMMA. In addition, we provide methods to featurize water pockets.

Trajectories are processed and written using [MDAnalysis](https://www.mdanalysis.org/). Plots are generated using [Matplotlib](https://matplotlib.org/).

## Documentation
PENSA's documentation pages are [here](https://pensa.readthedocs.io/en/latest/). 
You can also refer to the [tutorial](https://github.com/drorlab/pensa/tree/master/tutorial) and the installation instructions below.

#### Demo on Google Colab
We demonstrate how to use the PENSA library in an interactive and animated example on Google Colab, where we use freely available simulations of a mu-Opioid Receptor from GPCRmd.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1difJjlcwpN-0hSmGCGrPq9Cxq5wJ7ZDa)


## Citation

General:
```
Martin Vögele. PENSA. http://doi.org/10.5281/zenodo.4362136
```

To get the citation and DOI for a particular version, see [Zenodo](https://zenodo.org/record/4362136):


## Acknowledgments

#### Contributors
Martin Vögele, Neil Thomson, Sang Truong

#### Beta-Tests
Alex Powers, Lukas Stelzl

#### Funding & Support 
This project was started by Martin Vögele at Stanford University, supported by an EMBO long-term fellowship (ALTF 235-2019), as part of the INCITE computing project 'Enabling the Design of Drugs that Achieve Good Effects Without Bad Ones' (BIP152).


