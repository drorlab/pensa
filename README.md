# PENSA - Protein Ensemble Analysis

![Package](https://github.com/drorlab/pensa/workflows/package/badge.svg)
[![Documentation
Status](https://readthedocs.org/projects/pensa/badge/?version=latest)](http://pensa.readthedocs.io/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pensa.svg)](https://badge.fury.io/py/pensa)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4362136.svg)](https://doi.org/10.5281/zenodo.4362136)

A collection of python methods for exploratory analysis and comparison of protein structural ensembles, e.g., from molecular dynamics simulations.

With PENSA, you can (currently):
- compare structural ensembles of proteins via the relative entropy of their features and visualize deviations on a reference structure.
- project ensembles on their combined principal components (PCs) and sort the structures along a PC.
- cluster structures via k-means and via regular-space clustering and write out the resulting clusters as trajectories.

Proteins are featurized via [PyEMMA](http://emma-project.org/latest/) using backbone torsions, sidechain torsions, or backbone C-alpha distances, making PENSA compatible to all functionality available in PyEMMA. Trajectories are processed and written using [MDAnalysis](https://www.mdanalysis.org/). Plots are generated using [Matplotlib](https://matplotlib.org/). 

All functionality is available as a python package (installation see below). For the most common applications, example [python scripts](https://github.com/drorlab/pensa/tree/master/scripts) are provided. To get started, see the [tutorial](https://github.com/drorlab/pensa/tree/master/tutorial).

The [documentation](https://pensa.readthedocs.io/en/latest/) is still under construction. For now, please refer to the [tutorial](https://github.com/drorlab/pensa/tree/master/tutorial) and the installation instructions below.


## Installation

### Conda environment

Create and activate a conda environment:

    conda create --name pensa python=3.7 numpy scipy matplotlib mdtraj==1.9.3 pyemma mdshare MDAnalysis cython -c conda-forge
    conda activate pensa

If you want to use PENSA with Jupyter notebooks:

    conda install jupyter

### Library installation

#### Option 1: Install the PENSA library from PyPI

Within the environment created above, execute:

    pip install pensa

This installs the latest released version.

To use the example scripts or tutorial folder, you'll have to download them from the repository.

#### Option 2: Create editable installation from source

Within the environment created above, execute:

    git clone https://github.com/drorlab/pensa.git
    cd pensa
    pip install -e . 

This installs the latest version from the repository, which might not yet be officially released.

## Citation

General:
```
Martin Vögele. PENSA. http://doi.org/10.5281/zenodo.4362136
```

To get the citation and DOI for a particular version, see [Zenodo](https://zenodo.org/record/4362136):


## Acknowledgments

#### Beta-Tests
Alex Powers, Sang Truong, Lukas Stelzl

#### Funding & Support 
This project was started by Martin Vögele at Stanford University, supported by an EMBO long-term fellowship (ALTF 235-2019), as part of the INCITE computing project 'Enabling the Design of Drugs that Achieve Good Effects Without Bad Ones' (BIP152).


