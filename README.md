# PENSA - Protein Ensemble Analysis

A collection of python methods for exploratory analysis and comparison of protein structural ensembles, e.g., from molecular dynamics simulations.

With PENSA, you can (currently):
- compare structural ensembles of proteins via the relative entropy of their features and visualize deviations on a reference structure.
- project ensembles on their combined principal components (PCs) and sort the structures along a PC.
- cluster structures via k-means and via regular-space clustering and write out the resulting clusters as trajectories.

Proteins are featurized via PyEMMA using backbone torsions, sidechain torsions, or backbone C-alpha distances, making PENSA compatible to all functionality available in PyEMMA. Trajectories are processed and written using MDAnalysis. Plots are generated using Matplotlib. 

All functionality is available as a python package (installation see below). For the most common applications, example [python scripts](https://github.com/drorlab/pensa/tree/master/scripts) are provided. To get started, see the [tutorial](https://github.com/drorlab/pensa/tree/master/tutorial).


## Installation

### Conda environment

Create and activate a conda environment:

    conda create --name pensa python=3.7 numpy scipy matplotlib pyemma mdshare MDAnalysis cython -c conda-forge
    conda activate pensa

If you want to use PENSA with Jupyter notebooks:

    conda install jupyter

### Library installation

#### Variant 1: Install PENSA library from PyPI

Within the environment created above, execute:

    pip install pensa

To use the example scripts or tutorial folder, you'll have to download them separately.

#### Variant 2: Download PENSA and create editable installation

Within the environment created above, execute:

    git clone https://github.com/drorlab/pensa.git
    cd pensa
    pip install -e . 


## Contributions

#### Development
Martin Vögele

#### Beta-Tests
Martin Vögele, Alex Powers, Sang Truong

#### Funding & Support 
This project was started by Martin Vögele at Stanford University, supported by an EMBO long-term fellowship (ALTF 235-2019), as part of the INCITE computing project 'Enabling the Design of Drugs that Achieve Good Effects Without Bad Ones' (BIP152).


