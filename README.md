# PENSA - Protein Ensemble Analysis

A collection of python methods for exploratory analysis and comparison of protein structural ensembles, e.g., from molecular dynamics simulations.

With PENSA, you can (currently):
- compare structural ensembles of proteins via the relative entropy of their features and visualize deviations on a reference structure.
- project ensembles on their combined principal components (PCs) and sort the structures along a PC.
- cluster structures via k-means and via regular-space clustering and write out the resulting clusters as trajectories.

Proteins are featurized via PyEMMA using backbone torsions, sidechain torsions, or backbone C-alpha distances, making PENSA compatible to all functionality available in PyEMMA. Trajectories are processed and written using MDAnalysis. Plots are generated using Matplotlib. 

All functionality is available as a python package (installation see below). For the most common applications, example [python scripts](https://github.com/drorlab/pensa/tree/master/scripts) are provided. To get started, see the [tutorial](https://github.com/drorlab/pensa/tree/master/tutorial).

## Requirements

Python 3.7 with:
- numpy
- scipy >= 1.2
- mdshare
- pyemma
- MDAnalysis
- matplotlib

## Installation

Create and activate a conda environment:

    conda create --name pensa python=3.7 numpy scipy>=1.2 matplotlib pyemma mdshare MDAnalysis -c conda-forge
    conda activate pensa

If you want to use PENSA with Jupyter notebooks:

    conda install jupyter
    
Download and install PENSA:

    git clone https://github.com/drorlab/pensa.git
    cd pensa
    pip install -e . 


## To do:

- add CNN difference learning
- implement Wasserstein distance

