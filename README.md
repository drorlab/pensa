# PENSA - Protein Ensemble Analysis

A collection of python methods to cluster and compare ensembles of protein structures.

## Features

- Featurization of proteins via backbone torsions, sidechain torsions, and backbone C-alpha distances 
- Comparison via mean differences and via relative entropy
- Principal component analysis (with projection of trajectories on the principal components)
- Clustering via k-means and via regular space (+writing clusters as trajectories)

## Requirements

Python 3.7 with:
- numpy
- scipy
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

- upload examples
- add CNN difference learning
- implement Wasserstein distance
- add job submission scripts (?)
