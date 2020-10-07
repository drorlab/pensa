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

    pip install -e . 

## To do:

- upload examples
- add CNN difference learning
- implement Wasserstein distance
- add job submission scripts (?)
