# pensa - Protein Ensemble Analysis

A collection of python methods to cluster and compare ensembles of protein structures.

## Features

- Featurization of proteins via backbone torsions, sidechain torsions, and backbone C-alpha distances 
- Comparison via mean differences and via relative entropy
- Principal component analysis (with projection of trajectories on the principal components)
- Clustering via k-means and via regular space (+writing clusters as trajectories)

## Requirements

Python 3 with:
- numpy
- scipy
- mdshare
- pyemma
- MDAnalysis
- matplotlib

## To do:

- upload examples
- add CNN difference learning
- implement Wasserstein distance
- add generic preprocessing script
- add job submission scripts like in DrorMD 
  (alternative: merge methods in there)
