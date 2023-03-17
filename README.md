# PENSA - Python Ensemble Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4362136.svg)](https://doi.org/10.5281/zenodo.4362136)
![Package](https://github.com/drorlab/pensa/workflows/package/badge.svg)
[![Documentation
Status](https://readthedocs.org/projects/pensa/badge/?version=latest)](http://pensa.readthedocs.io/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/drorlab/pensa/blob/master/LICENSE)
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)

A collection of Python methods for exploratory analysis and comparison of biomolecular conformational ensembles, e.g., from molecular dynamics simulations.
All functionality is available as a Python package.  

To get started, see the [__documentation__](https://pensa.readthedocs.io/en/latest/) which includes a tutorial for the PENSA library, or read our [__preprint__](https://arxiv.org/abs/2212.02714).

If you would like to contribute, check out our [__contribution guidelines__](https://github.com/drorlab/pensa/blob/master/CONTRIBUTING.md) and our [__to-do list__](https://github.com/drorlab/pensa/blob/master/TODO.md).

## Functionality

With PENSA, you can (currently):
- __compare structural ensembles__ of biomolecules (proteins, DNA or RNA) via the relative entropy of their features or statistical tests and visualize deviations on a reference structure.
- project several ensembles on a __joint reduced representation__ using principal component analysis (PCA) or time-lagged independent component analysis (tICA) and sort the structures along the obtained components.
- __cluster structures across ensembles__ via k-means or regular-space clustering and write out the resulting clusters as trajectories.
- trace allosteric information flow through a protein using __state-specific information__ analysis methods.

Biomolecules can be featurized using backbone torsions, sidechain torsions, or arbitrary distances (e.g., between all backbone C-alpha atoms) and we provide density-based methods to featurize water and ion pockets. PENSA also includes trajectory processing tools based on [MDAnalysis](https://www.mdanalysis.org/) and plotting functions using [Matplotlib](https://matplotlib.org/).

## Documentation
PENSA's documentation pages are [here](https://pensa.readthedocs.io/en/latest/), where you find installation instructions, API documentation, and a tutorial.

#### Example Scripts
For the most common applications, example [Python scripts](https://github.com/drorlab/pensa/tree/master/scripts) are provided. We show how to run the example scripts in a short separate [tutorial](https://github.com/drorlab/pensa/tree/master/tutorial).

#### Demo on Google Colab
We demonstrate how to use the PENSA library in an interactive and animated example on Google Colab, where we use freely available simulations of a mu-Opioid Receptor from GPCRmd.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1msHB6uGeu2tBw_MnAFFTxcxeW4RnR0is)


## Citation

General citation, representing the "concept" of the software:
```
Martin Vögele, Neil Thomson, Sang Truong, Jasper McAvity. (2021). PENSA. Zenodo. http://doi.org/10.5281/zenodo.4362136
```
To get the citation and DOI for a particular version, see [Zenodo](https://zenodo.org/record/4362136).

Please also consider citing our our [preprint](https://arxiv.org/abs/2212.02714):
```
Systematic Analysis of Biomolecular Conformational Ensembles with PENSA
M. Vögele, N. J. Thomson, S. T. Truong, J. McAvity, U. Zachariae, R. O. Dror
arXiv:2212.02714 [q-bio.BM] 2022
```


## Acknowledgments

#### Contributors
Martin Vögele, Neil Thomson, Sang Truong, Jasper McAvity

#### Beta-Testers
Alexander Powers, Lukas Stelzl, Nicole Ong, Eleanore Ocana, Emma Andrick, Callum Ives, and Bu Tran

#### Funding & Support 
This project was started by Martin Vögele at Stanford University, supported by an EMBO long-term fellowship (ALTF 235-2019), as part of the INCITE computing project 'Enabling the Design of Drugs that Achieve Good Effects Without Bad Ones' (BIP152). Neil Thomson was supported by a BBSRC EASTBIO PhD studentship and Jasper McAvity by the Stanford Computer Science department via the CURIS program. Stanford University, the Stanford Research Computing Facility, and the University of Dundee provided additional computational resources and support that contributed to these research results.

