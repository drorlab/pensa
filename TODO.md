### In Progress

- [ ] Tests
  - [x] Workflow test with example data
  - [ ] Trivial examples for each function
  - [ ] Unit tests for SSI 
  - [ ] Unit tests for density features
- [ ] Integrate [DiffNets](https://doi.org/10.1101/2020.07.01.182725).
  - [ ] Lay out module structure in separate branch.
  - [ ] Copy core network from DiffNets repo.
  - [ ] Try to use existing featurization.
  - [ ] Include existing DiffNets featurization and compare.
- [ ] exploratory analysis via correlation coefficients of the features
  - [x] First tests --> not very promising.
  - [ ] Try [different metric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html)
  - [ ] Find useful application or leave it out.
- [ ] Unified tutorial in documentation. Make one page for each subpackage
  - [x] preprocessing
    - [x] coordinates
    - [x] densities
  - [x] featurization
    - [x] structure features
    - [x] water features
    - [x] atom features
  - [x] comparison
  - [x] dimensionality reduction
  - [ ] clusters (show how to cluster on PCs)
  - [x] SSI

### Plans

- [ ] Try using MDAnalysis instead of biotite for water featurization
- [ ] Integrate more options for features from PyEMMA (think carefully about how to make it more flexible)
- [ ] More example tcl scripts for VMD 
- [ ] Facilitate calculation of JSD etc. on principal components
- [ ] Facilitate calculation of SSI on results of joint clustering.
- [ ] Weighted PCA/tICA? (to account for varying simulation lengths or uncertainty) 
- [ ] Feature comparison of more than two ensembles
  - [ ] with respect to the joint ensemble (all metrics)
  - [ ] with respect to a reference ensemble (will not always work for KLD)
- [ ] Implement T-distributed Stochastic Neighbor Embedding (t-SNE)
  - [ ] Read up on [t-SNE for molecular trajectories](https://www.frontiersin.org/articles/10.3389/fmolb.2020.00132/full)
  - [ ] See if we can import or adapt [existing code](https://github.com/spiwokv/tltsne).
  - [ ] First tests with (regular) t-SNE
  - [ ] Test time-lagged t-SNE. How to handle time-dependence across simulations/ensembles?
  - [ ] write module
  - [ ] write unit tests
- [ ] Implement a clustering algorithem designed for structural ensembles
  - [ ] Read up about [CLoNe](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa742/5895303) 
  - [ ] First tests
  - [ ] write module
  - [ ] write unit tests
- [ ] Put shared functionality of PCA and TICA into shared functions.
- [ ] Make file format (png/pdf?) for matplotlib optional.
- [ ] Implement [Linear Discriminant Analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
- [ ] Implement [Non-Negative Matrix Factorization](https://onlinelibrary.wiley.com/doi/10.1002/env.3170050203).

### Ideas
- [ ] Logo
- [ ] Hydrogen bonds as features
- [ ] Contacts as features (can PyEMMA do this?)
- [ ] Position deviations as features (similar to components of RMSD)
- [ ] Estimate thresholds for significance of feature differences
  - [ ] Calculate correlation times within trajectories
  - [ ] modify p-value of KS test using correlation time 
  - [ ] modify p-value of KS test using number of simulation runs per ensemble
- [ ] Wasserstein distance to compare ensembles
- [ ] Add options to save and load calculated features
- [ ] Add option to whiten features
- [ ] Featurizers for other molecule types
  - [ ] ligands
  - [ ] lipids
  - [ ] nucleic acids
- [ ] Simplify adding hand-crafted features 
- [ ] Account for [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction) in comparison.
- [ ] Implement conformational entropy calculations
  - [ ] Read papers, e.g, [1](https://www.pnas.org/content/111/43/15396), [2](https://www.mdpi.com/2079-3197/6/1/21/htm), [3](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01375)
  - [ ] Test implementations, e.g., [Xentropy](https://github.com/liedllab/X-Entropy) to find the best way to do it.
- [ ] Implement [multi-dimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling)
- [ ] Try to integrate [functional mode analysis](http://www3.mpibpc.mpg.de/groups/de_groot/fma.html).
- [ ] Try to integrate [VAMPnets](https://www.nature.com/articles/s41467-017-02388-1).
- [ ] Try to integrate [network analysis](https://aip.scitation.org/doi/full/10.1063/5.0020974).

### Done  ✓
- [x] Colab Tutorial
  - [x] Put Notebook on Colab and get it to run.
  - [x] Add visualizations.
  - [x] Fix installation via pip.
  - [x] Fix animations (they only show white canvas).
  - [x] Add TICA to Colab tutorial.
- [x] Include TICA in unit tests
- [x] Write "getting started" for documentation
- [x] Refactoring and fixes for release 0.2
  - [x] Restructure modules to subpackages
  - [x] Adapt README
  - [x] Adapt API documentation
  - [x] Include SSI to comparison example script
  - [x] Numbering of principal component trajectories starts with 0, should start with 1
  - [x] Axis labels and legend name for distance matrix plot
  - [x] Function pca_features() does not have labels
  - [x] Function compare_projections() does not have labels or legend
- [x] Slack channel for all developers and testers, and to provide support for the user community.
- [x] Implement clustering in principal component space


### Abandoned

- [ ] Frame classification via CNN on features
  - [x] Prototype to classify simulation frames --> Diffnets probably more powerful.
  - [ ] Interpret weights as relevance of features
  - [ ] Write module
  - [ ] Write unit tests
