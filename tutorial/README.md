# PENSA Tutorial 

This repository provides a python library and several ready-to-use [python scripts](https://github.com/drorlab/pensa/tree/master/scripts). 

## mu-Opioid receptor from GPCRmd
We explain the library in an [example notebook](https://github.com/drorlab/pensa/blob/master/tutorial/PENSA_Tutorial_GPCRmd.ipynb) using freely available simulation data from [GPCRmd](https://submission.gpcrmd.org/home/).
Additionally, we host an animated version on Google Colab.

[![Open Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1difJjlcwpN-0hSmGCGrPq9Cxq5wJ7ZDa?usp=sharing\)


## Rhodopsin Complexes
- [rhodopsin complexes](https://github.com/drorlab/pensa/blob/master/tutorial/PENSA_Tutorial_Sherlock.ipynb) (for users on [Sherlock](https://www.sherlock.stanford.edu/) at Stanford only)

and the scripts in the section below. 

All of this requires that the library is installed (as explained [here](https://github.com/drorlab/pensa#installation)).


## Accessing the library directly

In your custom python script or Jupyter Notebook, import the PENSA methods via

    from pensa import *
    
Have a look at the example notebooks that demonstrates the functionality of the library. 


## Usage-ready scripts

This tutorial shows the usage of the scripts for the basic applications provided with this repository. 
For each of the following four steps, a bash script runs the python script for an example system: the mu-opioid receptor, once in its apo form and once bound to the ligand BU72. We download the trajectories from GPCRmd.
For users of the Sherlock cluster at Stanford, an alternative system is available (no download necessary): rhodopsin, once bound to arrestin-1 and once bound to Gi. Below, we go through the steps as invoked by these bash scripts to demonstrate how to use the python code.

The following assumes that you invoke the tutorial scripts from the folder ```tutorial``` in the PENSA repository. If this is not the case, you should adapt the file paths accordingly.

Two notes for Sherlock users: 
- It might be useful to copy the tutorial folder to ```$OAK``` and run the scripts from there. Storage in the home directories is quite limited.
- You can skip the scripts ```0-``` and ```1-``` and start at ```1alt-``` instead. 

Preprocessing is necessary for all of the subsequent steps, which then are independent from one another.

### Downloading example data

For the MOR example, we use example data from [GPCRmd](https://submission.gpcrmd.org/home/).
Skip this step if you do the rhodopsin example on Sherlock or if you have already downloaded this data.

    python ~/pensa/scripts/get_tutorial_datasets.py -d "./mor-data"

### Preprocessing

To work with the protein coordinates, we first need to extract them from the simulation, i.e., remove the solvent, lipids etc. and write them in the .xtc format that the internal featurization understands. This is the hardest part but you usually only have to do it once and can then play with your data. Preprocessing can handle many common trajectory formats (as it is based on MDAnalysis) but the internal featurization (based on PyEMMA) is a bit more restrictive. 

We start by defining the trajectory files of the simulations that we want to compare:

    ROOT="./mor-data"

    REF_FILE_A="$ROOT/11427_dyn_151.psf"
    PDB_FILE_A="$ROOT/11426_dyn_151.pdb"
    TRJ_FILE_A="$ROOT/11423_trj_151.xtc $ROOT/11424_trj_151.xtc $ROOT/11425_trj_151.xtc"
    OUT_NAME_A="traj/condition-a"

    REF_FILE_B="$ROOT/11580_dyn_169.psf"
    PDB_FILE_B="$ROOT/11579_dyn_169.pdb"
    TRJ_FILE_B="$ROOT/11576_trj_169.xtc $ROOT/11577_trj_169.xtc $ROOT/11578_trj_169.xtc"
    OUT_NAME_B="traj/condition-b"

    OUT_NAME_COMBINED="traj/combined"


In each simulation, we need to select the protein or the part of it that we want to investigate. 
It is crucial for comparing two simulations that the residues selected from both simulations are the same! 
We provide a basic string (selection in [MDAnalysis format](https://docs.mdanalysis.org/1.0.0/documentation_pages/selections.html)) to which the corresponding residue numbers will be added later:

    SEL_BASE_A="(not name H*) and protein and "
    SEL_BASE_B="(not name H*) and protein and "

In this example, we also exclude hydrogen atoms because the residue Asp114 is protonated in the BU72 simulation but not in the apo simulation. This would prevent us from stitching the trajectories together. Comparisons would still work, as long as all derived features are the same in both conditions.

We also define the names of the processed trajectories, here: one for each simulation condition and a combined one:

    OUT_NAME_A="traj/condition-a"
    OUT_NAME_B="traj/condition-b"
    OUT_NAME_COMBINED="traj/combined"

Here, we want to select once the entire receptor and once only the transmembrane part (tm). For this purpose, we have to write the ranges of selected residues into a separate file. 
This, for example, is the file ```selection/mor_tm.txt```:

    76 98
    105 133
    138 173
    182 208
    226 264
    270 308
    315 354
 
In the following, we iterate over the two different selections and invoke the python scripts ```extract_coordinates.py``` and ```extract_coordinates_combined.py``` to each of them, saving the processed trajectories in the directory ```traj```:

    mkdir -p traj

    for PART in receptor tm; do

      SEL_FILE="selections/mor_${PART}.txt"

      echo CONDITION A, $PART
      python ~/pensa/scripts/extract_coordinates.py \
        --sel_base "$SEL_BASE_A" --sel_file "$SEL_FILE" \
        --ref_file "$REF_FILE_A" --pdb_file "$PDB_FILE_A" \
        --trj_file  $TRJ_FILE_A  --out_name "${OUT_NAME_A}_${PART}"

      echo CONDITION B, $PART
      python ~/pensa/scripts/extract_coordinates.py \
        --sel_base "$SEL_BASE_B" --sel_file "$SEL_FILE" \
        --ref_file "$REF_FILE_B" --pdb_file "$PDB_FILE_B" \
        --trj_file  $TRJ_FILE_B  --out_name "${OUT_NAME_B}_${PART}"

      echo COMBINED, $PART 
      # needs one reference file for each trajectory file
      python ~/pensa/scripts/extract_coordinates_combined.py \
        --ref_file_a  $REF_FILE_A $REF_FILE_A $REF_FILE_A \
        --ref_file_b  $REF_FILE_B $REF_FILE_B $REF_FILE_B \
        --trj_file_a  $TRJ_FILE_A  --trj_file_b  $TRJ_FILE_B  \
        --sel_base_a "$SEL_BASE_A" --sel_base_b "$SEL_BASE_B" \
        --sel_file "selections/mor_${PART}.txt" \
        --out_name "${OUT_NAME_COMBINED}_${PART}"

    done


Note for performance: A lot of trajectory reading and writing is going on. If you can, avoid running this locally if the trajectories are on a remotely mounted file system.

If you store your trajectories distributed across several files (as in the GPCRmd MOR example), the extract_coordinates.py script can take multiple arguments, e.g. 

    --trj_file "$TRJ_FILE_A1 $TRJ_FILE_A2 $TRJ_FILE_A3"


### Comparison of two structural ensembles

PENSA allows you to detect deviations between two simulations feature by feature. It analyzes the backbone torsions, side-chain torsions, and distances of C-alpha atoms (which we call "features" of the respective system).
In order to compare two structural ensembles, we need to provide the topology files (.gro) and trajectory files (.xtc) that we have extracted in the preprocessing step. With the option ```--out_plots```, we define the base for the filenames of the output plots and with ```--out_vispdb```, we define the same for the PDB files in which the deviation measure between the features will be saved in the field that usually contains the B factor. Additionally, you can provide the number of the simulation frame to start with via ```--start_frame``` and the number of residues with the strongest deviation to be printed via ```--print_num```.

    mkdir -p plots
    mkdir -p vispdb

    python ~/pensa/scripts/compare_feature_distributions.py \
        --ref_file_a traj/condition-a_receptor.gro \
        --trj_file_a traj/condition-a_receptor.xtc \
        --ref_file_b traj/condition-b_receptor.gro \
        --trj_file_b traj/condition-b_receptor.xtc \
        --out_plots  plots/receptor \
        --out_vispdb vispdb/receptor \
        --out_results results/receptor \
        --start_frame 0 \
        --print_num 12

In the python script used in this tutorial, the deviation is measured using the Jensen-Shannon distance, a symmetric measure that is the square root of the [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence). The same function in the PENSA module also allows to calculate the (asymmetric) [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) with respect to either of the simulations. Other functions can perform a [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) or simply compare the mean values and standard deviations of each feature.

The plots are saved as PDF files. For torsions, the plot is one-dimensional (with the residue number at the x-axis). For distances, the plot is a 2D heatmap.

The PDB files with the maximum deviation of any torsion related to a certain residue can be visualized using [VMD](https://www.ks.uiuc.edu/Research/vmd/) and we provide a tcl script that does the basic first steps:

    vmd vispdb/receptor_bb-torsions_jsd.pdb -e ~/pensa/scripts/residue_visualization.tcl
    vmd vispdb/receptor_sc-torsions_jsd.pdb -e ~/pensa/scripts/residue_visualization.tcl

As we can see here, sidechain and backbone torsions are treated separately.

### Principal component analysis

To characterize collective variations within a structural ensemble, PENSA allows you to calculate the principal components of a simulation but also of a combined ensemble from two simulations - which this section is about. The script below works either in the space of backbone torsions, side-chain torsions, or distances of C-alpha atoms (defined by the option ```--feature_type```).

    mkdir -p pca

    python ~/pensa/scripts/calculate_combined_principal_components.py \
        --ref_file_a 'traj/condition-a_receptor.gro' \
        --trj_file_a 'traj/condition-a_receptor.xtc' \
        --ref_file_b 'traj/condition-b_receptor.gro' \
        --trj_file_b 'traj/condition-b_receptor.xtc' \
        --out_results results/receptor \
        --out_plots 'plots/receptor' \
        --out_pc 'pca/receptor' \
        --start_frame 400 \
        --feature_type 'bb-torsions' \
        --num_eigenvalues 12 \
        --num_components 3 \
        --feat_threshold 0.4 
        
The PCA script, as invoked above, performs the following tasks:
 - It plots the eigenvalues of the dominant principal components (the number of which is determined via ```--num_eigenvalues```). 
 - It plots the distribution of simulation frames from each condition along the common PCs (the number of which is determined via ```--num_components```). 
 - It performs a feature correlation analysis and prints the features that are most correlated with each PC (the threshold for the correlation is provided via ```--feat_threshold```).
 - It sorts simulation frames along each PC which is crucial for visualizing the major directions of variablity within the examined protein ensemble. Again, you can visualize them using VMD.
 
The plots are saved as PDF files in the folder ```plots``` and the sorted frames are saved as trajectories in the folder ```pca```.
    
### Clustering

To detect major states of a structural ensemble, PENSA can calculate clusters of simulation frames. In the python script provided here, we use the combined ensemble form two simulations to determine these clusters. 

    mkdir -p clusters

    python ~/pensa/scripts/calculate_combined_clusters.py --no-write --wss \
        --ref_file_a 'traj/condition-a_receptor.gro' \
        --trj_file_a 'traj/condition-a_receptor.xtc' \
        --ref_file_b 'traj/condition-b_receptor.gro' \
        --trj_file_b 'traj/condition-b_receptor.xtc' \
        --label_a 'A' \ # Label for the plot
        --label_b 'B' \  # Label for the plot
        --out_plots 'plots/receptor' \
        --out_results results/receptor \
        --out_frames_a 'clusters/condition-a_receptor' \
        --out_frames_b 'clusters/condition-b_receptor' \
        --start_frame 2000 \
        --feature_type 'bb-torsions' \
        --algorithm 'kmeans' \
        --max_num_clusters 12 \
        --write_num_clusters 2

The clustering script, as invoked above, performs the following tasks:
 - It plots the number of frames from each simulation in each cluster. The number of clusters in which to divide the ensemble is determined via ```--write_num_clusters```). 
 - It sorts the frames from each simulation into their corresponding cluster. The bases of the corresponding filenames are given via ```out_frames_a```, and ```out_frames_b```, respectively.
 - It calculates the With-In-Sum-Of-Squares (WSS) for different numbers of clusters (the maximum number provided via ```--max_num_clusters```) and plots the result. This plot can be used to determine the optimal number of clusters.
 
The plots are saved as PDF files in the folder ```plots``` and the sorted frames are saved as trajectories in the folder ```clusters```.    

