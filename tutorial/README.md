# PENSA Tutorial 

This repository provides a python library and several ready-to-use python scripts. We start with the python scripts, explaining the basic functionalities, and then explain how to access the library directly to write your own scripts or to use its methods interactively in a Jupyter Notebook.

The following assumes that you have cloned the PENSA repository to your home directory (```~/```) and that you work on the Sherlock cluster at Stanford. If this is not the case, you should adapt the file paths accordingly.

A note for Sherlock users: It might be useful to copy the tutorial folder to ```$OAK``` and run the scripts from there. Storage in the home directories is quite limited.

## Usage-ready scripts

This tutorial shows the usage of the scripts for the basic applications provided with this repository. 
For each of the following four steps, a bash script runs the python script for an example system: rhodopsin, once bound to arrestin-1 and once bound to Gi. Below, we go through the steps as invoked by these bash scripts to demonstrate how to use the python code.

Preprocessing is necessary for all of the subsequent steps, which then are independent from one another.

### Preprocessing

To work with the protein coordinates, we first need to extract them from the simulation, i.e., remove the solvent, lipids etc. and write them in the .xtc format that the internal featurization understands. This is the hardest part but you usually only have to do it once and can then play with your data. Preprocessing can handle many common trajectory formats (as it is based on MDAnalysis) but the internal featurization (based on PyEMMA) is a bit more restrictive. 

We start by defining the trajectory files of the simulations that we want to compare:

    ROOT="/oak/stanford/groups/rondror/projects/MD_simulations/amber/INCITE_REMD_trajectories"

    REF_FILE_A="$ROOT/remd-rhodopsin-arr/system_reference.psf"
    PDB_FILE_A="$ROOT/remd-rhodopsin-arr/system.pdb"
    TRJ_FILE_A="$ROOT/remd-rhodopsin-arr/stitched_99999999999_310.nc"

    REF_FILE_B="$ROOT/remd-rhodopsin-gi/system_reference.psf"
    PDB_FILE_B="$ROOT/remd-rhodopsin-gi/system.pdb"
    TRJ_FILE_B="$ROOT/remd-rhodopsin-gi/stitched_99999999999_310.nc"

In each simulation, we need to select the protein or the part of it that we want to investigate. It is crucial for comparing two simulations that the residues selected from both simulations are the same! We provide a basic string (selction in [MDAnalysis format](https://docs.mdanalysis.org/1.0.0/documentation_pages/selections.html)) to which the corresponding residue numbers will be added later:

    SEL_BASE_A="protein and segid P229 and "
    SEL_BASE_B="protein and segid P3 and "

We also define the names of the processed trajectories, here: one for each simulation condition and a combined one:

    OUT_NAME_A="traj/rhodopsin_arrbound"
    OUT_NAME_B="traj/rhodopsin_gibound"
    OUT_NAME_COMBINED="traj/rhodopsin_combined"

Here, we want to select once the entire receptor and once only the transmembrane part (tm). For this purpose, we have to write the ranges of selected residues into a separate file. This, for example, is the file ```selection/rho_tm.txt```:

    43 65
    70 100
    106 141
    149 173
    199 237
    240 278
    285 321
    
In the following, we iterate over the two different selections and invoke the python scripts ```extract_coordinates.py``` and ```extract_coordinates_combined.py``` to each of them, saving the processed trajectories in the directory ```traj```:

    mkdir -p traj
    
    for PART in receptor tm; do 
    
      SEL_FILE="selections/rho_${PART}.txt"

      echo "CONDITION A, $PART"
      python ~/pensa/scripts/extract_coordinates.py \
        --sel_base "$SEL_BASE_A" --sel_file "$SEL_FILE" \
        --ref_file "$REF_FILE_A" --pdb_file "$PDB_FILE_A" \
        --trj_file "$TRJ_FILE_A" --out_name "${OUT_NAME_A}_${PART}"

      echo "CONDITION, B $PART"
      python ~/pensa/scripts/extract_coordinates.py \
        --sel_base "$SEL_BASE_B" --sel_file "$SEL_FILE" \
        --ref_file "$REF_FILE_B" --pdb_file "$PDB_FILE_B" \
        --trj_file "$TRJ_FILE_B" --out_name "${OUT_NAME_B}_${PART}"

      echo "COMBINED, $PART"
      python ~/pensa/scripts/extract_coordinates_combined.py \
        --ref_file_a "$REF_FILE_A" --ref_file_b "$REF_FILE_B" \
        --trj_file_a "$TRJ_FILE_A" --trj_file_b "$TRJ_FILE_B" \
        --sel_base_a "$SEL_BASE_A" --sel_base_b "$SEL_BASE_B" \
        --sel_file "$SEL_FILE" --out_name "${OUT_NAME_COMBINED}_${PART}"
    done

It is preferable to run this on Sherlock directly than on a local machine with the mounted file system because a lot of trajectory reading and writing is going on.

### Comparison of two structural ensembles

PENSA allows you to detect deviations between two simulations feature by feature. It analyzes the backbone torsions, side-chain torsions, and distances of C-alpha atoms (which we call "features" of the respective system).
In order to compare two structural ensembles, we need to provide the topology files (.gro) and trajectory files (.xtc) that we have extracted in the preprocessing step. With the option ```--out_plots```, we define the base for the filenames of the output plots and with ```--out_vispdb```, we define the same for the PDB files in which the deviation measure between the features will be saved in the field that usually contains the B factor. Additionally, you can provide the number of the simulation frame to start with via ```--start_frame``` and the number of residues with the strongest deviation to be printed via ```--print_num```.

    mkdir -p plots
    mkdir -p vispdb

    python ~/pensa/scripts/compare_feature_distributions.py \
        --ref_file_a traj/rhodopsin_arrbound_receptor.gro \
        --trj_file_a traj/rhodopsin_arrbound_receptor.xtc \
        --ref_file_b traj/rhodopsin_gibound_receptor.gro \
        --trj_file_b traj/rhodopsin_gibound_receptor.xtc \
        --out_plots  plots/rhodopsin_receptor \
        --out_vispdb vispdb/rhodopsin_receptor \
        --start_frame 0 \
        --print_num 12

In the python script used in this tutorial, the deviation is measured using the [Jensen-Shannon distance](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence), a symmetric measure. The same function in the PENSA module also allows to calculate the (asymmetric) Kullback-Leibler divergence with respect to either of the simulations. Other functions can perform a Kolmogorov-Smirnov test or simply compare the mean values and standard deviations of each feature.

The plots are saved as PDF files. The PDB files with the maximum deviation of any torsion related to a certain residue can be visualized using [VMD](https://www.ks.uiuc.edu/Research/vmd/) and we provide a tcl script that does the basic first steps:

    vmd vispdb/rhodopsin_receptor_bbtors-distributions_jsd.pdb -e ~/pensa/scripts/residue_visualization.tcl
    vmd vispdb/rhodopsin_receptor_sctors-distributions_jsd.pdb -e ~/pensa/scripts/residue_visualization.tcl

As you can see here, sidechain and backbone torsions are treated separately.

### Principal component analysis

### Clustering

## Accessing the library directly

In your custom python script or Jupyter Notebook, import the PENSA methods via

    from pensa import *
    
An example notebook that demonstrates the functionality will soon be provided.    
