#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --gres gpu:1
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mvoegele@stanford.edu
#SBATCH --qos=high_p

mkdir -p pca
mkdir -p plots
mkdir -p results

python ../scripts/calculate_combined_principal_components.py \
	--ref_file_a 'traj/condition-a_receptor.gro' \
	--trj_file_a 'traj/condition-a_receptor.xtc' \
	--ref_file_b 'traj/condition-b_receptor.gro' \
	--trj_file_b 'traj/condition-b_receptor.xtc' \
	--out_plots 'plots/receptor' \
	--out_pc 'pca/receptor' \
        --out_results 'results/receptor' \
	--start_frame 2000 \
	--feature_type 'bb-torsions' \
	--num_eigenvalues 12 \
	--num_components 3 \
	--feat_threshold 0.4 

