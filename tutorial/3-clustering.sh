#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --gres gpu:1
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mvoegele@stanford.edu
#SBATCH --qos=high_p

mkdir -p plots
mkdir -p clusters

python ~/pensa/scripts/calculate_combined_clusters.py --write --wss \
	--ref_file_a 'traj/rhodopsin_arrbound_receptor.gro' \
	--trj_file_a 'traj/rhodopsin_arrbound_receptor.xtc' \
	--ref_file_b 'traj/rhodopsin_gibound_receptor.gro' \
	--trj_file_b 'traj/rhodopsin_gibound_receptor.xtc' \
	--label_a 'Rho-Arr' \
	--label_b 'Rho-Gi' \
	--out_plots 'plots/rhodopsin_receptor' \
	--out_frames_a 'clusters/rhodopsin_arrbound_receptor' \
	--out_frames_b 'clusters/rhodopsin_gibound_receptor' \
	--start_frame 2000 \
	--feature_type 'bb-torsions' \
	--algorithm 'kmeans' \
	--max_num_clusters 12 \
	--write_num_clusters 2

