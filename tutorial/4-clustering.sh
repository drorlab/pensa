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
	--ref_file_a 'traj/condition-a_receptor.gro' \
	--trj_file_a 'traj/condition-a_receptor.xtc' \
	--ref_file_b 'traj/condition-b_receptor.gro' \
	--trj_file_b 'traj/condition-b_receptor.xtc' \
	--label_a 'A' \
	--label_b 'B' \
	--out_plots 'plots/receptor' \
	--out_frames_a 'clusters/receptor' \
	--out_frames_b 'clusters/receptor' \
	--start_frame 2000 \
	--feature_type 'bb-torsions' \
	--algorithm 'kmeans' \
	--max_num_clusters 12 \
	--write_num_clusters 2

