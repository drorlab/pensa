#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --qos=high_p

mkdir -p plots
mkdir -p vispdb
mkdir -p results

python ../scripts/compare_feature_distributions.py \
	--ref_file_a traj/condition-a_receptor.gro \
	--trj_file_a traj/condition-a_receptor.xtc \
	--ref_file_b traj/condition-b_receptor.gro \
	--trj_file_b traj/condition-b_receptor.xtc \
	--out_plots  plots/receptor \
	--out_vispdb vispdb/receptor \
	--out_results results/receptor \
	--start_frame 0 \
	--print_num 12

