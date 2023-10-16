#!/bin/bash

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

