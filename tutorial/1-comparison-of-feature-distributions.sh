#!/bin/bash

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

