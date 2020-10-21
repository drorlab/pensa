#!/bin/bash

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

