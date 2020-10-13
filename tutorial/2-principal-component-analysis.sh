mkdir -p pca

python ~/pensa/scripts/calculate_combined_principal_components.py \
	--ref_file_a 'traj/rhodopsin_arrbound_receptor.gro' \
	--trj_file_a 'traj/rhodopsin_arrbound_receptor.xtc' \
	--ref_file_b 'traj/rhodopsin_gibound_receptor.gro' \
	--trj_file_b 'traj/rhodopsin_gibound_receptor.xtc' \
	--out_plots 'plots/rhodopsin_receptor' \
	--out_pc 'pca/rhodopsin_receptor' \
	--start_frame 2000 \
	--feature_type 'bb-torsions' \
	--num_eigenvalues 12 \
	--num_components 3 \
	--feat_threshold 0.4 

