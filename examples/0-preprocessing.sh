ROOT="/oak/stanford/groups/rondror/projects/MD_simulations/amber/INCITE_REMD_trajectories"

REF_FILE_G="$ROOT/remd-rhodopsin-gi/system_reference.psf"
PDB_FILE_G="$ROOT/remd-rhodopsin-gi/system.pdb"
TRJ_FILE_G="$ROOT/remd-rhodopsin-gi/stitched_99999999999_310.nc"
SEL_BASE_G="protein and segid P3 and "

REF_FILE_A="$ROOT/remd-rhodopsin-arr/system_reference.psf"
PDB_FILE_A="$ROOT/remd-rhodopsin-arr/system.pdb"
TRJ_FILE_A="$ROOT/remd-rhodopsin-arr/stitched_99999999999_310.nc"
SEL_BASE_A="protein and segid P229 and "

mkdir -p traj

for PART in receptor tm; do 

	SEL_FILE="selections/rho_${PART}.txt"
	
	echo RHODOPSIN-GI $PART
	OUT_NAME_G="traj/rhodopsin_gibound_${PART}"
	python ~/pensa/scripts/extract_coordinates.py \
		--sel_base "$SEL_BASE_G" --sel_file "$SEL_FILE" \
		--ref_file "$REF_FILE_G" --pdb_file "$PDB_FILE_G" \
		--trj_file "$TRJ_FILE_G" --out_name "$OUT_NAME_G"

	echo RHODOPSIN-ARRESTIN1 $PART
        OUT_NAME_A="traj/rhodopsin_arrbound_${PART}"
        python ~/pensa/scripts/extract_coordinates.py \
		--sel_base "$SEL_BASE_A" --sel_file "$SEL_FILE" \
		--ref_file "$REF_FILE_A" --pdb_file "$PDB_FILE_A" \
		--trj_file "$TRJ_FILE_A" --out_name "$OUT_NAME_A"

	echo RHODOPSIN COMBINED $PART # (for joint PCA)
	OUT_NAME_COMBINED_M="traj/rhodopsin_combined_${PART}"
        python ~/pensa/scripts/extract_coordinates_combined.py \
		--ref_file_a "$REF_FILE_G" --ref_file_b "$REF_FILE_A" \
		--trj_file_a "$TRJ_FILE_G" --trj_file_b "$TRJ_FILE_A" \
                --sel_base_a "$SEL_BASE_G" --sel_base_b "$SEL_BASE_A" \
                --sel_file "$SEL_FILE" --out_name "$OUT_NAME_COMBINED_M"

done


