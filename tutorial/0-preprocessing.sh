ROOT="/oak/stanford/groups/rondror/projects/MD_simulations/amber/INCITE_REMD_trajectories"

REF_FILE_A="$ROOT/remd-rhodopsin-arr/system_reference.psf"
PDB_FILE_A="$ROOT/remd-rhodopsin-arr/system.pdb"
TRJ_FILE_A="$ROOT/remd-rhodopsin-arr/stitched_99999999999_310.nc"
SEL_BASE_A="protein and segid P229 and "
OUT_NAME_A="traj/rhodopsin_arrbound"

REF_FILE_B="$ROOT/remd-rhodopsin-gi/system_reference.psf"
PDB_FILE_B="$ROOT/remd-rhodopsin-gi/system.pdb"
TRJ_FILE_B="$ROOT/remd-rhodopsin-gi/stitched_99999999999_310.nc"
SEL_BASE_B="protein and segid P3 and "
OUT_NAME_B="traj/rhodopsin_gibound"

OUT_NAME_COMBINED="traj/rhodopsin_combined"

mkdir -p traj

for PART in receptor tm; do 

	SEL_FILE="selections/rho_${PART}.txt"
	
	echo CONDITION A, $PART
        python ~/pensa/scripts/extract_coordinates.py \
		--sel_base "$SEL_BASE_A" --sel_file "$SEL_FILE" \
		--ref_file "$REF_FILE_A" --pdb_file "$PDB_FILE_A" \
		--trj_file "$TRJ_FILE_A" --out_name "${OUT_NAME_A}_${PART}"

	echo CONDITION B, $PART
	python ~/pensa/scripts/extract_coordinates.py \
		--sel_base "$SEL_BASE_B" --sel_file "$SEL_FILE" \
		--ref_file "$REF_FILE_B" --pdb_file "$PDB_FILE_B" \
		--trj_file "$TRJ_FILE_B" --out_name "${OUT_NAME_B}_${PART}"

	echo COMBINED, $PART
        python ~/pensa/scripts/extract_coordinates_combined.py \
		--ref_file_a "$REF_FILE_A" --ref_file_b "$REF_FILE_B" \
		--trj_file_a "$TRJ_FILE_A" --trj_file_b "$TRJ_FILE_B" \
                --sel_base_a "$SEL_BASE_A" --sel_base_b "$SEL_BASE_B" \
                --sel_file "$SEL_FILE" --out_name "${OUT_NAME_COMBINED}_${PART}"

done


