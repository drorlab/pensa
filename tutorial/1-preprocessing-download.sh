#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --qos=high_p


ROOT="./mor-data"

REF_FILE_A="$ROOT/11427_dyn_151.psf"
PDB_FILE_A="$ROOT/11426_dyn_151.pdb"
TRJ_FILE_A="$ROOT/11423_trj_151.xtc $ROOT/11424_trj_151.xtc $ROOT/11425_trj_151.xtc"
SEL_BASE_A="protein and "
OUT_NAME_A="traj/condition-a"

REF_FILE_B="$ROOT/11580_dyn_169.psf"
PDB_FILE_B="$ROOT/11579_dyn_169.pdb"
TRJ_FILE_B="$ROOT/11576_trj_169.xtc $ROOT/11577_trj_169.xtc $ROOT/11578_trj_169.xtc"
SEL_BASE_B="protein and "
OUT_NAME_B="traj/condition-b"

OUT_NAME_COMBINED="traj/combined"

mkdir -p traj

for PART in receptor tm; do 

	SEL_FILE="selections/mor_${PART}.txt"
	
	echo CONDITION A, $PART
        python ~/pensa/scripts/extract_coordinates.py \
		--sel_base "$SEL_BASE_A" --sel_file "$SEL_FILE" \
		--ref_file "$REF_FILE_A" --pdb_file "$PDB_FILE_A" \
		--trj_file  $TRJ_FILE_A  --out_name "${OUT_NAME_A}_${PART}"

	echo CONDITION B, $PART
	python ~/pensa/scripts/extract_coordinates.py \
		--sel_base "$SEL_BASE_B" --sel_file "$SEL_FILE" \
		--ref_file "$REF_FILE_B" --pdb_file "$PDB_FILE_B" \
		--trj_file  $TRJ_FILE_B  --out_name "${OUT_NAME_B}_${PART}"

	echo COMBINED, $PART 
	# needs one reference file for each trajectory file
        python ~/pensa/scripts/extract_coordinates_combined.py \
		--ref_file_a  $REF_FILE_A $REF_FILE_A $REF_FILE_A \
		--ref_file_b  $REF_FILE_B $REF_FILE_B $REF_FILE_B \
		--trj_file_a  $TRJ_FILE_A  --trj_file_b  $TRJ_FILE_B  \
                --sel_base_a "$SEL_BASE_A" --sel_base_b "$SEL_BASE_B" \
                --sel_file "selections/mor_${PART}_without_asp114.txt" \
		--out_name "${OUT_NAME_COMBINED}_${PART}"

done


