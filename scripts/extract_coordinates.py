import argparse
from pensa import load_selection, extract_coordinates


# -------------#
# --- MAIN --- #
# -------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sel_base", type=str, default='protein and ')
    parser.add_argument("--sel_file", type=str, default='')
    parser.add_argument("--ref_file", type=str, default='system.psf')
    parser.add_argument("--pdb_file", type=str, default='system.pdb')
    parser.add_argument("--trj_file", type=str, default='stitched_310.nc',  nargs='+')
    parser.add_argument("--out_name", type=str, default='coordinates' )
    parser.add_argument("--start_frame",type=int, default=0  )
    args = parser.parse_args()

    # Load the selection and generate the string
    if len(args.sel_file) > 0: 
        sel_string = load_selection(args.sel_file, args.sel_base)
    else:
        sel_string = args.sel_base
    print(sel_string)

    # Extract the coordinates from the trajectory
    extract_coordinates(args.ref_file, args.pdb_file, args.trj_file, 
                        args.out_name, sel_string, args.start_frame)

