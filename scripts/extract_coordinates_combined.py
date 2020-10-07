import argparse
from pensa import load_selection, extract_coordinates_combined


# -------------#
# --- MAIN --- #
# -------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sel_file",   type=str, default='selection.txt')
    parser.add_argument("--sel_base_a", type=str, default='protein and ' )
    parser.add_argument("--ref_file_a", type=str, default='system_a.psf',  nargs='+')
    parser.add_argument("--trj_file_a", type=str, default='stitched_a.nc', nargs='+')
    parser.add_argument("--sel_base_b", type=str, default='protein and ' )
    parser.add_argument("--ref_file_b", type=str, default='system_b.psf',  nargs='+')
    parser.add_argument("--trj_file_b", type=str, default='stitched_b.nc', nargs='+')
    parser.add_argument("--out_name",   type=str, default='coordinates'  )
    parser.add_argument("--start_frame",type=int, default=0  )
    args = parser.parse_args()

    # Load the selection and generate the strings
    sel_string_a = load_selection(args.sel_file, args.sel_base_a)
    sel_string_b = load_selection(args.sel_file, args.sel_base_b)

    print(args.trj_file_a)
    print(args.trj_file_b)
    
    # Combine the lists of input files and selections
    ref_file_list = args.ref_file_a + args.ref_file_b
    trj_file_list = args.trj_file_a + args.trj_file_b
    sel_string_list = [sel_string_a]*len(args.ref_file_a) + [sel_string_b]*len(args.ref_file_b)
    
    # Extract the coordinates from the trajectories
    extract_coordinates_combined(ref_file_list, trj_file_list, sel_string_list, args.out_name, 
                                 start_frame=args.start_frame)

