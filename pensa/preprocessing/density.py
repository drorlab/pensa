"""
Methods to obtain a distribution for the water pockets which respresents
a combination of the water occupancy (binary variable) and the water polarisation (continuous variable).

For a water molecule to exist within a water pocket, just the oxygen must occupy the pocket.
If there is ever an instance where two water molecules occupy the same pocket at the same time,
then the water polarisation of the molecule ID that occupies the pocket most often is used.

The methods here are based on the following paper:

    |    Neil J. Thomson, Owen N. Vickery, Callum M. Ives, Ulrich Zachariae:
    |    Ion-water coupling controls class A GPCR signal transduction pathways.
    |    https://doi.org/10.1101/2020.08.28.271510

"""

import numpy as np
from scipy import ndimage as ndi
import os
from gridData import Grid
import MDAnalysis as mda
from MDAnalysis.analysis.density import DensityAnalysis
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis import align
import biotite.structure as struc
import biotite.structure.io as strucio
from tqdm import tqdm


# -- Processing trajectories for density analysis


def _match_sim_lengths(sim1, sim2):
    """
    Make two lists the same length by truncating the longer list to match.


    Parameters
    ----------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    Returns
    -------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    """
    if len(sim1) != len(sim2):
        if len(sim1) > len(sim2):
            sim1 = sim1[0:len(sim2)]
        if len(sim1) < len(sim2):
            sim2 = sim2[0:len(sim1)]
    return sim1, sim2


def _copy_coords(ag):
    """
    Copy the coordinates of the frames in a universe.

    Parameters
    ----------
    ag : Universe.atoms

    Returns
    -------
    array
        Copied atom positions.

    """
    return ag.positions.copy()


def local_maxima_3D(data, order=1):
    """
    Detects local maxima in a 3D array to obtain coordinates for density maxima.

    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima

    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]

    return coords, values


def extract_combined_grid(struc_a, xtc_a, struc_b, xtc_b, atomgroup, write_grid_as, out_name,
                          prot_prox=True, use_memmap=False, memmap='combined_traj.mymemmap'):
    """
    Writes out combined atomgroup density for both input simulations.

    Parameters
    ----------
    struc_a : str
        File name for the reference file (PDB or GRO format).
    xtc_a : str
        File name for the trajectory (xtc format).
    struc_b : str
        File name for the reference file (PDB or GRO format).
    xtc_b : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    write_grid_as : str
        The water model to convert the density into.
        Options are: SPC, TIP3P, TIP4P, water
    out_name : str
        Prefix for written filename.
    prot_prox : bool, optional
        Select only waters within 3.5 Angstroms of the protein. The default is True.
    use_memmap : bool, optional
        Uses numpy memmap to write out a pseudo-trajectory coordinate array.
        This is used for large trajectories to avoid memory errors with large
        python arrays. The default is False.
    memmap : str, default='combined_traj.mymemmap'
        The numpy memmap file for the combined pseudo-trajectory.

    """
    condition_a = mda.Universe(struc_a, xtc_a)
    condition_b = mda.Universe(struc_b, xtc_b)

    if use_memmap is True:
        # Combine both ensembles' atoms into one universe
        combined_conditions = mda.Merge(condition_a.atoms, condition_b.atoms)
        # The density needs to be formed from an even contribution of both conditions
        # otherwise it will be unevely biased towards one condition.
        # So iterate over the smallest simulation length
        smallest_traj_len = min(len(condition_a.trajectory), len(condition_b.trajectory))
        # The shape for memmap pseudo-trajetcory
        array_shape = [smallest_traj_len, len(condition_a.atoms) + len(condition_b.atoms), 3]
        # Write out pseudo-trajetcory
        merged_coords = np.memmap(memmap, dtype='float32', mode='w+',
                                  shape=(array_shape[0], array_shape[1], array_shape[2]))
        # Creating universe with blank timesteps from pseudo-trajectory
        combined_conditions.load_new(merged_coords, format=MemoryReader)

        # Create universe with correct timesteps
        for frameno in tqdm(range(smallest_traj_len)):
            condition_a.trajectory[frameno]
            condition_b.trajectory[frameno]
            # Extract trajectory coordinates at frame [frameno]
            coords_a = condition_a.atoms.positions
            coords_b = condition_b.atoms.positions
            # Then merge the coordinates into one system
            stacked = np.concatenate((coords_a, coords_b), axis=0)
            # Write over blank trajectory with new coordinates
            combined_conditions.trajectory[frameno].positions = stacked

    else:
        # Combine both ensembles' atoms into one universe
        combined_conditions = mda.Merge(condition_a.atoms, condition_b.atoms)
        # Extract trajectory coordinates
        aligned_coords_a = AnalysisFromFunction(_copy_coords,
                                                condition_a.atoms).run().results
        aligned_coords_b = AnalysisFromFunction(_copy_coords,
                                                condition_b.atoms).run().results
        # The density needs to be formed from an even contribution of both conditions
        # otherwise it will be unevely biased towards one condition.
        # So match the simulation lengths first
        sim1_coords, sim2_coords = _match_sim_lengths(aligned_coords_a, aligned_coords_b)

        # Then merge the coordinates into one system
        merged_coords = np.hstack([sim1_coords, sim2_coords])
        # Load in the merged coordinates into our new universe that contains
        # the receptor in both conditions
        combined_conditions.load_new(merged_coords, format=MemoryReader)

    # Grab the density for atomgroup proximal to protein only
    if prot_prox is True:
        _selection = "name " + atomgroup + " and around 3.5 protein"
        density_atomgroup = combined_conditions.select_atoms(_selection, updating=True)
    # Grab the density for atomgroup anywhere in simulation box
    else:
        density_atomgroup = combined_conditions.select_atoms("name " + atomgroup)
    # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
    D = DensityAnalysis(density_atomgroup, delta=1.0)
    D.run(verbose=True)
    D.density.convert_density(write_grid_as)
    D.density.export(out_name + atomgroup + "_density.dx", type="double")


def extract_aligned_coordinates(struc_a, xtc_a, struc_b, xtc_b, xtc_aligned=None, pdb_outname='alignment_ref.pdb'):
    """
    Aligns a trajectory (a) on the average structure of another one (b).

    Parameters
    ----------
    struc_a : str
        File name for the reference file (PDB or GRO format).
    xtc_a : str
        File name for the trajectory (xtc format).
    struc_b : str
        File name for the reference file of the trajectory to be aligned to (PDB or GRO format).
    xtc_b : str
        File name for the trajectory to be aligned to (xtc format).
    xtc_aligned: str, default=None
        File name for the aligned trajectory. If none, it will be constructed from the
    pdb_outname: str, default='alignment_ref.pdb'
        File name for the average structure of the trajectory to be aligned to (PDB format)

    """
    if xtc_aligned is None:
        xtc_aligned = xtc_a[:-4] + '_aligned.xtc'

    # # Before we extract the water densities, we need to first align the trajectories
    # # so that we can featurize water sites in both ensembles using the same coordinates
    condition_a = mda.Universe(struc_a, xtc_a)
    condition_b = mda.Universe(struc_b, xtc_b)

    # Align a onto the average structure of b
    p = condition_b.select_atoms("protein")
    p_avg = np.zeros_like(p.positions)
    # do a quick average of the protein (in reality you probably want to remove PBC and RMSD-superpose)
    for ts in condition_b.trajectory:
        p_avg += p.positions
    p_avg /= len(condition_b.trajectory)
    # temporarily replace positions with the average
    # p.load_new(p_avg)
    p.positions = p_avg
    # write average protein coordinates
    p.write(pdb_outname)
    # just make sure that we have clean original coordinates again (start at the beginning)
    condition_b = mda.Universe(pdb_outname)

    # Align condition a to condition b
    align.AlignTraj(condition_a,  # trajectory to align
                    condition_b,  # reference
                    select='name CA',  # selection of atoms to align
                    filename=xtc_aligned,  # file to write the trajectory to
                    match_atoms=True,  # whether to match atoms based on mass
                    ).run(verbose=True)


def generate_grid(u, atomgroup, write_grid_as=None, out_name=None, prot_prox=True):
    """
    Obtain the grid for atomgroup density.

    Parameters
    ----------
    u : MDAnalysis universe
        Universe to obtain density grid.
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    write_grid_as : str, optional
        If you choose to write out the grid, you must specify the water model
        to convert the density into. The default is None.
    out_name : str, optional
        Prefix for all written filenames. The default is None.
    prot_prox : bool, optional
        Select only waters within 3.5 Angstroms of the protein. The default is True.

    Returns
    -------
    g : grid
        Density grid.

    """

    if prot_prox is True:
        density_atomgroup = u.select_atoms("name " + atomgroup + " and around 3.5 protein", updating=True)
    else:
        density_atomgroup = u.select_atoms("name " + atomgroup)
    # a resolution of delta=1.0 ensures the coordinates of the maxima match the coordinates of the simulation box
    D = DensityAnalysis(density_atomgroup, delta=1.0)
    D.run()
    g = D.density
    # Write the grid if requested
    if write_grid_as is not None:
        D.density.convert_density(write_grid_as)
        D.density.export(out_name + '_' + atomgroup + "_density.dx", type="double")

    return g


def dens_grid_pdb(structure_input, xtc_input, atomgroup, top_atoms=35,
                  grid_input=None, write=False, write_grid_as=None, out_name=None):

    """
    Write out water pockets for the top X most probable atoms (top_atoms).

    Parameters
    ----------
    structure_input : str
        File name for the reference file (PDB or GRO format).
    xtc_input : str
        File name for the trajectory (xtc format).
    atomgroup : str
        Atomgroup selection to calculate the density for (atom name in structure_input).
    top_atoms : int, optional
        Number of atoms to featurize. The default is 35.
    grid_input : str, optional
        File name for the density grid input. The default is None, and a grid is automatically generated.
    write : bool, optional
        If True, a reference pdb will be written out. The default is False.
    write_grid_as : str, optional
        If you choose to write out the grid, you must specify the water model
        to convert the density into. The default is None. Options are suggested if default.
    out_name : str, optional
        Prefix for all written filenames. The default is None.

    Returns
    -------
        feature_names : list of str
            Names of all features
        features_data : numpy array
            Data for all features

    """

    if not write:
        if out_name is None:
            print('WARNING: You are writing results without providing out_name.')

    u = mda.Universe(structure_input, xtc_input)

    if write:
        p = u.select_atoms("protein")
        pdb_outname = out_name + "_WaterSites.pdb"
        p_avg = np.zeros_like(p.positions)
        # do a quick average of the protein (in reality you probably want to remove PBC and RMSD-superpose)
        for ts in u.trajectory:
            p_avg += p.positions
        p_avg /= len(u.trajectory)
        # temporarily replace positions with the average
        # p.load_new(p_avg)
        p.positions = p_avg
        # write average protein coordinates
        p.write(pdb_outname)
        # just make sure that we have clean original coordinates again (start at the beginning)
        u.trajectory.rewind()
        if grid_input is None:
            g = generate_grid(u, atomgroup, write_grid_as, out_name)
        else:
            g = Grid(grid_input)
    elif grid_input is None:
        g = generate_grid(u, atomgroup)
    else:
        g = Grid(grid_input)

    xyz, val = local_maxima_3D(g.grid)
    # Negate the array to get probabilities in descending order
    val_sort = np.argsort(-1 * val.copy())
    newvals = [val[max_val] for max_val in val_sort]
    coords = [xyz[max_val] for max_val in val_sort]
    maxdens_coord_str = [str(item)[1:-1] for item in coords]
    atom_information = []
    atom_dists = []

    if top_atoms > len(coords):
        top_atoms = len(coords)

    print('\n')
    print('Featurizing ', top_atoms, ' Waters')

    for at_no in tqdm(range(top_atoms)):

        print('\n')
        print('Atom no: ', at_no + 1)
        print('\n')

        # Find all water atoms within 3.5 Angstroms of density maxima
        # Shifting the coordinates of the maxima by the grid origin to match
        # the simulation box coordinates
        shifted_coords = coords[at_no] + g.origin
        point_str = str(shifted_coords)[1:-1]
        densval = newvals[at_no]

        atom_ID = "O" + str(at_no + 1)
        atom_location = shifted_coords

        atom_information.append([atom_ID, list(atom_location), densval])

        # Write data out and visualize water sites in pdb
        if write:
            write_atom_to_pdb(pdb_outname, atom_location, atom_ID, atomgroup)
            u_pdb = mda.Universe(pdb_outname)
            u_pdb.add_TopologyAttr('tempfactors')
            # Write values as beta-factors ("tempfactors") to a PDB file
            for res in range(len(atom_information)):
                # Scale the atom resid by the starting resid
                atom_resid = len(u_pdb.residues) - at_no - 1 + res
                u_pdb.residues[atom_resid].atoms.tempfactors = atom_information[res][-1]
            u_pdb.atoms.write(pdb_outname)

    # Return the dictionaries.
    return print('PDB file completed.')


def write_atom_to_pdb(pdb_outname, atom_location, atom_ID, atomgroup):
    """
    Write a new atom to a reference structure to visualise conserved non-protein atom sites.

    Parameters
    ----------
    pdb_outname : str
        Filename of reference structure.
    atom_location : array
        (x, y, z) coordinates of the atom location with respect to the reference structure.
    atom_ID : str
        A unique ID for the atom.
    atomgroup : str
        MDAnalysis atomgroup to describe the atom.

    """

    # PDB_VISUALISATION
    # rescursively add waters to the pdb file one by one as they are processed
    # Read the file into Biotite's structure object (atom array)
    atom_array = strucio.load_structure(pdb_outname)
    res_id = atom_array.res_id[-1] + 1
    # Add an HETATM
    atom = struc.Atom(
        coord=atom_location,
        chain_id="X",
        # The residue ID is the last ID in the file +1
        res_id=res_id,
        res_name=atom_ID,
        hetero=True,
        atom_name=atomgroup,
        element="O"
    )
    atom_array += struc.array([atom])
    # Save edited structure
    strucio.save_structure(pdb_outname, atom_array)


def data_out(filename, data):
    """
    Write out lists of data

    Parameters
    ----------
    filename : str
        Name for the written file.
    data : list of lists
        Data to be written out.

    """
    with open(filename, 'w') as output:
        for row in data:
            output.write(str(row)[1:-1] + '\n')


def convert_to_occ(distr, unocc_no, water=True):
    """
    Convert a distribution of pocket angles and occupancies into just occupancies.

    Parameters
    ----------
    distr : list
        Distribution to convert.
    unocc_no : float
        Value that represents unoccupied in the distribution.

    Returns
    -------
    occ : list
        Distribution representing pocket occupancy.

    """

    occ = np.ones(len(distr))

    if water is True:
        for item in range(len(occ)):
            if distr[item] == unocc_no:
                occ[item] = 0
    else:
        for item in range(len(occ)):
            if distr[item][0] == unocc_no:
                occ[item] = 0

    return list(occ)
