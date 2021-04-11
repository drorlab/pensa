Featurization
=============

Structure Features
******************

The analysis is not performed on the coordinates directly but on
features derived from these coordinates. PENSA uses the featurization
provided by PyEMMA, so far including: - backbone torsions:
``'bb-torsions'``, - backbone C-alpha distances: ``'bb-distances'``, and
- sidechain torsions: ``'sc-torsions'``.

You can combine these with any other function implemented in PyEMMA,
even if it is not included in PENSA.

The function ``get_structure_features`` loads the names of the features
and their values separately

.. code:: python

    sim_a_rec = get_structure_features("traj/condition-a_receptor.gro", 
                                       "traj/condition-a_receptor.xtc")
    sim_a_rec_feat, sim_a_rec_data = sim_a_rec

.. code:: python

    sim_b_rec = get_structure_features("traj/condition-b_receptor.gro",
                                       "traj/condition-b_receptor.xtc")
    sim_b_rec_feat, sim_b_rec_data = sim_b_rec

Having a look at the shape of the loaded data, we see that the first
dimension is the number of frames. The second dimension is the number of
features. It must be the same for both simulations.

.. code:: python

    for k in sim_a_rec_data.keys(): 
        print(k, sim_a_rec_data[k].shape)

.. code:: python

    for k in sim_b_rec_data.keys(): 
        print(k, sim_b_rec_data[k].shape)

Now do the same only for the transmembrane region.

.. code:: python

    sim_a_tmr = get_structure_features("traj/condition-a_tm.gro", 
                                       "traj/condition-a_tm.xtc")
    sim_b_tmr = get_structure_features("traj/condition-b_tm.gro", 
                                       "traj/condition-b_tm.xtc")
    sim_a_tmr_feat, sim_a_tmr_data = sim_a_tmr
    sim_b_tmr_feat, sim_b_tmr_data = sim_b_tmr
    
    for k in sim_a_rec_data.keys(): 
        print(k, sim_a_rec_data[k].shape)
    for k in sim_b_rec_data.keys(): 
        print(k, sim_b_rec_data[k].shape)
        
        
Water Features
**************

Waters are currently featurized from water density. Unlike residues which 
are fixed to a protein, a single water molecule can move throughout the entire 
simulation box, therefore featurizing a single water molecule does not make sense. 
Instead, it is the spatially conserved internal protein cavities in which water 
molecules occupy that are of interest. Water pocket featurization extracts 
a distribution that represents whether or not a specific protein cavity is occupied 
by a water molecule, and what that water molecule's orientation (polarisation) is. 

For the pdb visualisation, the trajectory needs to be fit to the first frame of the simulation
so that the density and protein align with each other.

Here we featurize the top 3 most probable water sites (top_waters = 3).
Orientation of the waters (water_data - spherical coordinates [radians]) is a 
timeseries distribution. When water is not present at the site, the orientation 
is recorded as 10000.0 to represent an empty state. If write=True, we can 
visualise the pocket occupancies on the reference structure in a pdb file with 
pocket occupancy saved as b_factors. 

You must specify the water model for writing out the grid.
options include:
SPC	
TIP3P
TIP4P	
water	

.. code:: python
    
    struc = "traj/cond-a_water.gro"
    xtc = "traj/cond-a_water.gro"
    water_feat, water_data = get_water_features(structure_input = struc, 
                                                xtc_input = xtc,
                                                top_waters = 1,
                                                atomgroup = "OH2",
                                                write = True,
                                                write_grid_as="TIP3P",
                                                out_name = "11426_dyn_151")

To featurize sites common to both ensembles, we obtain the density grid 
following the preprocessing steps in the density tutorial. This is then input 
and waters are featurized according to the combined ensemble density. Sites are 
therefore conserved across both ensembles and can be compared.

.. code:: python

    struc = "traj/cond-a_water.gro"
    xtc = "dens/cond-a_wateraligned.xtc"
    grid = "dens/ab_grid_OH2_density.xtc"
    water_feat, water_data = get_water_features(structure_input = struc, 
                                                xtc_input = xtc,
                                                top_waters = 5,
                                                atomgroup = "OH2",
                                                grid_input = grid)


Single-Atom Features
********************


For single atoms we use a similar protocol which provides the same functionality 
but ignores orientations as atoms are considered spherically symmetric.
Here we locate the sodium site which has the highest probability. The density is 
written (write=True) using the default density conversion "Angstrom^{-3}" in MDAnalysis.


.. code:: python

    struc = "mor-data/11426_dyn_151.pdb"
    xtc = "mor-data/11423_trj_151.xtc"
    atom_feat, atom_data = get_atom_features(structure_input = struc, 
                                              xtc_input = xtc,
                                              top_atoms = 1,
                                              atomgroup = "SOD",
                                              element = "Na",
                                              write = True,
                                              out_name = "11426_dyn_151")
                                              
                                              
                                              
                                              