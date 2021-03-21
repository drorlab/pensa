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

.. code:: ipython3

    sim_a_rec = get_structure_features("traj/condition-a_receptor.gro", 
                                       "traj/condition-a_receptor.xtc")
    sim_a_rec_feat, sim_a_rec_data = sim_a_rec

.. code:: ipython3

    sim_b_rec = get_structure_features("traj/condition-b_receptor.gro",
                                       "traj/condition-b_receptor.xtc")
    sim_b_rec_feat, sim_b_rec_data = sim_b_rec

Having a look at the shape of the loaded data, we see that the first
dimension is the number of frames. The second dimension is the number of
features. It must be the same for both simulations.

.. code:: ipython3

    for k in sim_a_rec_data.keys(): 
        print(k, sim_a_rec_data[k].shape)

.. code:: ipython3

    for k in sim_b_rec_data.keys(): 
        print(k, sim_b_rec_data[k].shape)

Now do the same only for the transmembrane region.

.. code:: ipython3

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

Single-Atom Features
********************

