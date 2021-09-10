Dimensionality Reduction
========================

Here we show how to calculate the principal components in the space of
backbone torsions. It is also common to calculate principal components
in the space of backbone distances. For the latter, again just change
``'bb-torsions'`` to ``'bb-distances'``. As mentioned above, we only
consider the transmembrane region here, so flexible loops outside the
membrane do not distort the more important slow motions in the receptor
core.

Featurization
^^^^^^^^^^^^^^^^^
First, load the structural features as described in the corresponding 
tutorial:

.. code:: python

    sim_a_tmr = get_structure_features("traj/condition-a_tm.gro", 
                                       "traj/condition-a_tm.xtc",
                                       cossin=True)
    sim_b_tmr = get_structure_features("traj/condition-b_tm.gro", 
                                       "traj/condition-b_tm.xtc",
                                       cossin=True)
    sim_a_tmr_feat, sim_a_tmr_data = sim_a_tmr
    sim_b_tmr_feat, sim_b_tmr_data = sim_b_tmr
    
Note that here we load the cosine/sine of the torsions instead of their 
values in radians.
    
Combined PCA
^^^^^^^^^^^^^^^^^

In the spirit of comparing two simulations, we calculate the principal
components of their joint ensemble of structures.

.. code:: python

    # Combine the data of the different simulations
    combined_data_tors = np.concatenate([sim_a_tmr_data['bb-torsions'],sim_b_tmr_data['bb-torsions']],0)

We can now calculate the principal components of this combined dataset.
The corresponding function returns a PyEMMA PCA object, so you can
combine it with all functionality in PyEMMA to perform more advanced or
specialized analysis.

.. code:: python

    pca_combined = calculate_pca(combined_data_tors)

To find out how relevant each PC is, let’s have a look at their
eigenvalues.

.. code:: python

    pca_eigenvalues_plot(pca_combined, num=12, plot_file='plots/combined_tmr_eigenvalues.pdf')

Let us now have a look at the most relevant features of the first three
principal components. Here, we define a feature as important if its
correlation with the respective PC is above a threshold of 0.4. The
function also plots the correlation analysis for each PC.

.. code:: python

    pca_features(pca_combined,sim_a_tmr_feat['bb-torsions'], 3, 0.4)

Now we can compare how the frames of each ensemble are distributed along
the principal components.

.. code:: python

    compare_projections(sim_a_tmr_data['bb-torsions'],
                        sim_b_tmr_data['bb-torsions'],
                        pca_combined,
                        label_a='A', 
                        label_b='B')

To get a better glimpse on what the Principal components look like, we
would like to visualize them. For that purpose, let us sort the
structures from the trajectories along the principal components instead
of along simulation time. We can then look at the resulting PC
trajectories with a molecular visualization program like VMD.

The trajectory to be sorted does not have to be the same subsystem from
which we calcualted the PCA. Here, we are going to write frames with the
entire receptor, sorted by the PCs of the transmembrane region.

.. code:: python

    _ = sort_trajs_along_common_pc(sim_a_tmr_data['bb-torsions'],
                                   sim_b_tmr_data['bb-torsions'],
                                   "traj/condition-a_receptor.gro",
                                   "traj/condition-b_receptor.gro",
                                   "traj/condition-a_receptor.xtc",
                                   "traj/condition-b_receptor.xtc",
                                   "pca/receptor_by_tmr",
                                   num_pc=3, start_frame=feature_start_frame)

The above function deals with the special case of two input
trajectories. We also provide the functions for a single one (see
below). You use these to calculate PCA for any number of combined
simulations and then sort the single or combined simulations.

Single simulation
^^^^^^^^^^^^^^^^^

Here are the major steps of a PCA demonstrated for a single simulation.

.. code:: python

    sim_a_tmr_data['bb-torsions'].shape

.. code:: python

    pca_a = calculate_pca(sim_a_tmr_data['bb-torsions'])

.. code:: python

    pca_features(pca_a, sim_a_tmr_feat['bb-torsions'], 3, 0.4)

.. code:: python

    _, __, ___ = sort_traj_along_pc(sim_a_tmr_data['bb-torsions'],
                                    "traj/condition-a_receptor.gro", 
                                    "traj/condition-a_receptor.xtc", 
                                    "pca/condition-a_receptor_by_tmr", 
                                    start_frame = feature_start_frame, 
                                    pca=pca_a, num_pc=3)
