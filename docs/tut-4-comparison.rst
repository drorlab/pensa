Ensemble Comparison 
===================

Here we compare the two ensembles using measures for the relative
entropy.

You can as well calculate the Kolmogorov-Smirnov metric and the
corresponding p value using the function
``kolmogorov_smirnov_analysis()``.

Another possibility is to compare only the means and standard deviations
of the distributions using ``mean_difference_analysis()``.

Featurization
-------------
First, load the structural features as described in the previous tutorial:

.. code:: python

    sim_a_rec = get_structure_features("traj/condition-a_receptor.gro", 
                                       "traj/condition-a_receptor.xtc")
    sim_b_rec = get_structure_features("traj/condition-b_receptor.gro",
                                       "traj/condition-b_receptor.xtc")
    sim_a_rec_feat, sim_a_rec_data = sim_a_rec
    sim_b_rec_feat, sim_b_rec_data = sim_b_rec

Backbone Torsions
-----------------

We start with the backbone torsions, which we can select via
``'bb-torsions'``. To do the same analysis on sidechain torsions,
replace ``'bb-torsions'`` with ``'sc-torsions'``.

.. code:: python

    # Relative Entropy analysis with torsions
    relen = relative_entropy_analysis(sim_a_rec_feat['bb-torsions'], 
                                      sim_b_rec_feat['bb-torsions'], 
                                      sim_a_rec_data['bb-torsions'], 
                                      sim_b_rec_data['bb-torsions'],
                                      bin_num=10, verbose=False)
    names_bbtors, jsd_bbtors, kld_ab_bbtors, kld_ba_bbtors = relen 

The above function also returns the Kullback-Leibler divergences of A
with respect to B and vice versa.

To find out where the ensembles differ the most, let’s print out the
most different features and the corresponding value.

.. code:: python

    # Print the features with the 12 highest values
    sf = sort_features(names_bbtors, jsd_bbtors)
    for f in sf[:12]: print(f[0], f[1])

To get an overview of how strongly the ensembles differ in which region,
we can plot the maximum deviation of the features related to a certain
residue.

.. code:: python

    # Plot the maximum Jensen-Shannon distance per residue as "B factor" in a PDB file
    ref_filename = "traj/condition-a_receptor.gro"
    out_filename = "receptor_bbtors-deviations_tremd"
    vis = residue_visualization(names_bbtors, jsd_bbtors, ref_filename, 
                                "plots/"+out_filename+"_jsd.pdf", 
                                "vispdb/"+out_filename+"_jsd.pdb",
                                y_label='max. JS dist. of BB torsions')


.. code:: python

    # Save the corresponding data
    np.savetxt('results/'+out_filename+'_relen.csv', 
               np.array(relen).T, fmt='%s', delimiter=',', 
               header='Name, JSD(A,B), KLD(A,B), KLD(B,A)')
    np.savetxt('results/'+out_filename+'_jsd.csv', 
               np.array(vis).T, fmt='%s', delimiter=',', 
               header='Residue, max. JSD(A,B)')

The same method can be used to investigate differences in conformational 
states of the torsion distributions, rather than the entire continuous distributions, 
and the data is plotted in the same manner. 

.. code:: python

    names_bbtors, ssi_bbtors = ssi_ensemble_analysis(a_rec_feat, b_rec_feat,
                                                     a_rec_data, b_rec_data,
                                                     torsions='sc', verbose=True)
                                             
    ref_filename = "traj/condition-a_receptor.gro"
    out_filename = "receptor_sctors-deviations_ssi"
    vis = residue_visualization(data_names, data_ssi, ref_filename,
                                "plots/"+out_filename+"_ssi.pdf",
                                "vispdb/"+out_filename+"_ssi.pdb",
                                y_label='max. SSI of SC torsions')    




Backbone C-alpha Distances
--------------------------

Another common representation for the overall structure of a protein are
the distances between the C-alpha atoms. We can perform the same
analysis on them.

.. code:: python

    # Relative entropy analysis for C-alpha distances
    relen = relative_entropy_analysis(sim_a_rec_feat['bb-distances'], 
                                      sim_b_rec_feat['bb-distances'], 
                                      sim_a_rec_data['bb-distances'], 
                                      sim_b_rec_data['bb-distances'],
                                      bin_num=10, verbose=False)
    names_bbdist, jsd_bbdist, kld_ab_bbdist, kld_ba_bbdist = relen 

.. code:: python

    # Print the features with the 12 highest values
    sf = sort_features(names_bbdist, jsd_bbdist)
    for f in sf[:12]: print(f[0], f[1])

To visualize distances, we need a two-dimensional representation with
the residues on each axis. We color each field with the value of the
Jensen-Shannon distance (but could as well use Kullback-Leibler
divergence, Kolmogorov-Smirnov statistic etc. instead).

.. code:: python

    # Visualize the deviations in a matrix plot
    matrix = distances_visualization(names_bbdist, jsd_bbdist, 
                                     "plots/receptor_jsd-bbdist.pdf",
                                     vmin = 0.0, vmax = 1.0,
                                     cbar_label='JSD')


