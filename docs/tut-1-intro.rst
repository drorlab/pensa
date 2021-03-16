Introduction
============

in this tutorial, we show some common functions included in PENSA, using trajectories of a G protein-coupled receptor (GPCR). We retrieve the molecular dynamics trajectories for this tutorial from `GPCRmd <https://submission.gpcrmd.org/home/>`_, an online platform for collection and curation of GPCR simulations. It is described in more detail `here <https://www.nature.com/articles/s41592-020-0884-y>`_.

.. image:: https://pbs.twimg.com/media/Ej8-VJ5WkAAbgJc?format=jpg&name=large
  :width: 500
  :alt: GPCRmd logo

The example system is the mu-opioid receptor (mOR), once in its apo form and once bound to the ligand `BU72 <https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=9363>`_. 
The structure of this GPCR has been reported by `Huang et al (2015) <https://www.nature.com/articles/nature14886>`_. 
We are going to compare the structural ensembles of the receptor in these two conditions.

This tutorial assumes that you can download the trajectories (see below). If you can't, you can use any other system you have available and adapt the file names and residue selections accordingly.

Modules
-------

We only need to import the module "os" and all functions from PENSA itself which in turn loads all the modules it needs.

  .. code:: python
    
    import os
    from pensa import *

Download
--------

PENSA has a predefined function to download GPCRmd trajectories.

  .. code:: python

    # Define where to save the GPCRmd files
    root_dir = './mor-data'
    # Define which files to download
    md_files = ['11427_dyn_151.psf','11426_dyn_151.pdb', # MOR-apo
                '11423_trj_151.xtc','11424_trj_151.xtc','11425_trj_151.xtc',
                '11580_dyn_169.psf','11579_dyn_169.pdb', # MOR-BU72
                '11576_trj_169.xtc','11577_trj_169.xtc','11578_trj_169.xtc']
    # Download all the files that do not exist yet
    for file in md_files:
        if not os.path.exists(os.path.join(root_dir,file)):
            download_from_gpcrmd(file,root_dir)

