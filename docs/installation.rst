Installation
============

Conda environment
"""""""""""""""""

Create and activate a conda environment:

  .. code:: bash

    conda create --name pensa python=3.7 numpy scipy matplotlib mdtraj==1.9.3 pyemma mdshare MDAnalysis cython biotite -c conda-forge
    conda activate pensa

If you want to use PENSA with Jupyter notebooks:

  .. code:: bash

    conda install jupyter

Option 1: Install the PENSA library from PyPI
"""""""""""""""""""""""""""""""""""""""""""""

This installs the latest released version.

Within the environment created above, execute:

  .. code:: bash

    pip install pensa

To use the example scripts or tutorial folder, you'll have to download them from the repository.

Option 2: Create editable installation from source
""""""""""""""""""""""""""""""""""""""""""""""""""

This installs the latest version from the repository, which might not yet be officially released.

Within the environment created above, execute:

  .. code:: bash

    git clone https://github.com/drorlab/pensa.git  
    cd pensa
    pip install -e . 



