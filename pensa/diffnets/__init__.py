"""
diffnets
Supervised and self-supervised autoencoders to identify the mechanistic basis for biochemical differences between protein variants.
"""

# Add imports here
from . import training, data_processing, analysis, utils, exmax, nnutils

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
