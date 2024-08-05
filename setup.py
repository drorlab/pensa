from setuptools import setup, find_packages

setup(
    name='pensa',
    version='0.4.0',
    description='exploratory analysis and comparison of biomolecular conformational ensembles.',
    url='http://github.com/drorlab/pensa',
    author='Martin Voegele, Neil Thomson, Sang Truong, Jasper McAvity',
    author_email='martinvoegele1989@gmail.com',
    license='MIT',
    packages=find_packages(
        include=[
            'pensa',
            'pensa.preprocessing',
            'pensa.features',
            'pensa.comparison',
            'pensa.dimensionality',
            'pensa.clusters',
            'pensa.statesinfo',
        ]
    ),
    zip_safe=False,
    install_requires=[
        'numpy==1.22', # density functions in MDAnalysis 2 use np.histogramdd() with keyword normed which is deprecated in numpy 1.21 and removed in numpy 1.24
        'scipy==1.9',
        'pandas==1.4',
        'matplotlib==3.5',
        'deeptime',
        'MDAnalysis==2.2', # some features we use will likely be removed in MDA 3
        'biotite',
        'gpcrmining',
    ],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        # license (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Supported Python versions
        'Programming Language :: Python :: 3',
    ],
)
