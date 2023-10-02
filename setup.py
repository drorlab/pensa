from setuptools import setup, find_packages

setup(
    name='pensa',
    version='0.3.0',
    description='PENSA - Protein ENSemble Analysis',
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
        'numpy',
        'scipy>=1.2',
        'pandas',
        'matplotlib',
        'deeptime',
        'MDAnalysis',
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
