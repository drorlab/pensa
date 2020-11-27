from setuptools import setup

setup(name='pensa',
      version='0.1.0',
      description='PENSA - protein ensemble analysis',
      url='http://github.com/drorlab/pensa',
      author='Martin Voegele',
      author_email='mvoegele@stanford.edu',
      license='MIT',
      packages=['pensa'],
      zip_safe=False,
      install_requires=[
        'numpy',
        'scipy>=1.2',
        'mdshare',
        'pyemma',
        'MDAnalysis',
        'matplotlib'
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
      ],)

