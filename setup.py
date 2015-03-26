from setuptools import setup, find_packages

import imp

version = imp.load_source('muda.version', 'muda/version.py')

setup(
    name='muda',
    version=version.version,
    description='Python module for musical data augmentation',
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/muda',
    download_url='http://github.com/bmcfee/muda/releases',
    packages=find_packages(),
    long_description="""A python module for musical data augmentation.""",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'librosa>=0.4',
        'pyrubberband',
        'pandas',
        'pyjams>=0.1',
        'pysoundfile',
        'sklearn',
        'six',
        'jsonpickle',
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)
