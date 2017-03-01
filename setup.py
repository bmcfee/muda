from setuptools import setup, find_packages

import imp

version = imp.load_source('muda.version', 'muda/version.py')

setup(
    name='muda',
    version=version.version,
    description='Musical data augmentation',
    author='Brian McFee',
    author_email='brian.mcfee@nyu.edu',
    url='http://github.com/bmcfee/muda',
    download_url='http://github.com/bmcfee/muda/releases',
    packages=find_packages(),
    package_data={'': ['deformers/data/*']},
    long_description="""Musical data augmentation.""",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    keywords='audio music sound',
    license='ISC',
    install_requires=[
        'librosa>=0.4',
        'pyrubberband>=0.1',
        'pandas',
        'jams>=0.2',
        'pysoundfile>=0.8',
        'six',
        'jsonpickle',
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)
