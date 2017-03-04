.. _muda:
.. toctree::
    :maxdepth: 3

Musical Data Augmentation
=========================

The `muda` package implements annotation-aware musical data augmentation, as described in
the `muda paper <http://bmcfee.github.io/papers/ismir2015_augmentation.pdf>`_ [1]_.
The goal of this package is to make it easy for practitioners to consistently apply
perturbations to annotated music data for the purpose of fitting statistical models.

.. [1] McFee, B., Humphrey, E.J., and Bello, J.P.
    "A software framework for Musical Data Augmentation."
    16th International Society for Music Information Retrival conference (ISMIR).
    2015.

.. _introduction:

Introduction
------------
.. note:: Before reading ahead, it is recommended to familiarize yourself with the `JAMS documentation <http://pythonhosted.org/jams/>`_.

The design of `muda` is patterned loosely after the `Transformer` abstraction in `scikit-learn <http://scikit-learn.org/stable/>`_.
At a high level, each input example consists of an audio clip (with sampling rate) as a `numpy.ndarray` and its annotations stored 
in JAMS format.  To streamline the deformation process, audio data is first stored within the JAMS object so that only a single payload
needs to be transferred throughout the system.

*Deformation objects* (``muda.core.BaseTransformer``) have a single user-facing method, ``transform()``,
which accepts an input JAMS object and generates a sequence of deformations of that object. 
By operating on JAMS objects, the deformation object can simultaneously modify both the audio and all
of its corresponding annotations.

After applying deformations, the modified audio and annotations can be stored to disk by calling ``muda.save()``.
Alternatively, because transformations are generators, results can be processed online by a stochastic learning algorithm.


Requirements
------------

Installing `muda` via ``pip install muda`` should automatically satisfy the python
dependencies:

    * JAMS 0.2
    * librosa 0.4
    * pyrubberband 0.1
    * pysoundfile 0.8
    * jsonpickle

However, certain deformers require external applications that must be installed
separately.

    * sox
    * rubberband-cli

Examples
--------
.. toctree::
    :maxdepth: 2

    examples

API Reference
-------------

.. toctree::
    :maxdepth: 3

    core
    deformers

Release notes
-------------
.. toctree::
    :maxdepth:: 1

    changes

Contribute
----------
- `Issue Tracker <http://github.com/bmcfee/muda/issues>`_
- `Source Code <http://github.com/bmcfee/muda>`_
