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


Architecture
------------



Example usage
-------------

The following example constructs a two-stage deformation pipeline.  The first stage
applies random pitch shifts, while the second stage applies random time stretches.
The pipeline therefore generates 25 examples from the input `j_orig`.

.. code-block:: python

    >>> # Load an example audio file with annotation
    >>> j_orig = muda.load_jam_audio('orig.jams', 'orig.ogg')
    >>> # Construct a deformation pipeline
    >>> pitch_shift = muda.deformers.RandomPitchShift(n_samples=5)
    >>> time_stretch = muda.deformers.RandomTimeStretch(n_samples=5)
    >>> pipeline = muda.Pipeline(steps=[('pitch_shift', pitch_shift),
    ...                                 ('time_stretch', time_stretch)])
    >>> for j_new in pipeline.transform(j_orig):
            process(j_new)


Requirements
------------

* Python dependencies
    * JAMS 0.2
    * librosa 0.4
    * pyrubberband 0.1
    * pysoundfile 0.8
    * jsonpickle
* External application dependencies
    * sox
    * rubberband-cli

API Reference
-------------

.. toctree::
    :maxdepth: 3

    core
    deformers

Contribute
----------
- `Issue Tracker <http://github.com/bmcfee/muda/issues>`_
- `Source Code <http://github.com/bmcfee/muda>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

