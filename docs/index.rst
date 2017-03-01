.. _muda:

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


Example usage
-------------
This section gives a quick introduction to using `muda` through example applications.

Loading data
^^^^^^^^^^^^

In `muda`, all data pertaining to a track is contained within a `jams` object.
Before processing any tracks with `muda`, the jams object must be prepared using one of
`muda.load_jam_audio` or `muda.jam_pack`.  These functions prepare the `jams` object to
contain (deformed) audio and store the deformation history objects.


.. code-block:: python

    >>> # Loading data from disk
    >>> j_orig = muda.load_jam_audio('orig.jams', 'orig.ogg')
    >>> # Ready to go!

    >>> # Loading audio form disk with an existing jams
    >>> j_orig = jams.load('existing_jams_file.jams')
    >>> j_orig = muda.load_jam_audio(existing_jams, 'orig.ogg')
    >>> # Ready to go!

    >>> # Loading in-memory audio with an existing jams
    >>> j_orig = jams.load('existing_jams_file.jams')
    >>> j_orig = muda.jam_pack(existing_jams, _audio=dict(y=y, sr=sr))
    >>> # Ready to go!


Applying a deformation
^^^^^^^^^^^^^^^^^^^^^^
Once the data has been prepared, we are ready to start applying deformations.
This example uses a simple linear pitch shift deformer to generate five perturbations of
an input.  Each deformed example is then saved to disk.

.. code-block:: python

    >>> pitch = muda.deformers.LinearPitchShift(n_samples=5, lower=-1, upper=1)
    >>> for i, jam_out in enumerate(pitch.transform(j_orig)):
            muda.save('output_{:02d}.ogg'.format(i),
    ...               'output_{:02d}.jams'.format(i),
    ...               jam_out)

The deformed audio data can be accessed directly in the dictionary
``jam_out.sandbox.muda._audio``.  Note that a full history of applied transformations 
is recorded within ``jam_out.sandbox.muda`` under the ``state`` and ``history`` objects.

Pipelines
^^^^^^^^^

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

Bypass operators
^^^^^^^^^^^^^^^^
When using pipelines, it is sometimes beneficial to allow a stage to be skipped, so that
the input to one stage can be fed through to the next stage without intermediate
processing.  This is easily accomplished with `Bypass` objects, which first emit the
input unchanged, and then apply the contained deformation as usual.  This is demonstrated
in the following example, which is similar to the pipeline example, except that it
guarantees that each stage is applied to `j_orig` in isolation, as well as in
composition.  It therefore generates 36 examples (including `j_orig` itself as the first
output).

.. code-block:: python

    >>> # Load an example audio file with annotation
    >>> j_orig = muda.load_jam_audio('orig.jams', 'orig.ogg')
    >>> # Construct a deformation pipeline
    >>> pitch_shift = muda.deformers.RandomPitchShift(n_samples=5)
    >>> time_stretch = muda.deformers.RandomTimeStretch(n_samples=5)
    >>> pipeline = muda.Pipeline(steps=[('pitch_shift', muda.deformers.Bypass(pitch_shift)),
    ...                                 ('time_stretch', muda.deformers.Bypass(time_stretch))])
    >>> for j_new in pipeline.transform(j_orig):
            process(j_new)


Saving deformations
^^^^^^^^^^^^^^^^^^^
All deformation objects, including bypasses and pipelines, can be serialized to
plain-text (JSON) format, saved to disk, and reconstructed later.
This is demonstrated in the following example.  

.. code-block:: python

    >>> pipe_str = muda.serialize(pipeline)
    >>> new_pipe = muda.deserialize(pipe_str)
    >>> for j_new in new_pipe.transform(j_orig):
            process(j_new)

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

API Reference
-------------

.. toctree::
    :maxdepth: 3

    muda
    core
    deformers
    changes

Contribute
----------
- `Issue Tracker <http://github.com/bmcfee/muda/issues>`_
- `Source Code <http://github.com/bmcfee/muda>`_
