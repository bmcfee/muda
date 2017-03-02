.. _examples:

Example usage
=============

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

    >>> # Loading audio from disk with an existing jams
    >>> j_orig = jams.load('existing_jams_file.jams')
    >>> j_orig = muda.load_jam_audio(existing_jams, 'orig.ogg')
    >>> # Ready to go!

    >>> # Loading in-memory audio (y, sr) with an existing jams
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
    ...     muda.save('output_{:02d}.ogg'.format(i),
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

Unions
^^^^^^

`Union` operators are similar to `Pipelines`, in that they allow multiple deformers to be
combined as a single object that generates a sequence of deformations.
The difference between `Union` and `Pipeline` is that a pipeline composes deformations
together, so that a single output is the result of multiple stages of processing;
a union only applies one deformation at a time to produce a single output.

The following example is similar to the pipeline example above:

.. code-block:: python

    >>> # Load an example audio file with annotation
    >>> j_orig = muda.load_jam_audio('orig.jams', 'orig.ogg')
    >>> # Construct a deformation pipeline
    >>> pitch_shift = muda.deformers.RandomPitchShift(n_samples=5)
    >>> time_stretch = muda.deformers.RandomTimeStretch(n_samples=5)
    >>> union = muda.Union(steps=[('pitch_shift', pitch_shift),
    ...                           ('time_stretch', time_stretch)])
    >>> for j_new in union.transform(j_orig):
    ...     process(j_new)

Each of the resulting `j_new` objects produced by the `union` has had either
its pitch shifted by the `pitch_shift` object or its time stretched by the
`time_stretch` object, but not both.

Unions apply deformations in a round-robin schedule, so that the first output
is produced by the first deformer, the second output is produced by the second
deformer, and so on, until the list of deformers is exhausted and the first deformer
produces its second output.


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
    ...     process(j_new)


Saving deformations
^^^^^^^^^^^^^^^^^^^
All deformation objects, including bypasses and pipelines, can be serialized to
plain-text (JSON) format, saved to disk, and reconstructed later.
This is demonstrated in the following example.  

.. code-block:: python

    >>> # Encode an existing pitch shift deformation object
    >>> pitch_shift = muda.deformers.RandomPitchShift(n_samples=5)
    >>> ps_str = muda.serialize(pitch_shift)
    >>> print(ps_str)
    {"params": {"n_samples": 5, "mean": 0.0, "sigma": 1.0},
     "__class__": {"py/type": "muda.deformers.pitch.RandomPitchShift"}}

    >>> # Reconstruct the pitch shifter from its string encoding
    >>> ps2 = muda.deserialize(ps_str)

    >>> # Encode a full pipeline as a string
    >>> pipe_str = muda.serialize(pipeline)
    
    >>> # Decode the string to reconstruct a new pipeline object
    >>> new_pipe = muda.deserialize(pipe_str)
    
    >>> # Process jams with the new pipeline
    >>> for j_new in new_pipe.transform(j_orig):
    ...     process(j_new)

