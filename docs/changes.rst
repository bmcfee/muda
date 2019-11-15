.. _changes:

Release notes
=============

v0.4.1
------
* `#78`_ Fixed a bug in impulse response deformations for annotation metadata. *Brian McFee*

.. _#78: https://github.com/bmcfee/muda/pull/78


v0.4.0
------

* `#73`_ Fixed a bug in impulse response deformations.  *Brian McFee*
* `#74`_ Fixed serialization of JAMS objects when using randomized deformations.  *Brian McFee*
* `#76`_ Future-proofed wavefile IO against upcoming librosa deprecations. *Brian
  McFee*

.. _#76: https://github.com/bmcfee/muda/pull/76
.. _#74: https://github.com/bmcfee/muda/pull/74
.. _#73: https://github.com/bmcfee/muda/pull/73

v0.3.0
------

* `#71`_ `muda.replay()` to regenerate audio deformation from JAMS. *Brian McFee*
* `#70`_ Support random seed preservation. *Brian McFee*
* `#69`_ Support deformation of audio with no annotation. *Brian McFee*
* `#68`_ Pitch_hz namespace deprecation support. *Frank Cwitkowitz*
* `#67`_ Colored noise and impulse response deformers. *Chuy Yeliuy*

.. _#67: https://github.com/bmcfee/muda/pull/67
.. _#68: https://github.com/bmcfee/muda/pull/68
.. _#69: https://github.com/bmcfee/muda/pull/69
.. _#70: https://github.com/bmcfee/muda/pull/70
.. _#71: https://github.com/bmcfee/muda/pull/71



v0.2.0
------
* `#61`_ Exposed JAMS input-output arguments in `load_jam_audio` and `save`.
* `#59`_ Fixed an off-by-one error in background noise sampling -- Vincent Lostanlen

.. _#61: https://github.com/bmcfee/muda/pull/61
.. _#59: https://github.com/bmcfee/muda/pull/59


v0.1.4
------
* `#58`_ Updated for compatibility with JAMS 0.3.0

.. _#58: https://github.com/bmcfee/muda/pull/58

v0.1.3
------

* `#40`_ `BackgroundNoise` now stores sample positions in its output history
* `#44`_ fixed a bug in reconstructing muda-output jams files
* `#47`_ removed dependency on scikit-learn
* `#48`_, `#54`_ converted unit tests from nose to py.test
* `#49`_ `TimeStretch` and `PitchShift` deformers now support multiple values
* `#52`_ added the `Union` class

.. _#40: https://github.com/bmcfee/muda/pull/40
.. _#44: https://github.com/bmcfee/muda/pull/44
.. _#47: https://github.com/bmcfee/muda/pull/47
.. _#48: https://github.com/bmcfee/muda/pull/48
.. _#49: https://github.com/bmcfee/muda/pull/49
.. _#52: https://github.com/bmcfee/muda/pull/52
.. _#54: https://github.com/bmcfee/muda/pull/54


v0.1.2
------
This is a minor bug-fix revision.

* The defaults for `LogspaceTimeStretch` have been changed to a more reasonable setting.
* Track duration is now overridden when loading audio into a jams object.

v0.1.1
------
This is a minor bug-fix revision.

* pypi distribution now includes the `drc_presets` data.

v0.1.0
------
Initial public release.
