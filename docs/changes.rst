.. _changes:

Release notes
=============

v0.1.4

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
