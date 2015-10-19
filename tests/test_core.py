#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''MUDA core tests'''

import numpy as np

import muda

import jams


def test_jampack():

    jam = jams.JAMS()

    sr = 22050
    y = np.zeros(sr)

    muda.jam_pack(jam, y=y, sr=sr)

    # Make sure the jam now has a mudabox
    assert hasattr(jam.sandbox, 'muda')
    assert hasattr(jam.sandbox.muda, 'history')
    assert hasattr(jam.sandbox.muda, 'state')

    assert jam.sandbox.muda['y'] is y
    assert jam.sandbox.muda['sr'] == sr
