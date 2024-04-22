#!/usr/bin/env python3

# Purpose: Simple pytest to validate that `elx-dreg` modules can be loaded.

import sys

sys.path.append("src")

import itk

itk.auto_progress(2)


def test_loadmodule():
    import itk_dreg.elastix
    import itk_dreg.elastix.util
    import itk_dreg.elastix.register
