#!/usr/bin/env python3

# Purpose: Simple pytest to validate that `itk-dreg` modules can be loaded.

import sys

sys.path.append("src")

import itk

itk.auto_progress(2)


def test_loadmodule():
    import itk_dreg
    import itk_dreg.itk
    import itk_dreg.base.image_block_interface
    import itk_dreg.base.itk_typing
    import itk_dreg.base.registration_interface
    import itk_dreg.block.convert
    import itk_dreg.block.image
    import itk_dreg.register
