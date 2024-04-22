#!/usr/bin/env python3

# Purpose: Simple pytest to validate that `itk-dreg` modules can be loaded.

import sys

sys.path.append("src")

import itk

itk.auto_progress(2)


def test_loadmodule():
    pass
