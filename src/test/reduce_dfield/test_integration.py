#!/usr/bin/env python3

import sys
import tempfile

import itk
itk.auto_progress(2)

import numpy as np
import dask

sys.path.append('./src')
import itk_dreg.itk
import itk_dreg.reduce_dfield.dreg
from itk_dreg.register import register_images

sys.path.append("./test")
from util import mock as dreg_mock

"""
Test the `itk_dreg` registration scheduling framework.
"""

def test_run_dreg():
    dask.config.set(scheduler='single-threaded')

    fixed_arr = np.ones([100]*3)
    moving_arr = np.ones([50]*3)
    
    register_method = dreg_mock.CountingBlockPairRegistrationMethod()
    reduce_method = itk_dreg.reduce_dfield.dreg.ReduceToDisplacementFieldMethod()

    registration_result = None
    registration_schedule = None

    with tempfile.TemporaryDirectory() as testdir:
        FIXED_FILEPATH = f'{testdir}/fixed_image.mha'
        MOVING_FILEPATH = f'{testdir}/moving_image.mha'
        fixed_image = itk.image_view_from_array(fixed_arr)
        itk.imwrite(fixed_image, FIXED_FILEPATH, compression=False)
        fixed_cb = lambda : itk_dreg.itk.make_reader(FIXED_FILEPATH)
        moving_image = itk.image_view_from_array(moving_arr)
        moving_image.SetSpacing([2]*3)
        itk.imwrite(moving_image, MOVING_FILEPATH, compression=False)
        moving_cb = lambda : itk_dreg.itk.make_reader(MOVING_FILEPATH)

        registration_schedule = register_images(
            fixed_chunk_size=(10,20,100),
            initial_transform=itk.TranslationTransform[itk.D,3].New(),
            moving_reader_ctor=moving_cb,
            fixed_reader_ctor=fixed_cb,
            block_registration_method=register_method,
            reduce_method=reduce_method,
            overlap_factors=[0.1]*3,
            displacement_grid_scale_factors=[10.0,10.0,10.0]
        )

        registration_result = registration_schedule.registration_result.compute()

        print(registration_result)

    assert register_method.num_calls == np.product(registration_schedule.fixed_da.numblocks)
    assert registration_result.status.shape == registration_schedule.fixed_da.numblocks

