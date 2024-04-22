#!/usr/bin/env python3

import logging
import sys
import tempfile

import itk
import numpy as np
import dask

import itk_dreg.itk
from itk_dreg.register import register_images

from itk_dreg.elastix.register import ElastixDRegBlockPairRegistrationMethod
from itk_dreg.elastix.serialize import parameter_object_to_list

sys.path.append("./test")
from util import mock as dreg_mock

logging.basicConfig(level=logging.INFO)

itk.auto_progress(2)

"""
Test the `itk_dreg` registration scheduling framework.

FIXME: segfaults.
https://github.com/InsightSoftwareConsortium/ITKElastix/issues/255
"""

def test_run_dreg():
    dask.config.set(scheduler='single-threaded')

    fixed_arr = np.ones([100]*3)
    moving_arr = np.random.random_sample([50]*3).astype(np.float32)

    register_method = ElastixDRegBlockPairRegistrationMethod()
    reduce_method = dreg_mock.CountingReduceResultsMethod()

    registration_result = None
    registration_schedule = None

    with tempfile.TemporaryDirectory() as testdir:
        FIXED_FILEPATH = f'{testdir}/fixed_image.mha'
        MOVING_FILEPATH = f'{testdir}/moving_image.mha'
        fixed_image = itk.image_view_from_array(fixed_arr)
        itk.imwrite(fixed_image, FIXED_FILEPATH, compression=False)
        def fixed_cb():
            return itk_dreg.itk.make_reader(FIXED_FILEPATH)
        moving_image = itk.image_view_from_array(moving_arr)
        moving_image.SetSpacing([2]*3)
        itk.imwrite(moving_image, MOVING_FILEPATH, compression=False)
        def moving_cb():
            return itk_dreg.itk.make_reader(MOVING_FILEPATH)

        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterMap(
            parameter_object.GetDefaultParameterMap('rigid')
        )

        registration_schedule = register_images(
            fixed_chunk_size=(10,20,100),
            initial_transform=itk.TranslationTransform[itk.D,3].New(),
            moving_reader_ctor=moving_cb,
            fixed_reader_ctor=fixed_cb,
            reduce_method=reduce_method,
            overlap_factors=[0.1]*3,
            block_registration_method=register_method,
            elx_parameter_object_serial=parameter_object_to_list(parameter_object),
            itk_transform_types=[itk.Euler3DTransform[itk.D]],
        )

        registration_result = registration_schedule.registration_result.compute()

        print(registration_result)

    assert reduce_method.num_calls == 1
    assert registration_result.status.shape == registration_schedule.fixed_da.numblocks

