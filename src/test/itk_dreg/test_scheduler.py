#!/usr/bin/env python3

import os
import sys
import tempfile
import logging

import itk
import numpy as np
import dask
import pytest
from urllib.request import urlretrieve

import itk_dreg.itk
import itk_dreg.register
from itk_dreg.base.image_block_interface import BlockRegStatus

sys.path.append("./test")
from util import mock as dreg_mock

itk.auto_progress(2)

"""
Test the `itk_dreg` registration scheduling framework.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# worker_logger = logging.getLogger('distributed.worker')
# worker_logger.setLevel(logging.DEBUG)

PIXEL_TYPE = itk.F
DIMENSION = 3  # 2D is planned but not yet supported (2023.10.20)


@pytest.fixture
def test_input_dir() -> str:
    TEST_INPUT_DIR = "test/data/input"
    os.makedirs(TEST_INPUT_DIR, exist_ok=True)
    yield TEST_INPUT_DIR


@pytest.fixture
def test_output_dir() -> str:
    TEST_OUTPUT_DIR = "test/data/output/itk_dreg"
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    return TEST_OUTPUT_DIR


@pytest.fixture
def fixed_image_filepath(test_input_dir):
    # A small 3D sample MRI image depicting a patient's head
    FIXED_IMAGE_FILEPATH = os.path.abspath(f"{test_input_dir}/HeadMRVolume_d.mha")
    COMPRESSED_FIXED_IMAGE_URL = (
        "https://data.kitware.com/api/v1/item/65328e0a5be10c8fb6ed4f01/download"
    )
    if not os.path.exists(FIXED_IMAGE_FILEPATH):
        with tempfile.TemporaryDirectory(dir=test_input_dir) as tmpdir:
            urlretrieve(COMPRESSED_FIXED_IMAGE_URL, f"{tmpdir}/HeadMRVolume.mha")
            image = itk.imread(f"{tmpdir}/HeadMRVolume.mha")
            itk.imwrite(image, FIXED_IMAGE_FILEPATH, compression=False)
    yield FIXED_IMAGE_FILEPATH


@pytest.fixture
def moving_image_filepath(test_input_dir) -> str:
    # The fixed image, but with an arbitrary translation and rotation applied
    MOVING_IMAGE_FILEPATH = os.path.abspath(
        f"{test_input_dir}/HeadMRVolume_rigid_d.mha"
    )
    COMPRESSED_MOVING_IMAGE_URL = (
        "https://data.kitware.com/api/v1/item/65328e0f5be10c8fb6ed4f04/download"
    )
    if not os.path.exists(MOVING_IMAGE_FILEPATH):
        with tempfile.TemporaryDirectory(dir=test_input_dir) as tmpdir:
            urlretrieve(COMPRESSED_MOVING_IMAGE_URL, f"{tmpdir}/HeadMRVolume_rigid.mha")
            image = itk.imread(f"{tmpdir}/HeadMRVolume_rigid.mha")
            itk.imwrite(image, MOVING_IMAGE_FILEPATH, compression=False)
    yield MOVING_IMAGE_FILEPATH


def test_run_singlethreaded(
    fixed_image_filepath, moving_image_filepath, test_output_dir
):
    dask.config.set(scheduler="single-threaded")
    logging.root.setLevel(logging.INFO)

    # Methods
    import functools

    fixed_reader_ctor = functools.partial(
        itk_dreg.itk.make_reader, filepath=fixed_image_filepath
    )
    moving_reader_ctor = functools.partial(
        itk_dreg.itk.make_reader, filepath=moving_image_filepath
    )
    register_method = dreg_mock.CountingBlockPairRegistrationMethod()
    logger.warning(f"reg def {register_method.default_result}")
    reduce_method = dreg_mock.CountingReduceResultsMethod()
    logger.warning(f"reduce def {reduce_method.default_result}")

    # Data
    fixed_chunk_size = (10, 100, 100)  # TODO investigate failure case (15,15,25)
    initial_transform = itk.TranslationTransform[itk.D, 3].New()
    overlap_factors = [0.1] * 3

    registration_graph = itk_dreg.register.register_images(
        fixed_chunk_size=fixed_chunk_size,
        initial_transform=initial_transform,
        moving_reader_ctor=moving_reader_ctor,
        fixed_reader_ctor=fixed_reader_ctor,
        block_registration_method=register_method,
        reduce_method=reduce_method,
        overlap_factors=overlap_factors,
    )

    itk.auto_progress(0)
    registration_result = registration_graph.registration_result.compute()
    print(registration_result)

    assert register_method.num_calls == np.product(
        registration_graph.fixed_da.numblocks
    )
    assert reduce_method.num_calls == 1
    assert registration_result.status.shape == registration_graph.fixed_da.numblocks


def test_localcluster(fixed_image_filepath, moving_image_filepath):
    import dask.distributed

    cluster = dask.distributed.LocalCluster(n_workers=1, threads_per_worker=1)
    client = dask.distributed.Client(cluster)  # noqa: F841

    # Methods
    import functools

    fixed_reader_ctor = functools.partial(
        itk_dreg.itk.make_reader, filepath=fixed_image_filepath
    )
    moving_reader_ctor = functools.partial(
        itk_dreg.itk.make_reader, filepath=moving_image_filepath
    )
    register_method = dreg_mock.ConstantBlockPairRegistrationMethod()
    reduce_method = dreg_mock.ConstantReduceResultsMethod()

    # Data
    fixed_chunk_size = (10, 100, 100)  # TODO investigate failure case (15,15,25)
    initial_transform = itk.TranslationTransform[itk.D, 3].New()
    overlap_factors = [0.1] * 3

    # logging.root.setLevel(logging.DEBUG)
    registration_graph = itk_dreg.register.register_images(
        fixed_chunk_size=fixed_chunk_size,
        initial_transform=initial_transform,
        moving_reader_ctor=moving_reader_ctor,
        fixed_reader_ctor=fixed_reader_ctor,
        block_registration_method=register_method,
        reduce_method=reduce_method,
        overlap_factors=overlap_factors,
    )

    itk.auto_progress(0)
    registration_result = registration_graph.registration_result.compute()
    print(registration_result)

    assert registration_result.status.shape == registration_graph.fixed_da.numblocks
    assert np.all(registration_result.status == BlockRegStatus.SUCCESS)
