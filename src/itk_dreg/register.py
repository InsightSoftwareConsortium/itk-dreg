#!/usr/bin/env python3

import math
import logging
import itertools
from dataclasses import dataclass
from typing import Iterator, List, Optional

import itk
import dask
import numpy as np
import numpy.typing as npt
from dask.delayed import Delayed

from itk_dreg.block import image as block_image
from itk_dreg.block import convert as block_convert
from itk_dreg.base.image_block_interface import (
    BlockInfo,
    BlockRegStatus,
    BlockPairRegistrationResult,
    RegistrationTransformResult,
    RegistrationResult,
    LocatedBlockResult,
)
from itk_dreg.base.registration_interface import (
    ConstructReaderMethod,
    BlockPairRegistrationMethod,
    ReduceResultsMethod,
)
import itk_dreg.itk

logger = logging.getLogger(__name__)
worker_logger = logging.getLogger("distributed.worker")

"""
Entry point for ITK-DReg multiresolution registration framework.
"""


@dataclass
class RegistrationScheduleResult:
    """
    Encapsulate the results from registration schedule generation
    prior to actually running image registration.
    """

    registration_result: Delayed
    """The `dask.delayed` callable object to prompt registration."""

    fixed_da: dask.array.Array
    """
    The lazy dask array representation of the fixed image voxel buffer,
    subdivided into subimage "chunks" over which pairwise registration
    tasks are scheduled.

    The fixed image dask array representation is used only for scheduling
    purposes, i.e. determining the number and size of subimages to register.
    Fixed subimage voxel buffers are loaded independently at registration
    task time using the `itk.ImageFileReader` construction mechanism.
    """


def register_images(
    # Methods
    fixed_reader_ctor: ConstructReaderMethod,
    moving_reader_ctor: ConstructReaderMethod,
    block_registration_method: BlockPairRegistrationMethod,
    reduce_method: ReduceResultsMethod,
    # Data
    initial_transform: itk.Transform,
    fixed_chunk_size: Optional[List[int]],
    overlap_factors: Optional[List[float]] = None,
    # Debugging
    debug_iter: Optional[Iterator[bool]] = None,
    **kwargs,
) -> RegistrationResult:
    """
    Register blocks of an input image.

    This is the main entry point into `itk-dreg` registration infrastructure.

    :param fixed_reader_ctor: A callable method to create an unbuffered `itk.ImageFileReader` to
        retrieve the fixed image from local or remote storage.
        See `itk_dreg.base.registration_interface` for more details.
    :param moving_reader_ctor: A callable method to create an unbuffered `itk.ImageFileReader` to
        retrieve the moving image from local or remote storage.
        See `itk_dreg.base.registration_interface` for more details.
    :param block_registration_method: A callable method to register a fixed and a moving subimage together.
        See `itk_dreg.base.registration_interface` for more details.
    :param reduce_method: A callable method to compose results from multiple subimage pairwise registration
        computations into a single `itk.Transform` representation that is valid across the fixed image domain.
        See `itk_dreg.base.registration_interface` for more details.
    :param initial_transform: The initial transform to map from fixed image to moving image space.
        Use an `itk.CompositeTransform` to compose transforms from previous registration stages.
    :param fixed_chunk_size: The desired size of each fixed subimage to register.
        Used to subdivide the fixed image into subimage registration tasks.
        Subdivision schedule may be viewed on the `result.fixed_da` output.
        See `overlap_factors` to additionally configure subimage size.
    :param overlap_factors: The desired overlap along each subimage axis in NumPy access order.
        In practice the actual voxel overlap is split evenly along each image edge such that
        the overlap factor represents total overlap length along each axis.
        For instance, running with `fixed_chunk_size==[100,100,100]` and `overlap_factors==[0.1,0.1,0.1]`
        would result in pairwise registration with fixed subimages of approximately [110,110,110] voxels,
        where the input buffered region is enlarged to include 5 voxels in either side of each axis.
        An overlap that would cause the buffered region to extend outside the largest available image region
        is discarded by cropping to the available region.
    :return: A `RegistrationResult` instance containing the following:
        1. An unbuffered `dask.array.Array` representing the registration schedule according to
            fixed image subdivision into chunks;
        2. A `dask.delayed` result representing the registration task graph. `result.registration_result`
            may be visualized, initiated locally with `.compute()`, or deferred to a cluster for computation
            with the `dask.distributed` library.
    """
    logger.info("Preparing registration task graph...")
    debug_iter = debug_iter or itertools.repeat(False)

    fixed_reader = fixed_reader_ctor()
    image_dimension = fixed_reader.GetOutput().GetImageDimension()
    if image_dimension not in (2, 3):
        raise ValueError(f"Registration is not support for {image_dimension}-D images")

    # Generate `dask.array.Array` subdivided representation of the fixed image
    # for scheduling without loading its buffer
    fixed_chunk_size = fixed_chunk_size or [256] * image_dimension
    fixed_da = itk_dreg.itk.make_dask_array(
        image_reader=fixed_reader, chunk_size=fixed_chunk_size
    )
    overlap_factors = overlap_factors or [0] * fixed_da.ndim
    fixed_block_info_iterable = list(iterate_block_info(fixed_da))

    logger.debug(
        f"Subdivided the fixed image into {fixed_block_info_iterable}"
        " overlapping subimages for registration"
    )

    # Register each subimage pair
    delayed_block_results = [
        dask.delayed(register_subimage)(
            block_info=fixed_block_info,
            initial_transform=initial_transform,
            moving_reader_ctor=moving_reader_ctor,
            fixed_reader_ctor=fixed_reader_ctor,
            block_registration_method=block_registration_method,
            overlap_factors=overlap_factors,
            write_debug=next(debug_iter),
            **kwargs,
        )
        for fixed_block_info in fixed_block_info_iterable
    ]

    delayed_located_block_results = [
        dask.delayed(LocatedBlockResult)(fixed_info=fixed_block_info, result=result)
        for fixed_block_info, result in zip(
            iter(fixed_block_info_iterable), delayed_block_results
        )
    ]

    # Postprocess pairwise registration results into a single `itk.Transform`
    reduced_transform_result = dask.delayed(reduce_method)(
        block_results=delayed_located_block_results,
        fixed_reader_ctor=fixed_reader_ctor,
        initial_transform=initial_transform,
        **kwargs,
    )

    # Compose status codes for output
    block_reg_results = dask.delayed(compose_block_status_output)(
        blocks_shape=fixed_da.numblocks,
        block_loc_list=fixed_block_info_iterable,
        results=delayed_block_results,
    )

    registration_result = dask.delayed(compose_output)(
        reduced_transform_result=reduced_transform_result, status=block_reg_results
    )

    return RegistrationScheduleResult(
        registration_result=registration_result, fixed_da=fixed_da
    )


def iterate_block_info(arr: dask.array) -> Iterator[BlockInfo]:
    """
    Return an iterator over chunks in the input array with metadata.

    :param arr: The input dask array to iterate over by chunks.
        The underlying buffer is not directly interrogated, meaning
        a lazy or unbuffered representation is acceptable for iteration.
    :return: An interator interface for the input array.
        Each iteration yields a new array chunk specifier.
    """
    return (
        BlockInfo(chunk_loc, array_slices)
        for chunk_loc, array_slices in zip(
            np.ndindex(*arr.numblocks), dask.array.core.slices_from_chunks(arr.chunks)
        )
    )


def register_subimage(
    # Methods
    fixed_reader_ctor: ConstructReaderMethod,
    moving_reader_ctor: ConstructReaderMethod,
    block_registration_method: BlockPairRegistrationMethod,
    # Data
    block_info: BlockInfo,
    initial_transform: itk.Transform,
    overlap_factors: Optional[List[float]] = None,
    # debug parameters
    default_result: Optional[BlockPairRegistrationResult] = None,
    **kwargs,
) -> BlockPairRegistrationResult:
    """
    Callback to register one moving block to a fixed image subregion.

    `register_subimage` fetches voxel data representing initialized,
    physically aligned image subregions and then calls into a provided
    registration callback for the actual registration process.

    Typically called indirectly via `register_images`.

    :param fixed_reader_ctor: A callable method to create an unbuffered `itk.ImageFileReader` to
        retrieve the fixed image from local or remote storage.
        See `itk_dreg.base.registration_interface` for more details.
    :param moving_reader_ctor: A callable method to create an unbuffered `itk.ImageFileReader` to
        retrieve the moving image from local or remote storage.
        See `itk_dreg.base.registration_interface` for more details.
    :param block_registration_method: A callable method to register a fixed and a moving subimage together.
        See `itk_dreg.base.registration_interface` for more details.
    :param block_info: Specifies the position and enumeration
        of the subvoxel grid to target for registration within the fixed image.
    :param initial_transform: The initial transform to map from fixed image
        to moving image space.
    :param overlap_factors: The desired overlap along each subimage axis in NumPy access order.
        In practice the actual voxel overlap is split evenly along each image edge such that
        the overlap factor represents total overlap length along each axis.
    :param default_result: The default result to return in the event that subimage registration fails.
    :return: The result of subimage-to-subimage registration.
    """
    worker_logger.info(f'Entering "register subimage" with block {block_info}')

    default_result = default_result or BlockPairRegistrationResult(
        transform=None,
        transform_domain=None,
        inv_transform=None,
        inv_transform_domain=None,
        status=BlockRegStatus.FAILURE,
    )
    overlap_factors = overlap_factors or [0] * block_info.ndim

    if not block_info:
        logger.error("Could not register block: no block info provided")
        return default_result

    # Parse dask inputs
    if any(
        [
            block_slice.step and block_slice.step != 1
            for block_slice in block_info.array_slice
        ]
    ):
        logger.warning(
            "Unexpected dask array slice step detected, proceeding with step size == 1 voxel"
        )

    chunk_loc_str = [str(x) for x in block_info.chunk_index]
    start_index = [
        block_slice.start for block_slice in block_info.array_slice
    ]  # NumPy access order
    padding = [
        math.ceil(data_len * overlap_factor * 0.5)
        for data_len, overlap_factor in zip(block_info.shape, overlap_factors)
    ]  # NumPy access order
    block_region = itk.ImageRegion[int(block_info.ndim)](
        [int(val) for val in np.flip(start_index)],
        [int(val) for val in np.flip(block_info.shape)],
    )  # ITK access order
    padded_region = itk.ImageRegion[int(block_info.ndim)](
        [int(val) for val in np.flip(np.array(start_index) - padding)],
        [
            int(val)
            for val in np.flip(np.array(block_info.shape) + 2 * np.array(padding))
        ],
    )  # ITK access order

    # Represent physical position of fixed voxel block
    fixed_reader = fixed_reader_ctor()
    padded_region.Crop(fixed_reader.GetOutput().GetLargestPossibleRegion())
    if not fixed_reader.GetOutput().GetLargestPossibleRegion().IsInside(padded_region):
        logger.warning(
            f"{chunk_loc_str} -> "
            f"Fixed padded region {padded_region} lies outside {fixed_reader.GetOutput().GetLargestPossibleRegion()}"
        )
        return default_result
    logger.debug(
        f"{chunk_loc_str} -> "
        f"Fixed block has unpadded region {block_region} and padded region {padded_region}"
    )

    fixed_reader.GetOutput().SetRequestedRegion(padded_region)
    fixed_reader.Update()
    fixed_block_image = itk.extract_image_filter(
        fixed_reader.GetOutput(),
        extraction_region=fixed_reader.GetOutput().GetBufferedRegion(),
    )
    if fixed_block_image.GetBufferedRegion() != padded_region:
        logger.warning(
            f"Expected fixed block buffered region {padded_region}"
            f"but read in {fixed_block_image.GetBufferedRegion}"
        )
    fixed_block_image.SetRequestedRegion(block_region)  # ROI for registration

    logger.debug(
        f"{chunk_loc_str} -> "
        f"Fixed block {chunk_loc_str} "
        f"has voxel region {block_convert.image_to_block_region(fixed_block_image.GetBufferedRegion())} "
        f"and physical region {block_convert.image_to_physical_region(fixed_block_image.GetBufferedRegion(), fixed_block_image)}"
    )
    logger.debug(
        f"{chunk_loc_str} -> "
        f"After init from previous stages fixed block has approximate physical bounds "
        f"{block_convert.image_to_physical_region(fixed_block_image.GetBufferedRegion(),fixed_block_image,src_transform=initial_transform)}"
    )

    # Retrieve corresponding moving block
    moving_reader = moving_reader_ctor()
    moving_block_region = block_convert.get_target_block_region(
        block_region=block_convert.image_to_block_region(
            fixed_block_image.GetRequestedRegion()
        ),
        src_image=fixed_block_image,
        target_image=moving_reader.GetOutput(),
        src_transform=initial_transform,
        crop_to_target=True,
    )

    moving_padded_block_region = block_convert.get_target_block_region(
        block_region=block_convert.image_to_block_region(
            fixed_block_image.GetBufferedRegion()
        ),
        src_image=fixed_block_image,
        target_image=moving_reader.GetOutput(),
        src_transform=initial_transform,
        crop_to_target=True,
    )
    moving_padded_region = block_convert.block_to_image_region(
        moving_padded_block_region
    )

    if (
        not moving_reader.GetOutput()
        .GetLargestPossibleRegion()
        .IsInside(moving_padded_region)
    ):
        logger.warning(
            f"{chunk_loc_str} -> "
            f"Moving region {moving_padded_region} lies outside "
            f"largest possible region {moving_reader.GetOutput().GetLargestPossibleRegion()}"
        )
        return default_result
    logger.debug(
        f"{chunk_loc_str}: "
        f"Moving unpadded region: {block_convert.block_to_image_region(moving_block_region)},"
        f" moving padded region: {moving_padded_region}"
    )

    moving_reader.GetOutput().SetRequestedRegion(moving_padded_region)
    moving_reader.Update()
    moving_block_image = itk.extract_image_filter(
        moving_reader.GetOutput(),
        extraction_region=moving_reader.GetOutput().GetBufferedRegion(),
    )

    # TODO determine root cause
    # Handle case where crop in fixed padded region can cause 1-voxel difference at target unpadded border
    moving_unpadded_region = block_convert.block_to_image_region(moving_block_region)
    moving_unpadded_region.Crop(moving_padded_region)
    moving_block_image.SetRequestedRegion(moving_unpadded_region)
    logger.debug(
        f"{chunk_loc_str} -> "
        f"Moving subimage largest {moving_block_image.GetLargestPossibleRegion()}"
        f" buffered {moving_block_image.GetBufferedRegion()}"
        f" requested {moving_block_image.GetRequestedRegion()}"
    )

    logger.debug(
        f"{chunk_loc_str} -> "
        f"Moving block has voxel region {block_convert.image_to_block_region(moving_block_image.GetBufferedRegion())}"
        f" and physical region {block_image.get_sample_bounds(moving_block_image)}"
    )

    try:
        if not np.any(moving_block_image):
            logger.warning(f"{chunk_loc_str} -> no signal observed in moving block")
            return default_result
    except RuntimeError as e:
        logger.error(f"{chunk_loc_str}: {e}")
        raise e

    try:
        # Perform registration
        registration_result = block_registration_method(
            fixed_subimage=fixed_block_image,
            moving_subimage=moving_block_image,
            initial_transform=initial_transform,
            block_info=block_info,
            **kwargs,
        )

        if not issubclass(type(registration_result), BlockPairRegistrationResult):
            raise TypeError(
                f"Received incompatible registration result of type"
                f" {type(registration_result)}: {registration_result}"
            )

        logger.info(
            f"{chunk_loc_str} -> Registration completed with status {registration_result.status}"
        )
        return registration_result
    except Exception as e:
        import traceback

        logger.warning(f"{chunk_loc_str} -> {e}")
        traceback.print_exc()
        return default_result


def compose_block_status_output(
    blocks_shape: List[int],
    block_loc_list: Iterator[BlockInfo],
    results: List[BlockPairRegistrationResult],
) -> npt.ArrayLike:
    """
    Compose status codes from pairwise registration into an ND array.

    Typically called indirectly via `register_images`.

    :param blocks_shape`: The fixed image subdivision into blocks in NumPy order.
        For instance, a 10x10x10 image subdivided into chunks of size 5x10x10
        would have `blocks_shape==(2,1,1)`.
    :param block_loc_list: Iterable providing ordered traversal through
        fixed image blocks after subdivision.
    :param results: Iterable providing ordered traversal through
        registration results corresponding to the order of `block_loc_list`.
    :return: An array with shape `block_shape` where each voxel reflects the
        registration status code result from the corresponding block.
    """
    results_arr = np.zeros(blocks_shape, dtype=np.uint8)
    for block_loc, result in zip(block_loc_list, results):
        try:
            results_arr[block_loc.chunk_index] = int(result.status)
        except TypeError as e:
            logger.error(
                f"Failed to convert status to int: {result}\n\n{result.status}"
            )
            raise TypeError(f"Failed to convert {result}") from e
    return results_arr


def compose_output(
    reduced_transform_result: RegistrationTransformResult, status: npt.ArrayLike
):
    """
    Helper method to compose registration result output.

    :param composed_transform_result: The reduced result of subimage registration
        computations.
    :param status: The composed ND array of registration status codes per block
        in NumPy access order.
    """
    return RegistrationResult(transforms=reduced_transform_result, status=status)
