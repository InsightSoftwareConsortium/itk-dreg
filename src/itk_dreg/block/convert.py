#!/usr/bin/env python3

"""
General ITK image utilities for converting between voxel and physical space specifiers.

Migrated from https://github.com/AllenNeuralDynamics/aind-ccf-alignment-experiments
"""

import itertools
from typing import Callable, Union, List, Optional

import itk
import numpy as np
import numpy.typing as npt

###############################################################################
# Image block streaming helpers
###############################################################################

# Terms:
# - "block": A representation in voxel space with integer image access.
# - "physical": A representation in physical space with 3D floating-point representation.
#
# - "block region": a 2x3 voxel array representing axis-aligned [lower,upper) voxel bounds
#                   in ITK access order.
#                   To mimic NumPy indexing the lower bound is inclusive and the upper
#                   bound is one greater than the last included voxel index.
#                   If "k" is fastest and "i" is slowest:
#                   [ [ lower_k, lower_j, lower_i ]
#                       upper_k, upper_j, upper_i ] ]
#
# - "physical region": a 2x3 voxel array representing axis-aligned inclusive
#                      upper and lower bounds in physical space.
#                      [ [ lower_x, lower_y, lower_z ]
#                          upper_x, upper_y, upper_z ] ]
#
# - "ITK region": an `itk.ImageRegion[3]` representation of a block region.
#                 itk.ImageRegion[3]( [ [lower_k, lower_j, lower_i], [size_k, size_j, size_i] ])
#
# IMPORTANT: Note that all voxel accessors are implemented to assume ITK access order, which is
#            reversed (np.flip) from NumPy conventional access order.
#


def arr_to_continuous_index(index: Union[List, npt.ArrayLike]) -> itk.ContinuousIndex:
    r"""
    Convert Python array-like representation of a continuous index into
    an `itk.ContinuousIndex` representation.

    Workaround for conversion issue:
    ```
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "C:\Users\tom.birdsong\Anaconda3\envs\venv-itk11\Lib\site-packages\itk\itkContinuousIndexPython.py", line 158, in __init__
            _itkContinuousIndexPython.itkContinuousIndexD3_swiginit(self, _itkContinuousIndexPython.new_itkContinuousIndexD3(*args))
                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ValueError: Expecting a sequence of int (or long)

    ```

    :param arr: The list or array representing a continuous index.
        The list or array must be a one dimensional collection of floating point scalar values.
    :returns: An `itk.ContinuousIndex` object representing the index.
    :raises ValueError: if the input does not represent a valid index
    :raises KeyError: if the input index length is not supported by `itk.ContinuousIndex`
    """
    index = np.array(index, dtype=np.float32)
    if index.ndim != 1:
        raise ValueError(f"Expected 1D input index but received {index}")
    itk_index = itk.ContinuousIndex[itk.D, len(index)]()
    for dim, val in enumerate(index):
        itk_index.SetElement(dim, float(val))
    return itk_index


def estimate_bounding_box(
    physical_region: npt.ArrayLike, transform: itk.Transform
) -> npt.ArrayLike:
    """
    Estimate a 3D axis-aligned bounding box for a physical region
    with a transform applied.

    Returns two points representing axis-aligned bounds that contain
    the corners of the transformed region.

    The resulting bounding box may not fully contain all transformed
    image points in the case that a deformable transform is applied.
    Also, an axis-aligned bounding box may not be a good representation
    of an image that is not aligned to physical axes. Use with care.

    :param physical_region: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in physical space.
    :param transform: The transform to apply at bounding box vertices.
    :return: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in physical space after the transform is applied.
    """
    DIMENSION = 3
    NUM_CORNERS = 8
    assert physical_region.ndim == 2 and physical_region.shape == (
        2,
        DIMENSION,
    )
    arr = np.zeros([NUM_CORNERS, DIMENSION])
    for index, (i, j, k) in enumerate(
        itertools.product(
            physical_region[:, 0], physical_region[:, 1], physical_region[:, 2]
        )
    ):
        pt = np.array(transform.TransformPoint([float(val) for val in (i, j, k)]))
        arr[index, :] = pt
    return np.array([np.min(arr, axis=0), np.max(arr, axis=0)])


def block_to_physical_size(
    block_size: npt.ArrayLike,
    ref_image: itk.Image,
    transform: Optional[itk.Transform] = None,
) -> npt.ArrayLike:
    """
    Convert from 3D voxel block size to corresponding size in physical space.

    Naive transform approach assumes that both the input and output regions
    are constrained along x/y/z planes aligned at two point extremes.
    May not be suitable for deformable regions.

    #TODO: Verify that we are handling the offset between voxel centers/edges correctly.

    :param block_size: Size of each edge of a voxel block in ITK access order (I,J,K).
    :param ref_image: The image to reference for voxel-to-physical-space conversion.
    :param transform: The transform to apply to the physical bounds before returning.
    :return: The corresponding size (X,Y,Z) in physical space.
    """
    block_index = [int(x) for x in block_size]

    if transform:
        return np.abs(
            transform.TransformPoint(
                ref_image.TransformIndexToPhysicalPoint(block_index)
            )
            - transform.TransformPoint(itk.origin(ref_image))
        )

    return np.abs(
        ref_image.TransformIndexToPhysicalPoint(block_index) - itk.origin(ref_image)
    )


def physical_to_block_size(
    physical_size: npt.ArrayLike, ref_image: itk.Image
) -> npt.ArrayLike:
    """
    Convert from physical size to corresponding discrete voxel size.

    :param physical_size: Axis-aligned physical size (X,Y,Z).
    :param ref_image: The image to reference for physical-to-voxel-space conversion.
    :return: The discrete voxel size in ITK access order (I,J,K).
    """
    return np.abs(
        ref_image.TransformPhysicalPointToIndex(itk.origin(ref_image) + physical_size)
    )


def block_to_physical_region(
    block_region: npt.ArrayLike,
    ref_image: itk.Image,
    transform: Optional[itk.Transform] = None,
    estimate_bounding_box_method: Callable[
        [npt.ArrayLike, itk.Transform], npt.ArrayLike
    ] = estimate_bounding_box,
) -> npt.ArrayLike:
    """
    Convert from a voxel region to a corresponding axis-aligned physical region.

    :param block_region: a 2x3 voxel array representing axis-aligned [lower,upper)
        voxel bounds in ITK access order (I,J,K).
    :param ref_image: The image to reference for voxel-to-physical-space conversion.
    :param transform: The transform to apply to the physical bounds before returning.
    :param estimate_bounding_box_method: The method to use to approximate an axis aligned
        bounding box. Used only if `transform` is provided.
    :return: A 2x3 voxel array representing axis-aligned inclusive upper and lower bounds in physical space.
    """
    # block region is a 2x3 matrix where row 0 is the lower bound and row 1 is the upper bound
    HALF_VOXEL_STEP = 0.5

    assert block_region.ndim == 2 and block_region.shape == (2, 3)
    block_region = np.array(
        [np.min(block_region, axis=0), np.max(block_region, axis=0)]
    )

    adjusted_block_region = block_region - HALF_VOXEL_STEP

    def index_to_physical_func(row: npt.ArrayLike) -> npt.ArrayLike:
        return ref_image.TransformContinuousIndexToPhysicalPoint(
            arr_to_continuous_index(row)
        )

    physical_region = np.apply_along_axis(
        index_to_physical_func, 1, adjusted_block_region
    )

    if not transform:
        return np.array(
            [np.min(physical_region, axis=0), np.max(physical_region, axis=0)]
        )

    return estimate_bounding_box_method(
        physical_region=physical_region, transform=transform
    )


def physical_to_block_region(
    physical_region: npt.ArrayLike, ref_image: itk.Image
) -> npt.ArrayLike:
    """
    Convert from a physical region to a corresponding voxel block.

    :param physical_region: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in (X,Y,Z) physical space.
    :param ref_image: The image to reference for voxel-to-physical-space conversion.
    :return: a 2x3 voxel array representing axis-aligned [lower,upper)
        voxel bounds in ITK access order (I,J,K).
    """
    HALF_VOXEL_STEP = 0.5

    # block region is a 2x3 matrix where row 0 is the lower bound and row 1 is the upper bound
    assert physical_region.ndim == 2 and physical_region.shape == (2, 3)

    def physical_to_index_func(row: npt.ArrayLike) -> npt.ArrayLike:
        return ref_image.TransformPhysicalPointToContinuousIndex(
            [float(val) for val in row]
        )

    block_region = np.apply_along_axis(physical_to_index_func, 1, physical_region)
    adjusted_block_region = np.array(
        [np.min(block_region, axis=0), np.max(block_region, axis=0)]
    )
    return adjusted_block_region + HALF_VOXEL_STEP


def block_to_image_region(block_region: npt.ArrayLike) -> itk.ImageRegion[3]:
    """
    Convert from 2x3 bounds representation to `itk.ImageRegion` representation.

    :param block_region: a 2x3 voxel array representing axis-aligned [lower,upper)
        voxel bounds in ITK access order (I,J,K).
    :return: An `itk.ImageRegion` representation of the voxel region.
    """
    lower_index = [int(val) for val in np.min(block_region, axis=0)]
    upper_index = [int(val) - 1 for val in np.max(block_region, axis=0)]

    region = itk.ImageRegion[3]()
    region.SetIndex(lower_index)
    region.SetUpperIndex(upper_index)
    return region


def image_to_block_region(image_region: itk.ImageRegion[3]) -> npt.ArrayLike:
    """
    Convert from `itk.ImageRegion` to a 2x3 bounds representation.

    :param image_region: An `itk.ImageRegion` representation of the voxel region.
    :return: A 2x3 voxel array representing axis-aligned [lower,upper)
        voxel bounds in ITK access order (I,J,K).
    """
    return np.array(
        [image_region.GetIndex(), np.array(image_region.GetUpperIndex()) + 1]
    )


def physical_to_image_region(
    physical_region: npt.ArrayLike, ref_image: itk.Image
) -> itk.ImageRegion[3]:
    """
    Convert from a physical region to an `itk.ImageRegion` representation.

    :param physical_region: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in (X,Y,Z) physical space.
    :return: An `itk.ImageRegion` representation of the voxel region.
    """
    return block_to_image_region(
        block_region=physical_to_block_region(
            physical_region=physical_region, ref_image=ref_image
        )
    )


def image_to_physical_region(
    image_region: npt.ArrayLike,
    ref_image: itk.Image,
    src_transform: Optional[itk.Transform] = None,
) -> itk.ImageRegion[3]:
    """
    Convert from an `itk.ImageRegion` to a physical region representation.

    :param image_region: An `itk.ImageRegion` representation of the voxel region.
    :param ref_image: The image to reference for voxel-to-physical-space conversion.
    :param src_transform: The transform to apply to the converted physical region.
    :return: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in (X,Y,Z) physical space.
    """
    return block_to_physical_region(
        block_region=image_to_block_region(image_region=image_region),
        ref_image=ref_image,
        transform=src_transform,
    )


def get_target_block_size(
    block_size: npt.ArrayLike, src_image: itk.Image, target_image: itk.Image
) -> npt.ArrayLike:
    """
    Given a voxel region size in source image space, compute the corresponding
    voxel region size with physical alignment in target image space.

    :param block_size: Size of each edge of a voxel block in ITK access order (I,J,K)
        in source image voxel space.
    :param src_image: The source image to use as a reference to convert between
        the input block size and physical space.
    :param target_image: The target image to use as a reference to convert
        between physical space and the output block size.
    :return: Size of each edge of a voxel block in ITK access order (I,J,K)
        in target image voxel space.
    """
    return physical_to_block_size(
        block_to_physical_size(block_size, src_image), target_image
    )


def get_target_block_region(
    block_region: npt.ArrayLike,
    src_image: itk.Image,
    target_image: itk.Image,
    src_transform: Optional[itk.Transform] = None,
    crop_to_target: bool = False,
) -> npt.ArrayLike:
    """
    Given a voxel region in source image space, compute the corresponding
    voxel region with physical alignment in target image space.

    :param block_region: A 2x3 voxel array representing axis-aligned [lower,upper)
        voxel bounds in ITK access order (I,J,K) in source image voxel space.
    :param src_image: The source image to use as a reference to convert between
        the input block size and physical space.
    :param target_image: The target image to use as a reference to convert
        between physical space and the output block size.
    :param src_transform: A transform to apply to transform from
        source image physical space to target image physical space.
    :param crop_to_target: Whether to crop the resulting voxel region in target
        image space so that the voxel region is guaranteed to lie either fully
        within or fully outside of the target image largest possible voxel region.
    :return: A 2x3 voxel array representing axis-aligned [lower,upper)
        voxel bounds in ITK access order (I,J,K) in target image voxel space.
    """
    target_region = physical_to_block_region(
        physical_region=block_to_physical_region(
            block_region=block_region,
            ref_image=src_image,
            transform=src_transform,
        ),
        ref_image=target_image,
    )

    if crop_to_target:
        # TODO can we preserve continuous index input?
        image_region = block_to_image_region(block_region=target_region)
        image_region.Crop(target_image.GetLargestPossibleRegion())
        target_region = image_to_block_region(image_region=image_region)

    return target_region
