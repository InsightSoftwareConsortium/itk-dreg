#!/usr/bin/env python3

"""
General ITK image utilities for mapping between voxel and physical spaces.

Migrated from https://github.com/AllenNeuralDynamics/aind-ccf-alignment-experiments
"""

import logging
from typing import List, Union, Optional

import itk
import numpy as np
import numpy.typing as npt

from .convert import image_to_physical_region

logger = logging.getLogger(__name__)


def get_sample_bounds(image: itk.Image, transform: Optional[itk.Transform] = None):
    """
    Get the axis-aligned physical boundaries of the space sampled by the ITK image.

    Each voxel in an ITK image is considered to be a sample of the spatial
    volume occupied by that voxel taken at the spatial center of the volume.
    The physical point returned at each discrete voxel coordinate is
    considered to be the physical location of the sample point. We adjust by
    half a voxel in each direction to get the bounds of the space sampled
    by the image.

    Always returns the largest possible physical size of the image.
    Use `itk_dreg.block.convert.image_to_physical_region` to estimate the
    physical size of an image subregion such as a buffered or requested region.

    :param image: The image for which bounds should be estimated.
    :param transform: Transform to apply to the image sample bounds.
    :return: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in (X,Y,Z) physical space for the largest possible
        physical space sampled by the image.
    """
    if image.GetLargestPossibleRegion() != image.GetBufferedRegion():
        logger.warn(
            "Buffered and largest regions do not match."
            "Sample bounds may extend beyond buffered region."
        )
    return image_to_physical_region(
        image.GetLargestPossibleRegion(),
        ref_image=image,
        src_transform=transform,
    )


def get_physical_midpoint(
    image: itk.Image, transform: Optional[itk.Transform] = None
) -> npt.ArrayLike:
    """
    Estimate the physical midpoint of the image in physical space.

    :param image: The image representing an axis-aligned sampling region
        in physical space.
    :param transform: The transform to apply to the sampling region.
    :return: Physical coordinate (X,Y,Z) representing the center of
        the (transformed) image sampling region.
    """
    bounds = get_sample_bounds(image, transform)
    return np.mean(bounds, axis=0)


def block_to_itk_image(
    data: npt.ArrayLike, start_index: npt.ArrayLike, reference_image: itk.Image
) -> itk.Image:
    """
    Return an ITK image view into an array block.

    :param data: The array block data.
    :param start_index: The image index of the first voxel in the data array in ITK access order.
    :param reference_image: Reference ITK image metadata for the output image view.
    :return: An `itk.Image` view into the array block with updated metadata.
    """
    block_image = itk.image_view_from_array(data)

    buffer_offset = [int(val) for val in start_index]
    block_image = itk.change_information_image_filter(
        block_image,
        change_region=True,
        output_offset=buffer_offset,
        change_origin=True,
        output_origin=reference_image.GetOrigin(),
        change_spacing=True,
        output_spacing=reference_image.GetSpacing(),
        change_direction=True,
        output_direction=reference_image.GetDirection(),
    )
    return itk.extract_image_filter(
        block_image, extraction_region=block_image.GetBufferedRegion()
    )


def physical_region_to_itk_image(
    physical_region: npt.ArrayLike,
    spacing: List[float],
    direction: Union[itk.Matrix, npt.ArrayLike],
    extend_beyond: bool = True,
    image_type: Optional[type[itk.Image]] = None,
) -> itk.Image:
    """
    Represent a physical region as an unallocated itk.Image object.

    An itk.Image metadata object represents a mapping from continuous
    physical space with X,Y,Z axes to a discrete voxel space with
    I,J,K axes. This method initializes an itk.Image to sample a
    given three-dimensional space over a discrete voxel grid.

    A procedure is developed as follows:
    1. Process the requested direction and spacing to get the step size
        in physical space corresponding to a step in any voxel direction;
    2. Subdivide the physical region into discrete steps;
    3. Align to the center of the region such that any over/underlap
        is symmetric on opposite sides of the grid;
    4. Compute the origin at the centerpoint of the 0th voxel;
    5. Compute the output voxel size according to the equation:

    size = ((D * S) ^ -1) * (upper_bound - lower_bound)

    The resulting image is a metadata representation of the relationship
    between spaces and has no pixel buffer allocation.

    :param physical_region: A 2x3 voxel array representing axis-aligned inclusive
        upper and lower bounds in (X,Y,Z) physical space.
    :param spacing: The desired output image spacing. Determines how to subdivide
        the physical region into voxels.
    :param direction: The desired output image direction. Assumes that the
        input direction represents some 90 degree orientation mapping from I,J,K to X,Y,Z axes.
    :param extend_beyond: In the event that the physical region cannot be exactly subdivided
        into discrete voxels along any given axis, determines whether the sampling region
        represented by the output image may extend beyond the input physical region to fully
        cover the input physical region with the output voxel grid.
        If True, the region sampled by the output `itk.Image` may extend beyond the input
        physical region by up to one voxel width at each edge.
        If False, the region sampled by the output `itk.Image` may fail to fully cover the
        input physical region and may lay in the interior of the region by up to one voxel
        width at each edge.
    :param image_type: The type of `itk.Image` to construct.
    :return: A new `itk.Image` constructed with metadata to subdivide the input physical region
        into a discrete voxel grid. The image is unbuffered and may be manually allocated
        after its return by calling `image.Allocate()`.
    """
    direction = np.array(direction)
    image_type = image_type or itk.Image[itk.F, 3]
    assert not np.any(np.isclose(spacing, 0)), f"Invalid spacing: {spacing}"
    assert np.all(
        (direction == 0) | (direction == 1) | (direction == -1)
    ), f"Invalid direction: {direction}"

    # Set up unit vectors mapping from voxel to physical space
    voxel_step_vecs = np.matmul(np.array(direction), np.eye(3) * spacing)
    physical_step = np.ravel(
        np.take_along_axis(
            voxel_step_vecs,
            np.expand_dims(np.argmax(np.abs(voxel_step_vecs), axis=1), axis=1),
            axis=1,
        )
    )
    assert physical_step.ndim == 1 and physical_step.shape[0] == 3
    assert np.all(physical_step)

    output_grid_size_f = (
        np.max(physical_region, axis=0) - np.min(physical_region, axis=0)
    ) / np.abs(physical_step)
    output_grid_size = (
        np.ceil(output_grid_size_f) if extend_beyond else np.floor(output_grid_size_f)
    )

    centerpoint = np.mean(physical_region, axis=0)
    output_region = np.array(
        [
            centerpoint - (output_grid_size / 2) * physical_step,
            centerpoint + (output_grid_size / 2) * physical_step,
        ]
    )

    voxel_0_corner = np.array(
        [
            np.min(output_region, axis=0)[dim]
            if physical_step[dim] > 0
            else np.max(output_region, axis=0)[dim]
            for dim in range(3)
        ]
    )
    voxel_0_origin = voxel_0_corner + 0.5 * physical_step
    output_size = np.matmul(
        np.linalg.inv(voxel_step_vecs), (output_region[1, :] - voxel_0_corner)
    )

    output_image = image_type.New()
    output_image.SetOrigin(voxel_0_origin)
    output_image.SetSpacing(spacing)
    output_image.SetDirection(direction)
    output_image.SetRegions(
        [int(size) for size in output_size]
    )  # always 0 index offset
    return output_image
