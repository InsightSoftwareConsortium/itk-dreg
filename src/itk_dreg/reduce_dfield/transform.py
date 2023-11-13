#!/usr/bin/env python3

import logging
from typing import List

import itk
import numpy as np

# from registration_methods import register_elastix
from itk_dreg.base.itk_typing import TransformType
import itk_dreg.block.image as block_convert
import itk_dreg.block.image as block_image
from .transform_collection import TransformCollection

logger = logging.getLogger(__name__)

"""
Utility methods to aid distributed registration
using ITK Python, ITKElastix, and Dask.
"""


def collection_to_deformation_field_transform(
    transform_collection: TransformCollection,
    reference_image: itk.Image,
    initial_transform: TransformType,
    scale_factors: List[float],
) -> itk.DisplacementFieldTransform[itk.D, 3]:
    """
    Compose multiple input displacement fields into one transform.

    Follows the same algorithm as `itkTransformToDisplacementFieldFilter`,
    but is compatible with the Python `TransformCollection` implementation.

    Assumptions:
     - input physical regions cover output physical region
     - domain overlap is handled in TransformCollection

    :param transform_collection: The `TransformCollection` to discretely sample into
        an `itk.DisplacementFieldTransform` output.
    :param reference_image: The image to reference to apply spatial metadata to
        the output displacement field image.
    :param initial_transform: The initial transform to apply to the reference image
        to get appropriate initial positioning for the displacement field image.
    :param scale_factors: The desired scale factors to reduce or increase the
        size of the displacement field image grid relative to the reference image.
    :return: An `itk.DisplacementFieldTransform` discretizing the input
        collection of transforms. May be applied in sequence after `initial_transform`
        to map from an input to an output image domain.
    """
    dimension = reference_image.GetImageDimension()
    DEFAULT_VALUE = itk.Vector[itk.D, dimension]([0] * dimension)

    # Get oriented, unallocated image representing the requested bounds
    output_field = block_image.physical_region_to_itk_image(
        physical_region=block_convert.image_to_physical_region(
            image_region=reference_image.GetLargestPossibleRegion(),
            ref_image=reference_image,
            src_transform=initial_transform,
        ),
        spacing=[
            spacing * scale_factor
            for spacing, scale_factor in zip(
                itk.spacing(reference_image), scale_factors
            )
        ],
        direction=reference_image.GetDirection(),
        extend_beyond=True,
        image_type=itk.Image[itk.Vector[itk.D, dimension], dimension],
    )
    output_field.Allocate()
    output_field.FillBuffer(DEFAULT_VALUE)
    logger.info(
        f"Output field has size {itk.size(output_field)}"
        f" and domain "
        f"{block_image.image_to_physical_region(output_field.GetBufferedRegion(), output_field)}"
    )

    # TODO serial bottleneck, parallelization required.
    # To be resolved with `TransformCollection` ITK C++ implementation.
    for x, y, z in np.ndindex(tuple(itk.size(output_field))):
        if np.random.uniform() < 0.005:
            logger.debug(f"Sampling displacement at voxel [{x},{y},{z}]")
        index = [int(x), int(y), int(z)]
        physical_point = output_field.TransformIndexToPhysicalPoint(index)
        try:
            output_field.SetPixel(
                index,
                transform_collection.transform_point(physical_point) - physical_point,
            )
        except ValueError as e:
            logger.debug(f"Failed to sample displacement at [{x},{y},{z}]: {e}")
            continue

    vector_type = itk.template(output_field)[1][0]
    scalar_type = itk.template(vector_type)[1][0]
    output_transform = itk.DisplacementFieldTransform[scalar_type, dimension].New()
    output_transform.SetDisplacementField(output_field)
    return output_transform
