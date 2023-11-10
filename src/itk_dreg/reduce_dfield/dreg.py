#!/usr/bin/env python3

import logging
from typing import Iterable, List, Optional

import itk
import numpy as np

from itk_dreg.base.image_block_interface import (
    LocatedBlockResult,
    BlockRegStatus,
    RegistrationTransformResult,
)
from itk_dreg.base.itk_typing import TransformType
from itk_dreg.base.registration_interface import (
    ReduceResultsMethod,
    ConstructReaderMethod,
)
import itk_dreg.block.convert as block_convert

from .transform_collection import TransformCollection, TransformEntry
from .transform import collection_to_deformation_field_transform
from .matrix_transform import (
    estimate_euler_transform_consensus,
    itk_matrix_transform_to_matrix,
    to_itk_euler_transform,
)

"""
Adapter methods to use transform utilities in ITK-DReg framework.

#TODO: Extend for 2D
"""

logger = logging.getLogger(__name__)


class ReduceToDisplacementFieldMethod(ReduceResultsMethod):
    """
    Implements `itk-dreg` registration reduction by composing an
    `itk.DisplacementFieldTransform` from pairwise subimage registration results.
    """

    def __call__(
        self,
        block_results: Iterable[LocatedBlockResult],
        fixed_reader_ctor: ConstructReaderMethod,
        initial_transform: itk.Transform,
        displacement_grid_scale_factors: Optional[List[float]] = [1.0, 1.0, 1.0],
        **kwargs,
    ) -> RegistrationTransformResult:
        target_reader = fixed_reader_ctor()
        forward_transform = reduce_to_deformation_field_transform(
            block_results=block_results,
            reference_image=target_reader.GetOutput(),
            initial_transform=initial_transform,
            scale_factors=displacement_grid_scale_factors,
        )
        return RegistrationTransformResult(
            transform=forward_transform, inv_transform=None
        )


class EulerConsensusReduceResultsMethod(ReduceResultsMethod):
    """
    Implements `itk-dreg` registration reduction by composing an
    `itk.Euler3DTransform` from a pairwise subimage rigid registration results.
    """

    def __call__(
        self, block_results: Iterable[LocatedBlockResult], **kwargs
    ) -> RegistrationTransformResult:
        samples_arr = np.zeros([0, 4, 4], dtype=np.float32)
        for result in map(lambda located_result: located_result.result, block_results):
            logger.debug(f"Attempting to reduce transform {result.transform}")
            transform = result.transform
            if (
                type(result.transform) == itk.CompositeTransform[itk.D, 3]
                and result.transform.GetNumberOfTransforms() == 1
            ):
                transform = itk.Euler3DTransform[itk.D].cast(
                    transform.GetNthTransform(0)
                )
            if transform and type(transform) != itk.Euler3DTransform[itk.D]:
                raise TypeError(
                    f"Could not get rigid consensus with transform type {type(transform)}"
                )
            if transform and result.status == BlockRegStatus.SUCCESS:
                samples_arr = np.vstack(
                    (
                        samples_arr,
                        np.expand_dims(itk_matrix_transform_to_matrix(transform), 0),
                    )
                )
        rigid_consensus_mat = estimate_euler_transform_consensus(samples_arr)
        return RegistrationTransformResult(
            transform=to_itk_euler_transform(rigid_consensus_mat), inv_transform=None
        )


def reduce_to_deformation_field_transform(
    block_results: Iterable[LocatedBlockResult],
    reference_image: itk.Image[itk.F, 3],
    initial_transform: TransformType,
    scale_factors: List[float] = [10, 10, 10],
    default_transform: itk.Transform = None,
) -> itk.DisplacementFieldTransform[itk.D, 3]:
    """
    Resample from a set of block registration results into a deformation field transform.
    """
    default_transform = default_transform or itk.TranslationTransform[itk.D, 3].New()

    organized_transforms = TransformCollection(
        blend_method=TransformCollection.blend_distance_weighted_mean
    )
    for located_result in block_results:
        if located_result.result.status == BlockRegStatus.SUCCESS:
            organized_transforms.push(
                TransformEntry(
                    transform=located_result.result.transform,
                    domain=located_result.result.transform_domain,
                )
            )
            continue
        else:
            # TODO estimate the physical domain for the failed block and
            # supply a stand-in default transform instead
            pass

    if not organized_transforms.transforms:
        raise ValueError("Failed to compose at least one transform for sampling")
    logger.debug(f"Collected domains: {organized_transforms.domains}")
    physical_domains = [
        block_convert.block_to_physical_region(
            block_region=block_convert.image_to_block_region(
                image_region=domain.GetLargestPossibleRegion()
            ),
            ref_image=domain,
        )
        for domain in organized_transforms.domains
    ]
    logger.debug(f"Physical domains: {physical_domains}")
    return collection_to_deformation_field_transform(
        organized_transforms, reference_image, initial_transform, scale_factors
    )


class TransformCollectionReduceResultsMethod(ReduceResultsMethod):
    """
    Return a transform collection of results.

    Note (2023.11.10): `transform_collection` does not yet extend `itk.Transform`.
    This should not be used in production.
    """

    def __call__(self, block_results: Iterable[LocatedBlockResult], **kwargs):
        organized_transforms = TransformCollection(
            blend_method=TransformCollection.blend_distance_weighted_mean
        )
        for located_result in block_results:
            if located_result.result.status == BlockRegStatus.SUCCESS:
                organized_transforms.push(
                    TransformEntry(
                        transform=located_result.result.transform,
                        domain=located_result.result.transform_domain,
                    )
                )
        return RegistrationTransformResult(
            transform=organized_transforms, inv_transform=None
        )
