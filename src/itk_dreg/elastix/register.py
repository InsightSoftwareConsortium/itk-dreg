#!/usr/bin/env python3

import os
import logging
from typing import List, Optional

import itk
import numpy as np

from itk_dreg.base.itk_typing import ImageType, TransformType
from itk_dreg.base.image_block_interface import (
    BlockPairRegistrationResult,
    BlockRegStatus,
    BlockInfo,
)
from itk_dreg.base.registration_interface import BlockPairRegistrationMethod
import itk_dreg.block.convert as block_convert
import itk_dreg.block.image as block_image

from itk_dreg.elastix.serialize import (
    list_to_parameter_object,
    SerializableParameterObjectType,
)
from itk_dreg.elastix.util import get_elx_itk_transforms

logger = logging.getLogger(__name__)
worker_logger = logging.getLogger("distributed.worker")


class ElastixRegistrationResult(BlockPairRegistrationResult):
    """
    Block pair registration result extended with Elastix-specific results.
    """

    def __init__(
        self, registration_method: itk.ElastixRegistrationMethod, **kwargs: dict
    ):
        """
        :param registration_method: The filter used to run registration.
        :param kwargs: Parameters to forward to `BlockPairRegistrationResult` initialization.
        """
        super().__init__(**kwargs)
        self.registration_method = registration_method


"""
ITKElastix registration implementation for `itk-dreg` registration framework
"""


class ElastixDRegBlockPairRegistrationMethod(BlockPairRegistrationMethod):
    def __call__(
        self,
        # ITK-DReg inherited parameters
        fixed_subimage: ImageType,
        moving_subimage: ImageType,
        initial_transform: TransformType,
        block_info: BlockInfo,
        # Elastix-DReg parameters
        log_directory: Optional[str] = None,
        elx_parameter_object_serial: SerializableParameterObjectType = None,
        itk_transform_types: List[type] = None,
        preprocess_initial_transform: bool = False,  # TODO
        **kwargs,
    ) -> BlockPairRegistrationResult:
        """
        Compute a series of ITKElastix transforms mapping
        from the moving image to the fixed image.

        :param fixed_subimage: The padded subimage in the fixed image domain.
            The subimage requested region indicates the unpadded region of interest.
        :param moving_subimage: The padded subimage in the moving image domain.
            The subimage requested region indicates the unpadded region of interest.
        :param initial_transform: The initial fixed-to-moving transform.
        :param block_info: Metadata describing the block position in fixed image voxel space.
        :param log_directory: The log directory for Elastix outputs, if any.
        :param elx_parameter_object_spec: A serializable list of Python dictionaries
            describing parameter map configurations for ITKElastix registration stages.
            See generation with `itk_dreg.elastix.util.parameter_object_to_list`.
            Note: May be replaced with `itk.ParameterObject` if the Elastix parameter object
            wrapping becomes serializable in the future (as of 2023.10.22).
        :param itk_transform_types: Ordered list of `itk.Transform` types corresponding
            to the order of Elastix transforms to be optimized.
        :param preprocess_initial_transform: If True, the initial fixed subimage will
            be resampled by the initial transform before registration begins.
            Available as a workaround for limitations regarding the Elastix `AdvancedInitialTransform`.
            Setting to `True` may have severe impact on performance and memory requirements.
        :returns: Registration result with:
            - An optimized fixed-to-moving `itk.CompositeTransform` for the given stage,
                which does not include `initial_transform`. Each stage is an `itk.Transform`
                corresponding to an Elastix registration stage as configured via the
                `elx_parameter_object` input.
                Value is `None` if registration fails.
            - The spatial domain over which the forward transform is valid.
                Value is `None` if registration fails.
            - The registration result status.
        :raises RuntimeError: If the Elastix registration procedure encounters an error.
        """
        worker_logger.info("Entering Elastix registration")

        if not elx_parameter_object_serial:
            raise ValueError("An Elastix parameter object must be provided")
        elx_parameter_object = list_to_parameter_object(elx_parameter_object_serial)
        itk_transform_types = itk_transform_types or []

        LOG_FILENAME = "elxLog.txt"
        block_log_directory = None
        if log_directory:
            block_log_directory = (
                f'{log_directory}/{"-".join(map(str, block_info.chunk_index))}'
            )
            os.makedirs(block_log_directory, exist_ok=True)
            logger.debug(
                f"{block_info.chunk_index}: "
                f"Elastix logs will be written to {block_log_directory}"
            )

        if "ElastixRegistrationMethod" not in dir(itk):
            raise KeyError("Elastix methods not found, please pip install itk-elastix")

        if initial_transform and itk_transform_types[-1]:
            # Cannot directly convert an external init ITK transfrom from Elastix
            itk_transform_types.append(None)

        logger.debug(
            f"{block_info.chunk_index}: "
            f"Register with parameter object:{elx_parameter_object}"
        )

        # preprocess_initial_transform = \
        #    initial_transform and itk.BSplineTransform[itk.D,3,3] in itk_transform_types
        if preprocess_initial_transform:
            worker_logger.warning(
                f"{block_info.chunk_index}: " "Resampling fixed image"
            )
            # B-spline requires Jacobian which is not supported by AdvancedExternalTransform
            # so we must resample the image ourselves and discard the initial transform
            physical_region = block_image.image_to_physical_region(
                fixed_subimage.GetBufferedRegion(), fixed_subimage, initial_transform
            )
            logger.debug(f"Resampling fixed image to initial domain {physical_region}")
            fixed_subimage = itk.resample_image_filter(
                fixed_subimage,
                transform=initial_transform.GetInverseTransform(),  # TODO
                use_reference_image=True,
                reference_image=block_image.physical_region_to_itk_image(
                    physical_region=physical_region,
                    spacing=itk.spacing(fixed_subimage),
                    direction=np.array(fixed_subimage.GetDirection()),
                    extend_beyond=True,
                ),
            )
            itk_transform_types = itk_transform_types[:-1]

        registration_method = itk.ElastixRegistrationMethod[
            type(fixed_subimage), type(moving_subimage)
        ].New(
            fixed_image=fixed_subimage,
            moving_image=moving_subimage,
            parameter_object=elx_parameter_object,
        )

        if initial_transform and not preprocess_initial_transform:
            logger.debug(
                f"{block_info.chunk_index}: "
                f"initial transform {str(initial_transform)}"
            )
            registration_method.SetExternalInitialTransform(initial_transform)

        # If we are debugging, make the buffered subimages available on disk
        # for later review
        if block_log_directory and logger.getEffectiveLevel() == logging.DEBUG:
            import itk_dreg.itk

            logger.info(f"Writing buffered subimages to {block_log_directory}")
            try:
                itk_dreg.itk.write_buffered_region(
                    image=fixed_subimage,
                    filepath=f"{block_log_directory}/fixed_subimage.mha",
                )
                itk_dreg.itk.write_buffered_region(
                    image=moving_subimage,
                    filepath=f"{block_log_directory}/moving_subimage.mha",
                )
            except Exception as e:
                logger.warning(f"Failed to write to {block_log_directory}: {e}")

        if block_log_directory:
            registration_method.SetLogToFile(True)
            registration_method.SetOutputDirectory(block_log_directory)
            registration_method.SetLogFileName(LOG_FILENAME)

        # Run registration with `itk-elastix`, may take a few minutes
        logger.info(f"{block_info.chunk_index}: Running pairwise registration")
        registration_method.Update()

        # Get the ITKElastix result as a composite transform.
        # Does not include the initial transform.
        itk_composite_transform = get_elx_itk_transforms(
            registration_method, itk_transform_types
        )

        # Estimate the domain over which the output transform is valid
        # TODO determine whether the unpadded (requested) region or
        # padded (buffered) region is more reasonable as the "valid transform region".
        # For now we use the padded region as the valid transform region.
        transform_domain = block_image.physical_region_to_itk_image(
            physical_region=block_convert.image_to_physical_region(
                image_region=fixed_subimage.GetBufferedRegion(),
                ref_image=fixed_subimage,
                src_transform=(
                    None if preprocess_initial_transform else initial_transform
                ),
            ),
            spacing=itk.spacing(fixed_subimage),
            direction=fixed_subimage.GetDirection(),
            extend_beyond=True,
        )

        return ElastixRegistrationResult(
            transform=itk_composite_transform,
            transform_domain=transform_domain,
            inv_transform=None,
            inv_transform_domain=None,
            status=BlockRegStatus.SUCCESS,
            registration_method=registration_method,
        )
