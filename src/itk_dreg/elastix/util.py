#!/usr/bin/env python3

import logging
from typing import List, Tuple, Dict, Any

import itk

import itk_dreg.block.image as block_image

logger = logging.getLogger(__name__)

ParameterObjectType = itk.ParameterObject
ParameterMapType = Any  # FIXME itk.elxParameterObjectPython.mapstringvectorstring
SerializableParameterMapType = Dict[str, Tuple[str, str]]
SerializableParameterObjectType = List[SerializableParameterMapType]


def compute_initial_translation(
    source_image: itk.Image, target_image: itk.Image
) -> itk.TranslationTransform[itk.D, 3]:
    """
    Compute the initial overlap transform as the translation to sample
    from the center of the target image to the center of the source image.

    Assumes content is centered in source and target images.

    :param source_image: The image serving as the starting point for translation.
        I.e., the "fixed" image to register.
    :param target_image: The image serving as the endpoint for translation.
        I.e., the "moving" image to register.
    """
    target_midpoint = block_image.get_physical_midpoint(target_image)
    source_midpoint = block_image.get_physical_midpoint(source_image)

    translation_transform = itk.TranslationTransform[itk.D, 3].New()
    translation_transform.Translate(source_midpoint - target_midpoint)

    return translation_transform


def get_elx_itk_transforms(
    registration_method: itk.ElastixRegistrationMethod,
    itk_transform_types: List[itk.Transform],
) -> itk.CompositeTransform:
    """
    Convert Elastix registration results to an ITK composite transform stack
    of known, corresponding types.

    :param registration_method: The Elastix registration method previously used
        for registration computation.
    :param itk_transform_types: Ordered list of `itk.Transform` types corresponding
        to Elastix transform parameter object outputs.
        `None` indicates that the given Nth transform should be ignored, such as
        in the case where an advanced initial transform is provided to Elastix.
    :return: An `itk.CompositeTransform` representing Elastix results mapping from
        fixed image to moving image space.
    """
    value_type = itk.D
    dimension = 3

    if registration_method.GetNumberOfTransforms() != len(itk_transform_types):
        raise ValueError(
            f"Elastix to ITK Transform conversion failed: "
            f"Found {registration_method.GetNumberOfTransforms()} Elastix transforms to convert from "
            f"and {len(itk_transform_types)} ITK transforms to convert to"
        )

    itk_composite_transform = itk.CompositeTransform[value_type, dimension].New()

    try:
        for transform_index, itk_transform_type in enumerate(itk_transform_types):
            if not itk_transform_type:  # skip on None
                continue

            elx_transform = registration_method.GetNthTransform(transform_index)
            itk_base_transform = registration_method.ConvertToItkTransform(
                elx_transform
            )
            itk_transform = itk_transform_type.cast(itk_base_transform)
            itk_composite_transform.AddTransform(itk_transform)
    except RuntimeError as e:  # handle bad cast
        logger.error(e)
        return None

    return itk_composite_transform


def get_elx_parameter_maps(
    registration_method: itk.ElastixRegistrationMethod,
) -> List[itk.ParameterObject]:
    """
    Return a series of transform parameter results from Elastix registration.

    :param registration_method: The Elastix registration method previously used
        to compute registration.
    :return: A list of Elastix transform parameter objects from Elastix registration.
    """
    transform_parameter_object = registration_method.GetTransformParameterObject()
    output_parameter_maps = [
        transform_parameter_object.GetParameterMap(parameter_map_index)
        for parameter_map_index in range(
            transform_parameter_object.GetNumberOfParameterMaps()
        )
    ]
    return output_parameter_maps


def flatten_composite_transform(
    transform: itk.Transform,
) -> itk.CompositeTransform[itk.D, 3]:
    """
    Recursively flatten an `itk.CompositeTransform` that may contain
    `itk.CompositeTransform` members so that the output represents a
    single layer of non-composite transforms.

    :param transform: A transform to flatten. If the transform is not
        an `itk.CompositeTransform` then the result will be a simple
        one-level `itk.CompositeTransform` wrapping the input transform.
    :return: An `itk.CompositeTransform` representation of the input
        that is guaranteed to not contain any `itk.CompositeTransform`
        members.
    """
    inner_transforms = _flatten_composite_transform_recursive(transform)

    output_transform = itk.CompositeTransform[itk.D, 3].New()
    for transform in inner_transforms:
        output_transform.AppendTransform(transform)
    return output_transform


def _flatten_composite_transform_recursive(
    transform: itk.Transform,
) -> List[itk.Transform]:
    """
    Internal implementation detail for flattening composite transforms.
    """
    t = None
    try:
        t = itk.CompositeTransform[itk.D, 3].cast(transform)
    except RuntimeError:
        return [transform]

    transform_list = []
    for index in range(t.GetNumberOfTransforms()):
        transform_list += [
            *_flatten_composite_transform_recursive(t.GetNthTransform(index))
        ]
    return transform_list
