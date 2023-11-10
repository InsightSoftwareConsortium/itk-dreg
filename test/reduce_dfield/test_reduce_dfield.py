#!/usr/bin/env python3

# Purpose: Simple pytest to validate that `itk_dreg.reduce_dfield` modules can be loaded.

import sys

sys.path.append("src")

import itk
import numpy as np

itk.auto_progress(2)


def test_import():
    import itk_dreg.reduce_dfield
    import itk_dreg.reduce_dfield.dreg
    import itk_dreg.reduce_dfield.matrix_transform
    import itk_dreg.reduce_dfield.transform_collection
    import itk_dreg.reduce_dfield.transform    

def test_collection_to_deformation_field_transform():
    import itk_dreg.reduce_dfield.transform
    import itk_dreg.reduce_dfield.transform_collection

    INPUT_SIZE = [10] * 3
    SCALE_FACTORS = [2] * 3
    EXPECTED_OUTPUT_SIZE = [size / scale for size, scale in zip(INPUT_SIZE, SCALE_FACTORS)]

    TRANSLATION_COMPONENT = 1
    input_transform = itk.TranslationTransform[itk.D,3].New()
    input_transform.Translate([TRANSLATION_COMPONENT] * 3)
    reference_image = itk.Image[itk.F,3].New()
    reference_image.SetRegions([10,10,10])

    transforms = itk_dreg.reduce_dfield.transform_collection.TransformCollection(
        transform_and_domain_list=[itk_dreg.reduce_dfield.transform_collection.TransformEntry(input_transform, None)]
    )

    output_transform = itk_dreg.reduce_dfield.transform.collection_to_deformation_field_transform(transforms,
                                reference_image=reference_image,
                                initial_transform=itk.TranslationTransform[itk.D,3].New(),
                                scale_factors=SCALE_FACTORS)
    
    assert all([output_size == expected_size
                for output_size, expected_size
                in zip(itk.size(output_transform.GetDisplacementField()), EXPECTED_OUTPUT_SIZE)]),\
                    f'Output has size {itk.size(output_transform.GetDisplacementField())}'
    assert np.all(itk.array_view_from_image(output_transform.GetDisplacementField()) == TRANSLATION_COMPONENT)


