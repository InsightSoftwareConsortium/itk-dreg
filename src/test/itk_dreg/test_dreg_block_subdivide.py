#!/usr/bin/env python3

import sys

sys.path.append("src")

import itk
import numpy as np

import itk_dreg.block.convert
import itk_dreg.block.image

itk.auto_progress(2)

"""
Tests to validate `itk_dreg.block.image.image.physical_region_to_itk_image`
method for subdividing an axis-aligned physical region into a voxel grid.

We can use this method to rescale an `itk.Image` voxel grid specification
before any sampling/resampling is performed.
"""

def test_rescale_physical_region_norescale():
    SCALE_FACTORS = [1] * 3
    IMAGE_SIZE = [10] * 3
    input_image = itk.Image[itk.F,3].New()
    input_image.SetRegions(IMAGE_SIZE)

    physical_region = itk_dreg.block.convert.image_to_physical_region(
            image_region=input_image.GetLargestPossibleRegion(),
            ref_image=input_image
        )
    output_image = itk_dreg.block.image.physical_region_to_itk_image(
        physical_region=physical_region,
        spacing = [spacing * scale for spacing, scale in zip(itk.spacing(input_image), SCALE_FACTORS)],
        direction=np.array(input_image.GetDirection()),
        extend_beyond=False
    )

    assert itk.size(output_image) == itk.size(input_image)
    assert itk.origin(output_image) == itk.origin(input_image)
    assert itk.spacing(output_image) == itk.spacing(input_image)
    assert np.all(itk_dreg.block.image.get_sample_bounds(output_image) ==
                  itk_dreg.block.image.get_sample_bounds(input_image))
    
def test_rescale_physical_region_downscale():
    SCALE_FACTORS = [2] * 3
    IMAGE_SIZE = [10] * 3
    input_image = itk.Image[itk.F,3].New()
    input_image.SetRegions(IMAGE_SIZE)

    EXPECTED_SIZE = [5] * 3
    EXPECTED_SPACING = [2] * 3
    EXPECTED_ORIGIN = [0.5] * 3

    physical_region = itk_dreg.block.convert.image_to_physical_region(
            image_region=input_image.GetLargestPossibleRegion(),
            ref_image=input_image
        )
    output_image = itk_dreg.block.image.physical_region_to_itk_image(
        physical_region=physical_region,
        spacing = [spacing * scale for spacing, scale in zip(itk.spacing(input_image), SCALE_FACTORS)],
        direction=np.array(input_image.GetDirection()),
        extend_beyond=False
    )

    assert itk.size(output_image) == EXPECTED_SIZE
    assert itk.spacing(output_image) == EXPECTED_SPACING
    assert itk.origin(output_image) == EXPECTED_ORIGIN

def test_rescale_physical_region_offset():
    SCALE_FACTORS = [2] * 3
    IMAGE_SIZE = [10] * 3
    IMAGE_INDEX = [10] * 3
    input_image = itk.Image[itk.F,3].New()

    image_region = itk.ImageRegion[3]()
    image_region.SetIndex(IMAGE_INDEX)
    image_region.SetSize(IMAGE_SIZE)
    input_image.SetRegions(image_region)
    assert input_image.TransformIndexToPhysicalPoint(IMAGE_INDEX) == IMAGE_INDEX

    EXPECTED_ORIGIN = [10.5] * 3
    EXPECTED_SPACING = SCALE_FACTORS
    EXPECTED_SIZE = [5] * 3

    physical_region = itk_dreg.block.convert.image_to_physical_region(
            image_region=input_image.GetLargestPossibleRegion(),
            ref_image=input_image
        )
    output_image = itk_dreg.block.image.physical_region_to_itk_image(
        physical_region=physical_region,
        spacing = [spacing * scale for spacing, scale in zip(itk.spacing(input_image), SCALE_FACTORS)],
        direction=np.array(input_image.GetDirection()),
        extend_beyond=False
    )

    assert itk.size(output_image) == EXPECTED_SIZE
    assert itk.spacing(output_image) == EXPECTED_SPACING
    assert itk.origin(output_image) == EXPECTED_ORIGIN


def test_rescale_physical_region_requested():
    SCALE_FACTORS = [2] * 3

    IMAGE_SIZE=[100]*3
    input_image = itk.Image[itk.F,3].New()
    input_image.SetRegions(IMAGE_SIZE)

    REQUESTED_REGION = itk.ImageRegion[3]([1]*3,[10]*3)
    assert input_image.GetLargestPossibleRegion().IsInside(REQUESTED_REGION)

    EXPECTED_ORIGIN = [1.5] * 3
    EXPECTED_SPACING = SCALE_FACTORS
    EXPECTED_SIZE = [5] * 3

    physical_region = itk_dreg.block.convert.image_to_physical_region(
            image_region=REQUESTED_REGION,
            ref_image=input_image
        )
    output_image = itk_dreg.block.image.physical_region_to_itk_image(
        physical_region=physical_region,
        spacing = [spacing * scale for spacing, scale in zip(itk.spacing(input_image), SCALE_FACTORS)],
        direction=np.array(input_image.GetDirection()),
        extend_beyond=False
    )

    assert itk.size(output_image) == EXPECTED_SIZE
    assert itk.spacing(output_image) == EXPECTED_SPACING
    assert itk.origin(output_image) == EXPECTED_ORIGIN

    assert np.all(itk_dreg.block.image.get_sample_bounds(output_image) ==
                  itk_dreg.block.convert.image_to_physical_region(
                      REQUESTED_REGION,
                      ref_image=input_image
                  ))
    

def test_rescale_physical_region_with_direction():
    SCALE_FACTORS = [2] * 3

    IMAGE_SIZE=[100]*3
    IMAGE_DIRECTION = np.array([[0,0,-1],[1,0,0],[0,-1,0]])
    input_image = itk.Image[itk.F,3].New()
    input_image.SetRegions(IMAGE_SIZE)
    input_image.SetDirection(IMAGE_DIRECTION)
    assert input_image.TransformIndexToPhysicalPoint([25]*3) == [-25,25,-25]

    REQUESTED_REGION = itk.ImageRegion[3]([25]*3,[15]*3)
    assert input_image.GetLargestPossibleRegion().IsInside(REQUESTED_REGION)
    assert np.all(itk_dreg.block.convert.image_to_physical_region(REQUESTED_REGION, input_image) ==\
                np.array([[-39.5,  24.5, -39.5],
                            [-24.5,  39.5, -24.5]]))

    # The input physical region cannot be evenly subdivided into a voxel grid with the given spacing,
    # so allow the grid to extend beyond the input physical region by up to 1 voxel width.
    EXTEND_BEYOND = True

    EXPECTED_ORIGIN = [-25, 25, -25]
    EXPECTED_SPACING = SCALE_FACTORS
    EXPECTED_SIZE = [8] * 3
    EXPECTED_PHYSICAL_REGION = np.array([
        [-40,  24, -40],
        [-24,  40, -24]
    ])

    physical_region = itk_dreg.block.convert.image_to_physical_region(
            image_region=REQUESTED_REGION,
            ref_image=input_image
        )
    output_image = itk_dreg.block.image.physical_region_to_itk_image(
        physical_region=physical_region,
        spacing = [spacing * scale for spacing, scale in zip(itk.spacing(input_image), SCALE_FACTORS)],
        direction=IMAGE_DIRECTION,
        extend_beyond=EXTEND_BEYOND
    )

    assert itk.size(output_image) == EXPECTED_SIZE
    assert itk.spacing(output_image) == EXPECTED_SPACING
    assert itk.origin(output_image) == EXPECTED_ORIGIN
    assert output_image.GetDirection() == input_image.GetDirection()

    assert np.all(itk_dreg.block.image.get_sample_bounds(output_image) ==
                    EXPECTED_PHYSICAL_REGION)
