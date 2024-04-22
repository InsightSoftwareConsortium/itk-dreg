#!/usr/bin/env python3

import sys

sys.path.append("src")

import itk
import numpy as np
import pytest

itk.auto_progress(2)

from itk_dreg.reduce_dfield.transform_collection import TransformEntry, TransformCollection

def test_unbounded_transform():
    demo_transform = itk.TranslationTransform[itk.D,3].New()
    demo_transform.Translate([1,1,1])

    transforms = [TransformEntry(demo_transform, None)]
    transform_collection = TransformCollection(
        blend_method=TransformCollection.blend_simple_mean,
        transform_and_domain_list=transforms
    )

    assert len(transform_collection.transforms) == 1
    assert len(transform_collection.domains) == 1
    assert transform_collection.domains[0] is None

    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([0,0,0])) == [1,1,1])
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([5,10,15])) == [6,11,16])
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([-30,-0.1,-0.2])) == [-29,0.9,0.8])

def test_bounded_transform():
    demo_transform = itk.TranslationTransform[itk.D,3].New()
    demo_transform.Translate([1,1,1])
    demo_domain = itk.Image[itk.F,3].New()
    demo_domain.SetOrigin([1,1,1])

    r = itk.ImageRegion[3]()
    r.SetSize([1,1,1])
    demo_domain.SetLargestPossibleRegion(r)
    # no allocate -- use itk.Image as metadata container

    transforms = [TransformEntry(demo_transform, demo_domain)]
    transform_collection = TransformCollection(
        blend_method=TransformCollection.blend_simple_mean,
        transform_and_domain_list=transforms
    )

    assert len(transform_collection.transforms) == 1
    assert len(transform_collection.domains) == 1
    assert transform_collection.domains[0] == demo_domain

    with pytest.raises(Exception):
        print(transform_collection.transform_point([0,0,0]))

    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([1,1,1])) == [2,2,2])

def test_two_bounded_transforms():
    transforms = [TransformEntry(itk.TranslationTransform[itk.D,3].New(), itk.Image[itk.F,3].New()),
                  TransformEntry(itk.TranslationTransform[itk.D,3].New(), itk.Image[itk.F,3].New())]
    transforms[0].transform.Translate([1,1,1])
    transforms[1].transform.Translate([2,2,2])
    transforms[0].domain.SetOrigin([0,0,0])
    transforms[0].domain.SetRegions([2,2,2])
    transforms[1].domain.SetOrigin([1,1,1])
    transforms[1].domain.SetRegions([2,2,2])

    transform_collection = TransformCollection(
        blend_method=TransformCollection.blend_simple_mean,
        transform_and_domain_list=transforms)
    assert len(transform_collection.transforms) == 2
    assert len(transform_collection.domains) == 2

    # Transform over non-overlapping domain region
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([0,0,0])) == [1,1,1])
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([2.4,2.4,2.4])) == [4.4,4.4,4.4])

    # Transform over overlapping domain region
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([1,1,1])) == [2.5,2.5,2.5])

    # Transform over excluded domain region fails
    with pytest.raises(Exception):
        print(transform_collection.transform_point([-1,-1,-1]))

def test_distance_weighted_blending():    
    transforms = [TransformEntry(itk.TranslationTransform[itk.D,3].New(), itk.Image[itk.F,3].New()),
                  TransformEntry(itk.TranslationTransform[itk.D,3].New(), itk.Image[itk.F,3].New())]
    transforms[0].transform.Translate([1,1,1])
    transforms[1].transform.Translate([2,2,2])
    transforms[0].domain.SetOrigin([0,0,0])
    transforms[0].domain.SetRegions([4,4,4])
    transforms[1].domain.SetOrigin([1,1,1])
    transforms[1].domain.SetRegions([4,4,4])
    
    transform_collection = TransformCollection(
        blend_method=TransformCollection.blend_distance_weighted_mean,
        transform_and_domain_list=transforms)
    assert len(transform_collection.transforms) == 2
    assert len(transform_collection.domains) == 2

    # Transform over non-overlapping domain region
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([0,0,0])) == [1,1,1])
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([4.4,4.4,4.4])) == [6.4,6.4,6.4])

    # Transform over excluded domain region fails
    with pytest.raises(Exception):
        print(transform_collection.transform_point([-1,-1,-1]))

    # Transform over overlapping domain region weights by distance to domain edge
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([1,1,1])) == [2.25]*3)
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([2,2,2])) == [3.5]*3)
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([3,3,3])) == [4.75]*3)
    assert np.all(transform_collection.transform_point(itk.Point[itk.F,3]([1,2,3])) == [2.5,3.5,4.5])
