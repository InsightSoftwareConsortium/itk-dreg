#!/usr/bin/env python3

from typing import Union

import itk

"""
Define common "union" type hints for floating-point registration
in two or three dimensional space.
"""

ImagePixelType = itk.F
FloatImage2DType = itk.Image[ImagePixelType, 2]
FloatImage3DType = itk.Image[ImagePixelType, 3]
ImageType = Union[FloatImage2DType, FloatImage3DType]

ImageRegion2DType = itk.ImageRegion[2]
ImageRegion3DType = itk.ImageRegion[3]
ImageRegionType = Union[ImageRegion2DType, ImageRegion3DType]

FloatImage2DReaderType = itk.ImageFileReader[FloatImage2DType]
FloatImage3DReaderType = itk.ImageFileReader[FloatImage3DType]
ImageReaderType = Union[FloatImage2DReaderType, FloatImage3DReaderType]

TransformScalarType = itk.D
TransformType = Union[
    itk.Transform[TransformScalarType, 2], itk.Transform[TransformScalarType, 3]
]
