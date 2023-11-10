#!/usr/bin/env python3

"""
ITK implementation of utilities for the `itk-dreg` framework.

Includes implementations for dealing with
- Streaming reader construction,
- Debugging
"""
import os
import logging
from typing import List

import dask
import dask.array as da
import itk

from itk_dreg.base.itk_typing import ImageReaderType, FloatImage3DType

logger = logging.getLogger(__name__)


def make_reader(
    filepath: str,
    imageio: itk.ImageIOBase = None,
    image_type: type[itk.Image] = FloatImage3DType,
) -> ImageReaderType:
    """
    Create an ITK image reader with initialized metadata.

    :param filepath: The local or remote file to read.
        If the file is on the local disk, an absolute path must be provided.
    :param imageio: Explicitly specifies how the image should be read.
        If `imageio` is not provided then ITK will attempt to determine
        the correct ImageIO reading procedure via the ITK object factory.
    :param image_type: The type of `itk.Image` to read.
    :returns: An `itk.ImageFileReader` object with initialized metadata.
    :raises KeyError: If `itk.ImageFileReader` is not defined for the
        specified `image_type` parameter.
    :raises RuntimeError: If reading image metadata fails.
    """
    if not (filepath.startswith("http") or os.path.isabs(filepath)):
        raise ValueError(f"Expected an absolute path for {filepath}")

    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(filepath)
    if imageio:
        reader.SetImageIO(imageio)
    reader.UpdateOutputInformation()
    return reader


def make_dask_array(
    image_reader: ImageReaderType, chunk_size: List[int] = None
) -> da.Array:
    """
    Create a chunked, unbuffered array representing an image buffer.

    :param image_reader: The `itk.ImageFileReader` image source.
        TODO: Runtime failures observed when `image_reader` has zero
            buffered region but nonzero requested region. Investigate
            relation to NumPy bridge and determine whether this is a
            necessary requirement or can be worked around.
    :chunk_size: The requested size of each subdivided region in the
        result array. Default is 128 along each side.
    :returns: A subdivided `dask.Array` representing the unbuffered
        input image voxel region.
        TODO: Verify that iteration over the lazy array does not
            buffer voxel elements.
    """
    pixel_type = itk.template(image_reader.GetOutput())[1][0]
    dimension = image_reader.GetOutput().GetImageDimension()
    chunk_size = chunk_size or [128] * dimension
    delayed_np_array = dask.delayed(itk.array_view_from_image)(image_reader.GetOutput())
    delayed_dask_array = da.from_delayed(
        delayed_np_array, image_reader.GetOutput().shape, dtype=pixel_type
    )
    return delayed_dask_array.rechunk(chunk_size)


def write_buffered_region(image: itk.Image, filepath: str):
    """
    Write out only the buffered region of an image to disk.

    :param image: The image to write.
        `image.GetBufferedRegion()` may or may not differ from
        `image.GetLargestPossibleRegion()`.
    :param filepath: The destination for the buffered image write.
    :raises RuntimeError: If the image extract or write operation fails.
    """
    roi_image = itk.extract_image_filter(
        image, extraction_region=image.GetBufferedRegion()
    )
    itk.imwrite(roi_image, filepath, compression=True)
