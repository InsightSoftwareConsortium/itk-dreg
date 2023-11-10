#!/usr/bin/env python3

import itk
import dask.array
import numpy as np

from .convert import get_target_block_size


def rechunk_to_physical_block(
    array: dask.array.core.Array,
    array_image: itk.Image,
    reference_array: dask.array.core.Array,
    reference_image: itk.Image,
) -> dask.array.core.Array:
    """
    Rechunk the given array/image so that the physical size of each block
    approximately matches the physical size of chunks in the reference array/image.

    :param array: The Dask array to rechunk.
    :param array_image: The image or image view to reference to map from
        physical space to the Dask array voxel space.
    :param reference_array: The Dask array reference with chunk sizes to match.
    :param reference_image: The image or image view to reference to map from
        the reference Dask array voxel space to physical space.
    :return: The input `dask.array.Array` rechunked to approximately match
        physical chunk sizes with the reference array.
    """
    # Determine approximate physical size of a reference image chunk
    itk_ref_chunksize = np.flip(reference_array.chunksize)
    arr_chunksize = get_target_block_size(
        itk_ref_chunksize, reference_image, array_image
    )

    # Rechunk the input array to match physical chunk size
    return dask.array.rechunk(array, chunks=np.flip(arr_chunksize))
