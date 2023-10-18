#!/usr/bin/env python3

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List

import dask.array
import numpy.typing as npt

from .itk_typing import ImageType, TransformType

"""
Define common data structures for managing block regions and registration output.
"""


class BlockRegStatus(IntEnum):
    """
    Status codes indicating the registration outcome from a single block pair.

    TODO: To be extended with more granular error codes for `itk-dreg` infrastructure.
    """

    SUCCESS = 0
    """Registration yielded at least a forward transform result."""

    FAILURE = 1
    """Registration encountered an unspecified error."""


@dataclass
class BlockInfo:
    """
    Header information describing the position of a lazy dask subvolume (block)
    in voxel space with respect to a parent volume.

    Accessors are in NumPy order: (K,J,I) where K is slowest, I is fastest
    """

    chunk_index: List[int]
    """
    The chunk position in the greater image volume in terms of chunk counts.
    For instance, if traversing 2x2x2 volume along the fastest axis first:
     - the 0th chunk would have chunk index (0,0,0),
     - the 1st chunk would have chunk index (0,0,1),
     - the 7th chunk would have chunk index (1,0,0)
    """

    array_slice: List[slice]
    """
    The chunk position in the greater image volume in terms of voxel access.
    For instance, if a 100x100x100 volume is evenly subdivided into 10x10x10 chunks,
    the first chunk would slice along [(0,10,1),(0,10,1),(0,10,1)].
    """


@dataclass
class LocatedBlock:
    """
    Combined header and data access to get a lazy dask volume with respect
    to a parent volume in voxel space.

    Accessors are in NumPy order: (K,J,I) where K is slowest, I is fastest
    """

    loc: BlockInfo
    """
    The location of the block relative to the parent image voxel array.
    """

    arr: dask.array.core.Array
    """
    The dask volume for lazy voxel access.
    """


@dataclass
class BlockPairRegistrationResult:
    """Encapsulate result of fixed-to-moving registration over one block pair."""

    transform: Optional[TransformType]
    """
    The forward transform registration result, if any.
    The forward transform maps from moving to fixed space.
    """

    transform_domain: Optional[ImageType]
    """
    Oriented representation of the domain over which the forward transform is valid.
    `transform_domain` has no voxel data and serves as a metadata representation of an
    oriented bounding box in physical space.
    `transform_domain` must be available if and only if `transform` is available.
    """

    inv_transform: Optional[TransformType]
    """
    The inverse transform registration result, if any.
    The inverse transform maps from fixed to moving space.
    If `inv_transform` is available then `transform` must also be available.
    """

    inv_transform_domain: Optional[ImageType]
    """
    Oriented representation of the domain over which the inverse transform is valid.
    `inv_transform_domain` has no voxel data and serves as a metadata representation of an
    oriented bounding box in physical space.
    `inv_transform_domain` must be available if and only if `inv_transform` is available.
    """

    status: BlockRegStatus
    """Status code indicating registration success or failure."""


@dataclass
class RegistrationTransformResult:
    """
    Encapsulate result of fixed-to-moving registration over all block pairs.
    """

    transform: TransformType
    """
    The forward transform resulting from block postprocessing.
    The forward transform maps from moving to fixed space.
    """

    inv_transform: Optional[TransformType]
    """
    The inverse transform registration result from block postprocessing, if any.
    The inverse transform maps from fixed to moving space.
    If `inv_transform` is available then `transform` must also be available.
    """


@dataclass
class RegistrationResult:
    """
    Encapsulate result of fixed-to-moving registration over all block pairs.
    """

    transforms: RegistrationTransformResult
    """The forward and inverse transforms resulting from registration."""

    status: npt.ArrayLike
    """
    `status` is an ND array where each element reflects the status code output
    for block pair registration over the corresponding input moving chunk.
    
    `status` has the same shape as the moving input array of chunks.
    That is, if the moving input array is subdivided into 2 chunks x 3 chunks x 4 chunks,
    `status` will be an array of voxels with shape [2,3,4].
    """
