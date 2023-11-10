#!/usr/bin/env python3

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List

import itk
import dask.array
import numpy.typing as npt

from .itk_typing import ImageType, TransformType

"""
Define common data structures for managing block regions and registration output.
"""


class BlockRegStatus(IntEnum):
    """
    Status codes indicating the registration outcome from a single block pair.

    TODO: Extend with more granular error codes for `itk-dreg` infrastructure.
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

    @property
    def ndim(self) -> int:
        if len(self.chunk_index) != len(self.array_slice):
            raise ValueError(
                "Observed mismatch between chunk and slice index dimensions"
            )
        return len(self.chunk_index)

    @property
    def shape(self) -> List[int]:
        if any(
            [slice_val.step and slice_val.step != 1 for slice_val in self.array_slice]
        ):
            print()
            raise ValueError(
                "Illegal step size in `BlockInfo`, expected step size of 1"
            )
        return [slice_val.stop - slice_val.start for slice_val in self.array_slice]


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


class BlockPairRegistrationResult:
    """Encapsulate result of fixed-to-moving registration over one block pair."""

    def __init__(
        self,
        status: BlockRegStatus,
        transform: Optional[TransformType] = None,
        transform_domain: Optional[ImageType] = None,
        inv_transform: Optional[TransformType] = None,
        inv_transform_domain: Optional[ImageType] = None,
    ):
        """
        :param status: Status code indicating registration success or failure.
        :param transform: The forward transform registration result, if any.
            The forward transform maps from fixed to moving space.
            The forward transform is the result of a single registration stage and may be
            added to a previous initial transform to map from fixed to moving space.
        :param transform_domain: Oriented representation of the domain over which the forward transform is valid.
            `transform_domain` has no voxel data and serves as a metadata representation of an
            oriented bounding box in physical space.
            `transform_domain` must be available if and only if `transform` is available.
        :param inv_transform: The inverse transform registration result, if any.
            The inverse transform maps from moving to fixed space.
            If `inv_transform` is available then `transform` must also be available.
        :param inv_transform_domain: Oriented representation of the domain over which the inverse transform is valid.
            `inv_transform_domain` has no voxel data and serves as a metadata representation of an
            oriented bounding box in physical space.
            `inv_transform_domain` must be available if and only if `inv_transform` is available.
        """
        if status == BlockRegStatus.SUCCESS and not transform:
            raise ValueError(
                f"Pairwise registration indicated success ({status})"
                f" but no forward transform was provided"
            )
        if transform and not transform_domain:
            raise ValueError(
                "Pairwise registration returned incomplete forward transform:"
                " failed to provide forward transform domain"
            )
        if transform_domain and itk.template(transform_domain)[0] != itk.Image:
            raise TypeError(
                f"Received invalid transform domain type: {type(transform_domain)}"
            )
        if (
            transform_domain
            and np.product(transform_domain.GetLargestPossibleRegion().GetSize()) == 0
        ):
            raise ValueError("Received invalid transform domain with size 0")
        if inv_transform and not inv_transform_domain:
            raise ValueError(
                "Pairwise registration returned incomplete inverse transform:"
                " failed to provide inverse transform domain"
            )
        if inv_transform_domain and itk.template(inv_transform_domain)[0] != itk.Image:
            raise TypeError(
                f"Received invalid transform domain type: {type(transform_domain)}"
            )
        if (
            inv_transform_domain
            and np.product(inv_transform_domain.GetLargestPossibleRegion().GetSize())
            == 0
        ):
            raise ValueError("Received invalid transform domain with size 0")
        self.status = BlockRegStatus(status)
        self.transform = transform
        self.transform_domain = transform_domain
        self.inv_transform = inv_transform
        self.inv_transform_domain = inv_transform_domain

    def __repr__(self) -> str:
        s = f"status:{self.status}"
        s += f",transform:{type(self.transform)},domain:{type(self.transform_domain)}"
        s += f",inverse transform:{type(self.inv_transform)},domain:{type(self.inv_transform_domain)}"
        return s


@dataclass
class LocatedBlockResult:
    """
    Encapsules pairwise registration result information with contextual
    information describing how the fixed subimage in registration
    relates to the greater fixed image.
    """

    result: BlockPairRegistrationResult
    """
    The result of pairwise subimage registration.
    May include extended information for specific implementations.
    """

    fixed_info: BlockInfo
    """
    Oriented representation of the fixed image block over which
    pairwise registration was performed to produce the encapsulated
    registration result information.
    """


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
    for block pair registration over the corresponding input fixed chunk.
    
    `status` has the same shape as the fixed input array of chunks.
    That is, if the fixed input array is subdivided into 2 chunks x 3 chunks x 4 chunks,
    `status` will be an array of voxels with shape [2,3,4].
    """
