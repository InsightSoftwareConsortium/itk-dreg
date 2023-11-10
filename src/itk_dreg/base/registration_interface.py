#!/usr/bin/env python3

import itk
from abc import ABC, abstractmethod
from typing import Iterable

from .image_block_interface import (
    BlockPairRegistrationResult,
    LocatedBlockResult,
    RegistrationTransformResult,
    BlockInfo,
)
from .itk_typing import ImageType, ImageReaderType, TransformType

"""
Defines extensible components to extend with concrete implementations.
"""


class ConstructReaderMethod(ABC):
    """
    A method that generates a new `itk.ImageFileReader` for image registration.

    ITK provides the `itk.ImageFileReader` mechanism to retrieve all or part of
    a spatial image from a provided local or remote image source. `itk-dreg`
    registration infrastructure attempts to stream image subregions into memory
    at runtime in order to perform block-based pairwise registration without
    ever loading an entire image into memory at once.

    Extend this class to customize the image reading step for cases such as
    to attach domain-specific metadata or to convert from a reference type
    not supported by ITK by default.

    The resulting `itk.ImageFileReader` object MUST be initialized with metadata
    to represent the extent of the underlying data in voxel and physical space.

    It is strongly recommended that the resulting `itk.ImageFileReader` is NOT
    initialized with underlying voxel data. Voxel regions should be lazily
    initialized by `itk-dreg` registration infrastructure to match block
    requested regions.

    .. code-block:: python

        ReaderSource = ConstructReaderMethodSubclass(...)
        image_source = ReaderSource()
        image = image_source.UpdateLargestPossibleRegion()
    """

    @abstractmethod
    def __call__(self, **kwargs) -> ImageReaderType:
        pass


class BlockPairRegistrationMethod(ABC):
    """
    A method that registers two spatially located image blocks together.

    `fixed_subimage` and `moving_image` inputs are `itk.Image` representations
    of block subregions within greater fixed and moving inputs.

    Extend this class to implement custom registration method that plugs in
    to `itk-dreg` registration infrastructure.
    """

    @abstractmethod
    def __call__(
        self,
        fixed_subimage: ImageType,
        moving_subimage: ImageType,
        initial_transform: TransformType,
        block_info: BlockInfo,
        **kwargs,
    ) -> BlockPairRegistrationResult:
        """
        Run image-to-image pairwise registration.

        :param fixed_subimage: The reference fixed subimage.
            `fixed_subimage.RequestedRegion` reflects the requested subregion corresponding
            to the scheduled dask array chunk.
            The initial `fixed_subimage.BufferedRegion` includes the requested region
            and possibly an extra padding factor introduced before fetching fixed
            image voxel data.
        :param moving_subimage: The moving subimage to be registered onto fixed image space.
            `moving_subimage.RequestedRegion` reflects the requested subregion
            corresponding to the approximate physical bounds of `fixed_subimage.RequestedRegion`
            after initialization with `initial_transform`.
            The initial `moving_subimage.BufferedRegion` includes the requested region
            and possibly an extra padding factor introduced before fetching fixed
            image voxel data.
        :param initial_transform: The forward transform representing an initial alignment
            mapping from fixed to moving image space.
        :returns: The result of block pairwise registration, including a status code indicating
            whether registration succeeded, a forward transform to run after the initial transform,
            the domain over which the forward transform is considered valid, and an optional
            inverse transform. May be extended with additional implementation-specific information.
        """
        pass


class ReduceResultsMethod(ABC):
    """
    A method that reduces a sparse collection of pairwise block registration results
    to yield a generalized fixed-to-moving transform.

    Extend this class to implement a custom method mapping block results to a general transform.

    Possible implementations could include methods for finding global consensus among results,
    combination methods to yield a piecewise transform, or patchwise methods to normalize among
    bounded transform domains.
    """

    @abstractmethod
    def __call__(
        self,
        block_results: Iterable[LocatedBlockResult],
        fixed_reader_ctor: ConstructReaderMethod,
        initial_transform: itk.Transform,
        **kwargs,
    ) -> RegistrationTransformResult:
        """
        :param block_results: An iterable collection of subimages in fixed space
            and the corresponding registration result for the given subimage.
            Subimages are not buffered and represent the subdomains within the
            original fixed image space prior to initial transform application.
        :param fixed_reader_ctor: Method to create an image reader to stream
            part or all of the fixed image.
        :param initial_transform: Initial forward transform used in registration.
            The forward transform maps from the fixed to moving image.
        """
        pass


"""
my_fixed_image = ...
my_moving_image = ...

my_initial_transform = ...

# registration method returns an update to the initial transform

my_transform = itk_dreg.register_images(
    target_da=target_image_dask_voxel_array,
    initial_transform=my_initial_transform,
    source_reader_ctor=my_construct_streaming_reader_method,
    target_reader_ctor=my_construct_streaming_reader_method,
    block_registration_method=my_block_pair_registration_method_subclass,
    postprocess_method=my_postprocess_registration_method_subclass,
    overlap_factors=[0.1,0.1,0.1]
)

final_transform = itk.CompositeTransform()
final_transform.append_transform(my_initial_transform)
final_transform.append_transform(my_transform)

# we can use the result transform to resample the moving image to fixed image space

interpolator = itk.LinearInterpolateImageFunction.New(my_moving_image)

my_warped_image = itk.resample_image_filter(
    my_moving_image,
    transform=final_transform,
    interpolator=interpolator,
    use_reference_image=True,
    reference_image=my_fixed_image
)

"""
