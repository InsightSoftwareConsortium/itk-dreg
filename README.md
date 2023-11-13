# itk-dreg

A framework for distributed, large-scale image registration.

## Overview

The ITK Distributed Registration module (`itk-dreg`) provides a framework based on the
Insight Toolkit (ITK) and the `dask.distributed` library for the purpose of registering
large-scale images out of memory.

Traditional image registration techniques in ITK and other libraries (Elastix, ANTs) require
in-memory processing, meaning those techniques load images fully in memory (RAM) during
registration execution. Meanwhile, large image datasets can occupy terabytes of data for a single
image on the cloud and far exceed available memory. `itk-dreg` addresses this issue as a
map-reduce problem where images are successivly subdivided into subimages, registered,
and then composed into a descriptive output. Multiple `itk-dreg` registration graphs may be
executed in succession to yield a pipeline for multiresolution image registration.

`itk-dreg` provides three major components:
- Concepts to describe the map-reduce registration problem;
- A user-ready registration method to produce `itk.DisplacementFieldTransform`s from out-of-memory
    registration with Dask scheduling and ITKElastix registration;
- A developer framework to extend `itk-dreg` with novel registration and reduction methods.

## Getting Started

To use `itk-dreg`, clone the Git repository and install with `flit`.

```py
> python -m pip install flit
> git clone https://www.github.com/InsightSoftwareConsortium/itk-dreg.git
> cd itk-dreg/src
itk-dreg/src > python -m flit install
```

Several Jupyter Notebook examples are available for getting started. To run locally:

```py
itk-dreg/src > python -m flit install --extras
itk-dreg/src > cd ../examples
itk-dreg/examples > jupyter notebook
```

## Use Instructions

`itk_dreg` provides a framework to register a moving image onto a fixed image.
The output of a single run is an `itk.Transform` object that can be used
to resample the moving image onto the fixed image. Multiple runs can be chained
to successively refine registration over multiple image resolutions and over
various registration and reduction methods.

Use `itk_dreg.register.register_images` to assemble and run a task graph for distributed registration.


```py
my_initial_transform = ...

# registration method returns an update to the initial transform

my_registration_schedule = itk_dreg.register_images(
    fixed_chunk_size=(x,y,z),
    initial_transform=my_initial_transform,
    fixed_reader_ctor=my_construct_streaming_reader_method,
    moving_reader_ctor=my_construct_streaming_reader_method,
    block_registration_method=my_block_pair_registration_method_subclass,
    reduce_method=my_postprocess_registration_method_subclass,
    overlap_factors=[0.1,0.1,0.1]
)
my_result = my_registration_schedule.registration_result.compute()

final_transform = itk.CompositeTransform()
final_transform.append_transform(my_initial_transform)
final_transform.append_transform(my_result.transforms.transform)

# we can use the result transform to resample the moving image to fixed image space

interpolator = itk.LinearInterpolateImageFunction.New(my_moving_image)

my_warped_image = itk.resample_image_filter(
    my_moving_image,
    transform=final_transform,
    interpolator=interpolator,
    use_reference_image=True,
    reference_image=my_fixed_image
)

```

## Components

### Core Components

`itk_dreg` provides the following core components:

- `itk_dreg.register` defines scheduling infrastructure and the main entry point into the
    `itk_dreg` registration framework.
- `itk_dreg.base` defines common types and virtual interfaces for the `itk_dreg` registration framework.
    Virtual interfaces in `itk_dreg.base.registration_interface` serve as an entry point for
    contributors to write their own registration and reduction methods.
- `itk_dreg.block` defines common methods to map between voxel and spatial subdomains.

These components must be installed to use the `itk_dreg` registration framework.

### Extended Components

`itk_dreg` includes a few common implementations to get started with image registration.
These components act as extensions and are not necessarily required for running `itk_dreg`.

- `itk_dreg.itk` provides ITK-based methods to aid in image streaming and dask chunk scheduling.
- `itk_dreg.elastix` adapts the ITKElastix registration routines for distributed
    registration in `itk_dreg`.
- `itk_dreg.reduce_dfield` implements a transform-reduction method to estimate a single
    `itk.DeformationFieldTransform` from block registration results in `itk_dreg`.
- `itk_dreg.mock` provides mock implementations of common framework components for use in
    testing and debugging.

Alternate registration and transform reduction modules may be available in the future
either as part of `itk_dreg` or via community distributions.

## Contributing

Refer to [Contributing documentation](CONTRIBUTING.md) for getting started with `itk-dreg` development.
Please direct feature requests or bug reports to the `itk-dreg` [GitHub Issues](https://github.com/InsightSoftwareConsortium/itk-dreg/issues)
board.

## License

`itk-dreg` is distributed under the [Apache-2.0](LICENSE) permissive license.

## Questions and Queries

`itk-dreg` is part of the Insight Toolkit tools ecosystem for medical image processing. We encourage developers to
reach out to the ITK community with questions on the [ITK Discourse forums](https://discourse.itk.org/). Those
interested in custom or commercial development should reach out to [Kitware](https://www.kitware.com/contact/) to learn more.

## Acknowledgements

`itk-dreg` was developed in part by with support from:

- [NIH NIMH BRAIN Initiative](https://braininitiative.nih.gov/) under award 1RF1MH126732.
