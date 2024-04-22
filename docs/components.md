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

