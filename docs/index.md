# itk-dreg

> A framework for distributed, large-scale image registration.

[![itk-dreg version](https://badge.fury.io/py/itk-dreg.svg)](https://pypi.org/project/itk-dreg/)
[![build-test-publish](https://github.com/InsightSoftwareConsortium/itk-dreg/actions/workflows/build-test-publish.yml/badge.svg)](https://github.com/InsightSoftwareConsortium/itk-dreg/actions/workflows/build-test-publish.yml)

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

## Installation

```shell
pip install itk-dreg
```

```{toctree}
:hidden:
:maxdepth: 3
:caption: 👋 Introduction

getting_started.md
usage.md
components.md
```

```{toctree}
:hidden:
:maxdepth: 3
:caption: 📖 Reference

apidocs/index.rst
```

```{toctree}
:hidden:
:maxdepth: 3
:caption: 🔨 itk-dreg Development

contributing.md
develop.md
acknowledgements.md
```

## License

`itk-dreg` is distributed under the [Apache-2.0](LICENSE) permissive license.

## Questions and Queries

`itk-dreg` is part of the Insight Toolkit tools ecosystem for medical image processing. We encourage developers to
reach out to the ITK community with questions on the [ITK Discourse forums](https://discourse.itk.org/). Those
interested in custom or commercial development should reach out to [Kitware](https://www.kitware.com/contact/) to learn more.
