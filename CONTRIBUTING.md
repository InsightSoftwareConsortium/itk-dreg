# Contributing to `itk-dreg`

Welcome to the Insight Toolkit Distributed Registration framework! We are excited that you are here! Join us as a contributing member of the community.

There are two ways to contribute as part of the ITK and `itk-dreg` community:
1. Contribute features or fixes to the `itk-dreg` project directly; or
2. Create a new community extension to plug in to the `itk-dreg` framework.

We recommend [creating a new community extension](#creating-a-community-extension) for most new features.

## Prerequisites

You will need the following to get started:
- A PC for development;
- [Git version control](https://git-scm.com/downloads);
- [Python](https://www.python.org/) 3.9 or later (we recommend 3.11);

We recommend that new developers follow the [ITK contributing guidelines](https://docs.itk.org/en/latest/contributing/index.html)
to get started with Git development.

## Setting up for Development

We recommend developing in a Python virtual environment. Refer to the following:
- If you are using Anaconda, refer to [Anaconda's environments documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#managing-environments).
- Otherwise, refer to [Python virtual environment documentation](https://docs.python.org/3/library/venv.html).

We currently use [`flit`](https://pypi.org/project/flit/) to manage `itk-dreg` dependencies and installation. Open a shell prompt and activate your virtual environment, then run the following commands to install dependencies from PyPI and create a symbolic link to your local `itk-dreg` changes:

On Linux:
```sh
itk-dreg > cd src
itk-dreg/src > python -m pip install --upgrade pip
itk-dreg/src > flit install --symlink --extras develop
```

On Windows:
```pwsh
itk-dreg > cd src
itk-dreg/src > python -m pip install --upgrade pip
itk-dreg/src > flit install --pth-file --extras develop
```

## Development and Chores

Follow the [ITK development workflow](https://docs.itk.org/en/latest/contributing/index.html#workflow) to guide your local development:

1. If you are contributing to `itk-dreg`, [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) under your GitHub user account.
2. [Clone the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository) or [add a new remote](https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories) for your user fork.
2. Navigate to the repository directory on your PC and create a new development branch with `git checkout -b <my-branch-name>`
3. Develop locally.
4. Commit changes ("make a checkpoint") with `git commit`. Use [standard prefixes](https://docs.itk.org/en/latest/contributing/index.html#commit-messages) in your commit message.
5. When your changes are ready, [create a pull request](https://github.com/InsightSoftwareConsortium/itk-dreg/pulls) on the `itk-dreg` repository.

Proposed changes must meet two principle criteria to be eligible for merge into `itk-dreg`:

1. The change must be reviewed and approved by an `itk-dreg` maintainer. Review is subjective and includes criteria such as:
  - Is the change reasonable?
  - Does the change belong in `itk-dreg` or in a community extension module?
  - Is the change reasonably tested to ensure correctness?
  - Does the change meet community standards?
  - Does the change meet code quality standards?
2. Continuous integration workflows for the changes must pass, including linting and automated testing.

Pull request integration is at the discretion of `itk-dreg` maintainers. The following are best practices to improve the likelihood
of a successful pull request integration:

1. Communicate your work early and often. We advise opening an issue or a draft pull request early in development to solicit
community discussion and guidance.
2. Write and run unit tests early and often to police code correctness. `itk-dreg` uses [`pytest`](https://docs.pytest.org/en/7.4.x/) for automated testing:
```py
itk-dreg > pytest # runs all tests
itk-dreg > pytest -k <your-test-name> -vvv -s # runs your test with verbose output
```
3. Use linting tools to improve code quality. `itk-dreg` uses `ruff` and `black` for linting:
```py
itk-dreg > python -m ruff check ./src
itk-dreg > python -m black ./src
```

Refer to [developer documentation](docs/develop.md) for additional development and debugging suggestions.

## Creating a Community Extension

`itk-dreg` provides three major components:

1. A _virtual interface_ specifying inputs and outputs for registration and reduction methods;
2. A scheduling apparatus that receives _implementations_ of the virtual interface and connects
them to create and run a distributed registration method;
3. Two _concrete implementations_ of the virtual interface:
  - A registration approach based on ITKElastix that receives two subimages and outputs a forward transform; and
  - A result reduction method that receives a set of registration results and outputs a `itk.DisplacementFieldTransform`
    that is valid over the whole input image.

Many existing approaches exist in literature for image-to-image registration. We can extend the `itk-dreg` framework
to swap out the `itk_dreg.elastix` and `itk_dreg.reduce_dfield` approaches (3) for alternate methods.

Each extension should be distributed in its own Python module that depends on `itk-dreg`. Follow these steps to set up
a new `itk-dreg` extension:

1. [Create a GitHub repository](https://docs.github.com/en/get-started/quickstart/create-a-repo) to hold the module.
2. [Create a Python project](https://packaging.python.org/en/latest/tutorials/packaging-projects/). We recommend
[`hatch`](https://hatch.pypa.io/latest/) or [`flit`](https://flit.pypa.io/en/stable/) build systems for getting started.
3. [Add a dependency](https://peps.python.org/pep-0631/) on the latest version of `itk-dreg` to your `pyproject.toml`.

You are now ready to write your extension. `itk-dreg` provides three registration interface for extension.
You may choose to extend any or all of these interfaces. Visit
[`registration_interface.py`](src/itk_dreg/base/registration_interface.py) for more information in
the docstring for each interface.

1. `ConstructReaderMethod`: A method to generate an `itk.ImageFileReader` to stream an image subregion for
a given registration task. Usually the default `itk_dreg` implementation is sufficient, but can be extended
in the event of domain-specific metadata parsing. For instance, the [Lightsheet Registration notebook](examples/LightsheetRegistration.ipynb)
provides an extended reader to parse nonstandard lightsheet orientation metadata before registration occurs.

2. `BlockPairRegistrationMethod`: A method to register a fixed and moving subimage together. `itk-dreg` implements
the `itk_dreg.elastix` submodule to perform pairwise subimage registration with ITKElastix. Returns a result
including at least a status code and a forward transform result.

3. `ReduceResultsMethod`: A method to receive a collection of subimage domains with their corresponding forward transform
results and return a single `itk.Transform` forward transform mapping from the fixed to moving image domain.
`itk-dreg` implements the `itk_dreg.reduce_dfield` submodule to sample piecewise transform results into a
single output deformation field.

Once you've written your class or classes you may use them with the `itk-dreg` registration framework directly:

```py
import itk_dreg.register
import my_dreg_extension

... # set up methods, data

registration_schedule = itk_dreg.register.register_images(
    fixed_reader_ctor=my_dreg_extension.my_construct_streaming_reader_method,
    moving_reader_ctor=my_dreg_extension.my_construct_streaming_reader_method,
    block_registration_method=my_dreg_extension.my_block_pair_registration_method_subclass,
    reduce_method=my_dreg_extension.my_postprocess_registration_method_subclass,
    fixed_chunk_size=(x,y,z),
    initial_transform=my_initial_transform,
    overlap_factors=[a,b,c]
)

result = registration_schedule.registration_result.compute()

... # Resample, save the result, etc
```

We suggest using `pytest`, `ruff`, and `black` for testing and linting during development. We also suggest adding
example scripts and/or Jupyter Notebooks in the `examples/` repository of your project to aid in user understanding
and adoption.

## Additional Information

`itk-dreg` is part of the Insight Toolkit tools ecosystem for medical image processing. We encourage developers to
reach out to the ITK community with questions on the [ITK Discourse forums](https://discourse.itk.org/). Those
interested in custom or commercial development should reach out to [Kitware](https://www.kitware.com/contact/) to learn more.

Happy coding!
