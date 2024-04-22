## Getting Started

Install `itk-dreg` from the Python Package Index (PyPI):

```sh
(venv) > python -m pip install itk-dreg
```

For developers, clone the Git repository and install with `flit`.

```sh
(venv) > python -m pip install flit
(venv) > git clone https://www.github.com/InsightSoftwareConsortium/itk-dreg.git
(venv) > itk-dreg/src > python -m pip install ./src
```

Several Jupyter Notebook examples are available for getting started. To run locally:

```sh
itk-dreg > python -m pip install ./src[notebook]
itk-dreg > jupyter notebook
```
