#!/usr/bin/env python3

import logging
from typing import List, Tuple, Dict, Any

import itk


logger = logging.getLogger(__name__)

ParameterObjectType = itk.ParameterObject
ParameterMapType = Any  # FIXME itk.elxParameterObjectPython.mapstringvectorstring
SerializableParameterMapType = Dict[str, Tuple[str, str]]
SerializableParameterObjectType = List[SerializableParameterMapType]


def parameter_map_to_dict(
    parameter_map: ParameterMapType,
) -> SerializableParameterMapType:
    """
    Convert an ITKElastix parameter map to a pickleable dictionary.

    :param parameter_map: The `itk.elxParameterObjectPython.mapstringvectorstring` to convert.
    :return: A dictionary of parameter map values.
    """
    return {k: v for (k, v) in parameter_map.items()}


def dict_to_parameter_map(val: SerializableParameterMapType) -> ParameterMapType:
    """
    Create an ITKElastix parameter map from a dictionary representation.

    :param val: A dictionary of parameter map values to convert.
    :return: An `itk.elxParameterObjectPython.mapstringvectorstring` parameter map representation.
    """
    # Eagerly load ITKElastix definitions so that mapstringvectorstring is available
    # TODO investigate a cleaner approach for this
    _ = itk.ParameterObject

    parameter_map = itk.elxParameterObjectPython.mapstringvectorstring()
    for k, v in val.items():
        parameter_map[k] = v
        if not parameter_map[k] == v:
            raise ValueError(f"Failed to set parameter map value: {k}, {v}")
    return parameter_map


def parameter_object_to_list(
    parameter_object: ParameterObjectType,
) -> SerializableParameterObjectType:
    """
    Convert an ITKElastix parameter object to a pickleable collection.

    :param parameter_object: An `itk.ParameterObject` representing an Elastix registration configuration.
        A parameter object may contain 0 to many parameter maps.
    :return: A list of parameter maps.
    """
    result = []
    for map_index in range(parameter_object.GetNumberOfParameterMaps()):
        result.append(
            parameter_map_to_dict(parameter_object.GetParameterMap(map_index))
        )
    return result


def list_to_parameter_object(
    elastix_parameter_map_vals: SerializableParameterObjectType,
) -> ParameterObjectType:
    """
    Create an ITKElastix parameter object from a list-of-dictionaries representation.

    :param elastix_parameter_map_vals: The list of dictionaries that represent
        ITKElastix parameter maps.
    :return: An `itk.ParameterObject` populated with parameter maps.
    """
    parameter_object = itk.ParameterObject.New()
    for elastix_parameter_map_params in elastix_parameter_map_vals:
        parameter_object.AddParameterMap(
            dict_to_parameter_map(elastix_parameter_map_params)
        )
    return parameter_object


"""
TODO: Contribute back to ITKElastix

At the time of writing (2023.10.22), ITKElastix parameter objects and parameter maps
are not serializable (pickleable) by default. This block can be used to monkeypatch the appropriate
classes to be serializable in dask initialization as a short-term fix.


def get_mapstringvectorstring_state(self) -> SerializableParameterMapType:
    return parameter_map_to_dict(self)


def set_mapstringvectorstring_state(self, new_state: SerializableParameterMapType):
    other = dict_to_parameter_map(new_state)
    self.clear()
    self.swap(other)


def get_itkparameterobject_state(self) -> SerializableParameterMapType:
    return parameter_object_to_list(self)


def set_itkparameterobject_state(self, new_state: SerializableParameterMapType):
    other = list_to_parameter_object(new_state)
    try:
        other_pm = other.GetParameterMaps()
        self.SetParameterMaps(other_pm)
    except Exception as e:
        dask.distributed.print(f"itk_dreg.elastix error: {e}")


setattr(itk.elxParameterObjectPython.mapstringvectorstring, '__getstate__', get_mapstringvectorstring_state)
setattr(itk.elxParameterObjectPython.mapstringvectorstring, '__setstate__,', set_mapstringvectorstring_state)
setattr(itk.ParameterObject, '__getstate__', get_itkparameterobject_state)
setattr(itk.ParameterObject, '__setstate__', set_itkparameterobject_state)

"""
