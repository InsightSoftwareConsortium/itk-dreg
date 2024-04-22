import itk

import pickle

import itk_dreg.elastix.serialize

itk.auto_progress(2)

def validate_parameter_maps(m1, m2):
    assert all([k in m2.keys() for k in m1.keys()])
    assert all([k in m1.keys() for k in m2.keys()])
    for key in m1.keys():
        assert m1[key] == m2[key]

def validate_parameter_objects(p1, p2):
    assert p1.GetNumberOfParameterMaps() == p2.GetNumberOfParameterMaps()
    for map_index in range(p1.GetNumberOfParameterMaps()):
        validate_parameter_maps(p1.GetParameterMap(map_index), p2.GetParameterMap(map_index))

def test_serialize_elx_parameter_object():
    DEFAULT_PARAMETER_MAPS = ['rigid']
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(
        itk.ParameterObject.GetDefaultParameterMap(DEFAULT_PARAMETER_MAPS[0])
    )

    parameter_list = itk_dreg.elastix.serialize.parameter_object_to_list(parameter_object)
    pickled_parameter_list = pickle.dumps(parameter_list)
    unpickled_parameter_list = pickle.loads(pickled_parameter_list)
    unpickled_parameter_object = itk_dreg.elastix.serialize.list_to_parameter_object(unpickled_parameter_list)
    validate_parameter_objects(unpickled_parameter_object, parameter_object)

    # TODO: https://github.com/InsightSoftwareConsortium/ITKElastix/issues/257
    # pickled_parameter_object = pickle.dumps(parameter_object)
    # unpickled_parameter_object = pickle.loads(pickled_parameter_object)
    # validate_parameter_objects(unpickled_parameter_object, parameter_object)

    # dask_deserialized_object = dask.distributed.protocol.deserialize(
    #     dask.distributed.protocol.serialize(parameter_object)
    # )
    # validate_parameter_objects(dask_deserialized_object, unpickled_parameter_object)

    # Validate baseline is unchanged
    assert parameter_object.GetNumberOfParameterMaps() == 1
    validate_parameter_maps(parameter_object.GetParameterMap(0),
                            itk.ParameterObject.GetDefaultParameterMap(DEFAULT_PARAMETER_MAPS[0]))

    
