
import itk
itk.auto_progress(2)

import pickle
import dask.distributed.protocol

import itk_dreg.base.image_block_interface

def test_serialize_pairwise_result():
    failure_result = itk_dreg.base.image_block_interface.BlockPairRegistrationResult(
        status=itk_dreg.base.image_block_interface.BlockRegStatus.FAILURE
    )
    deserialized_result = pickle.loads(pickle.dumps(failure_result))
    assert deserialized_result.status == failure_result.status
    deserialized_result = dask.distributed.protocol.deserialize(
        *dask.distributed.protocol.serialize(failure_result)
    )
    assert deserialized_result.status == failure_result.status

    transform_domain = itk.Image[itk.F,3].New()
    transform_domain.SetRegions([10]*3)
    success_result = itk_dreg.base.image_block_interface.BlockPairRegistrationResult(
        status=itk_dreg.base.image_block_interface.BlockRegStatus.SUCCESS,
        transform=itk.TranslationTransform[itk.D,3].New(),
        transform_domain=transform_domain
    )
    # TODO Unbuffered `itk.Image` is not yet pickleable (ITK v5.4rc2)
    # ValueError: PyMemoryView_FromBuffer(): info->buf must not be NULL
    # https://github.com/InsightSoftwareConsortium/ITK/issues/4267
    deserialized_result = pickle.loads(pickle.dumps(success_result))
    assert deserialized_result.status == success_result.status
    # TODO validate transforms, transform domains
    deserialized_result = dask.distributed.protocol.deserialize(
        *dask.distributed.protocol.serialize(success_result)
    )
    assert deserialized_result.status == success_result.status
    # TODO validate transforms, transform domains
