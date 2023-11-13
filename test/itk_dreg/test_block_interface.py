
import numpy as np
import itk
itk.auto_progress(2)

import pytest

from itk_dreg.base.image_block_interface import BlockRegStatus, BlockPairRegistrationResult

def test_construct_failed_pairwise_result():
    # Verify no inputs required for failure
    result = BlockPairRegistrationResult(status=BlockRegStatus.FAILURE)
    assert result.status == BlockRegStatus.FAILURE
    assert result.transform == None
    assert result.transform_domain == None
    assert result.inv_transform == None
    assert result.inv_transform_domain == None

def test_construct_forward_pairwise_result():
    valid_transform = itk.TranslationTransform[itk.D,3].New()
    valid_transform_domain = itk.Image[itk.UC,3].New()
    valid_transform_domain.SetRegions([1,1,1])
    result = BlockPairRegistrationResult(
        status=BlockRegStatus.SUCCESS,
        transform=valid_transform,
        transform_domain=valid_transform_domain
    )
    assert result.status == BlockRegStatus.SUCCESS
    assert result.transform == valid_transform
    assert result.transform_domain == valid_transform_domain
    assert result.inv_transform == None
    assert result.inv_transform_domain == None

    # Validate incomplete construction
    with pytest.raises(ValueError):
        BlockPairRegistrationResult(
            status=BlockRegStatus.SUCCESS,
            transform=valid_transform
            # transform_domain required
        )
        
    invalid_transform_domain = [1,2,3]
    with pytest.raises(KeyError):
        BlockPairRegistrationResult(
            status=BlockRegStatus.SUCCESS,
            transform=valid_transform,
            transform_domain=invalid_transform_domain
        )

    invalid_transform_domain = itk.Image[itk.UC,3].New()
    assert all([size == 0 for size in invalid_transform_domain.GetLargestPossibleRegion().GetSize()])
    with pytest.raises(ValueError):
        BlockPairRegistrationResult(
            status=BlockRegStatus.SUCCESS,
            transform=valid_transform,
            transform_domain=invalid_transform_domain
        )

    
def test_construct_inverse_pairwise_result():
    valid_transform = itk.TranslationTransform[itk.D,3].New()
    valid_transform_domain = itk.Image[itk.UC,3].New()
    valid_transform_domain.SetRegions([1,1,1])
    valid_inverse_transform = itk.TranslationTransform[itk.D,3].New()
    valid_inverse_transform_domain = itk.Image[itk.UC,3].New()
    valid_inverse_transform_domain.SetRegions([1,1,1])
    valid_inverse_transform_domain.SetOrigin([-1]*3)
    valid_inverse_transform_domain.SetSpacing([0.1]*3)
    valid_inverse_transform_domain.SetDirection(np.array([[0,-1,0],[-1,0,0],[0,0,1]]))
    result = BlockPairRegistrationResult(
        status=BlockRegStatus.SUCCESS,
        transform=valid_transform,
        transform_domain=valid_transform_domain,
        inv_transform=valid_inverse_transform,
        inv_transform_domain=valid_inverse_transform_domain
    )
    assert result.status == BlockRegStatus.SUCCESS
    assert result.transform == valid_transform
    assert result.transform_domain == valid_transform_domain
    assert result.inv_transform == valid_inverse_transform
    assert result.inv_transform_domain == valid_inverse_transform_domain
    
    # validate incomplete construction
    with pytest.raises(ValueError):
        BlockPairRegistrationResult(
            status=BlockRegStatus.SUCCESS,
            transform=valid_transform,
            transform_domain=valid_transform_domain,
            inv_transform=valid_transform,
            # inv_transform_domain required
        )
        
    invalid_transform_domain = [1,2,3]
    with pytest.raises(KeyError):
        BlockPairRegistrationResult(
            status=BlockRegStatus.SUCCESS,
            transform=valid_transform,
            transform_domain=valid_transform_domain,
            inv_transform=valid_transform,
            inv_transform_domain=invalid_transform_domain
        )

    invalid_transform_domain = itk.Image[itk.UC,3].New()
    assert all([size == 0 for size in invalid_transform_domain.GetLargestPossibleRegion().GetSize()])
    with pytest.raises(ValueError):
        BlockPairRegistrationResult(
            status=BlockRegStatus.SUCCESS,
            transform=valid_transform,
            transform_domain=valid_transform_domain,
            inv_transform=valid_transform,
            inv_transform_domain=invalid_transform_domain
        )



