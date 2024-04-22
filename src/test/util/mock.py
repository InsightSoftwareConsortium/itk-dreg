from typing import Iterable, Iterator

import itk

from itk_dreg.base.image_block_interface import (
    BlockPairRegistrationResult,
    RegistrationTransformResult,
    BlockRegStatus,
    LocatedBlockResult,
)
from itk_dreg.base.registration_interface import (
    BlockPairRegistrationMethod,
    ReduceResultsMethod,
)

# TODO use unittest.mock


class ConstantBlockPairRegistrationMethod(BlockPairRegistrationMethod):
    """
    Return a constant, default registration result for each block.
    """

    def __init__(self, default_result: BlockPairRegistrationResult = None):
        default_transform_domain = itk.Image[itk.F, 3].New()
        default_transform_domain.SetRegions([10] * 3)
        self.default_result = default_result or BlockPairRegistrationResult(
            transform=itk.TranslationTransform[itk.D, 3].New(),
            transform_domain=default_transform_domain,
            inv_transform=None,
            inv_transform_domain=None,
            status=BlockRegStatus.SUCCESS,
        )

    def __call__(self, **kwargs):
        return self.default_result


class ConstantReduceResultsMethod(ReduceResultsMethod):
    """
    Return a constant, default transform result.
    """

    def __init__(self, default_result: RegistrationTransformResult = None):
        self.default_result = default_result or RegistrationTransformResult(
            transform=itk.TranslationTransform[itk.D, 3].New(), inv_transform=None
        )

    def __call__(self, **kwargs):
        return self.default_result


class CountingBlockPairRegistrationMethod(ConstantBlockPairRegistrationMethod):
    num_calls = 0

    def __call__(self, **kwargs) -> BlockPairRegistrationResult:
        self.num_calls += 1
        return super().__call__(**kwargs)


class CountingReduceResultsMethod(ConstantReduceResultsMethod):
    num_calls = 0

    def __call__(self, **kwargs) -> RegistrationTransformResult:
        self.num_calls += 1
        return super().__call__(**kwargs)


class IteratorBlockPairRegistrationMethod(BlockPairRegistrationMethod):
    def __init__(self, default_results: Iterator[BlockPairRegistrationResult]):
        self.default_results = default_results

    def __call__(self, **kwargs):
        return next(self.default_results)


class PassthroughReduceResultsMethod(ReduceResultsMethod):
    """
    Always return the Nth transform results.

    May fail if transform inputs are inherently bounded, such as a displacement field.
    """

    def __init__(self, return_index: int = 0):
        self.return_index = 0

    def __call__(self, block_results: Iterable[LocatedBlockResult], **kwargs):
        counter = 0
        results_iter = iter(block_results)
        while counter < self.return_index:
            next(results_iter)
        nth_block_result = next(results_iter)
        return RegistrationTransformResult(
            transform=nth_block_result.result.transform,
            inv_transform=nth_block_result.result.inv_transform,
        )
