#!/usr/bin/env python3

from typing import Union

import itk
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

# from registration_methods import register_elastix


def itk_matrix_transform_to_matrix(
    t: Union[itk.Euler3DTransform[itk.D], itk.AffineTransform[itk.D, 3]]
) -> npt.ArrayLike:
    """Get a 4x4 affine matrix from an ITK matrix transform"""
    output_arr = np.eye(4)
    output_arr[:3, :3] = np.array(t.GetMatrix())
    output_arr[:3, 3] = np.array(t.GetTranslation())
    return output_arr


def to_itk_euler_transform(mat: npt.ArrayLike) -> itk.Euler3DTransform[itk.D]:
    """
    Convert from a NumPy affine matrix to `itk.Euler3DTransform` representation.
    :param mat: The input 4x4 affine matrix
    :return: The corresponding `itk.Euler3DTransform`.
    """
    transform = itk.Euler3DTransform[itk.D].New()
    transform.SetMatrix(np_to_itk_matrix(mat[:3, :3]))
    transform.Translate(mat[:3, 3])
    return transform


def np_to_itk_matrix(arr: npt.ArrayLike) -> itk.Matrix[itk.D, 3, 3]:
    """Convert a 3x3 matrix from numpy to ITK format"""
    vnl_matrix = itk.Matrix[itk.D, 3, 3]().GetVnlMatrix()
    for i in range(3):
        for j in range(3):
            vnl_matrix.set(i, j, arr[i, j])
    return itk.Matrix[itk.D, 3, 3](vnl_matrix)


def estimate_euler_transform_consensus(transforms: npt.ArrayLike) -> npt.ArrayLike:
    """Estimate a mean representation of a list of transform results"""
    if transforms.ndim != 3 or transforms.shape[1] != 4 or transforms.shape[2] != 4:
        raise ValueError(
            f"Expected list of 4x4 euler transforms but received array with shape {transforms.shape}"
        )
    average_transform = np.eye(4)
    average_transform[:3, :3] = average_rotation(transforms[:, :3, :3])
    average_transform[:3, 3] = average_translation(transforms[:, :3, 3])
    return average_transform


def average_rotation(rotations: npt.ArrayLike) -> npt.ArrayLike:
    """Compute average rotation by way of linear quaternion averaging"""
    if rotations.ndim != 3 or rotations.shape[1] != 3 or rotations.shape[2] != 3:
        raise ValueError(
            "Expected list of 3x3 rotation matrices but received array with shape {rotations.shape}"
        )

    accum_quat = Rotation.from_matrix(np.eye(3)).as_quat()
    for index in range(rotations.shape[0]):
        rotation = rotations[index, :, :]
        if not np.all(np.isclose(np.matmul(rotation, rotation.T), np.eye(3))):
            raise ValueError(f"Matrix {index} is not a rigid rotation matrix")
        rot = Rotation.from_matrix(rotation)
        accum_quat += rot.as_quat()

    accum_quat /= np.linalg.norm(accum_quat)
    return Rotation.from_quat(accum_quat).as_matrix()


def average_translation(translations: npt.ArrayLike) -> npt.ArrayLike:
    """Compute linear average of translation vectors"""
    assert translations.ndim == 2
    assert translations.shape[1] == 3
    return np.mean(translations, axis=0)
