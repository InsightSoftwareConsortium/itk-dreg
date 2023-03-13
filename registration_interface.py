import itk
from abc import ABC, abstractmethod

# The function signature each registration approach should provide

FloatImage2DType = itk.Image[itk.F, 2]
FloatImage3DType = itk.Image[itk.F, 3]
FloatImageType = Union[Image2DType, Image3DType]

class RegistrationMethod(ABC):
    @abstractmethod
    def __call__(
          self,
          fixed_image: FloatImageType,
          moving_image: FloatImageType,
          initial_transform: itk.Transform[itk.D, 3],
          **kwargs
        ) -> itk.Transform[itk.D, 3]:
        pass
