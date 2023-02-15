import itk
from abc import ABC, abstractmethod

# The function signature each registration approach should provide

class RegistrationMethod(ABC):
    @abstractmethod
    def __call__(
          self,
          fixed_image: itk.Image[itk.F, 3],
          moving_image: itk.Image[itk.F, 3],
          initial_transform: itk.Transform[itk.D, 3],
          **kwargs
        ) -> itk.Transform[itk.D, 3]:
        pass
