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
        

"""
my_moving_image = ...
my_fixed_image = ...

my_initial_transform = ...

my_transform = registration_method(my_fixed_image, my_moving_image, my_initial_transform)

# registration method returns an update to the initial transform

final_transform = itk.CompositeTransform()
final_transform.append_transform(my_initial_transform)
final_transform.append_transform(my_transform)

my_warped_image = itk.resample_image_filter(
    my_moving_image,
    transform=final_transform,
    interpolator=interpolator,
    use_reference_image=True,
    reference_image=my_fixed_image
)

"""
