## Usage

`itk_dreg` provides a framework to register a moving image onto a fixed image.
The output of a single run is an `itk.Transform` object that can be used
to resample the moving image onto the fixed image. Multiple runs can be chained
to successively refine registration over multiple image resolutions and over
various registration and reduction methods.

Use `itk_dreg.register.register_images` to assemble and run a task graph for distributed registration.


```py
my_initial_transform = ...

# registration method returns an update to the initial transform

my_registration_schedule = itk_dreg.register_images(
    # Methods
    fixed_reader_ctor=my_construct_streaming_reader_method,
    moving_reader_ctor=my_construct_streaming_reader_method,
    block_registration_method=my_block_pair_registration_method_subclass,
    reduce_method=my_postprocess_registration_method_subclass,
    # Data
    fixed_chunk_size=(x,y,z),
    initial_transform=my_initial_transform,
    overlap_factors=[0.1,0.1,0.1]
)
my_result = my_registration_schedule.registration_result.compute()

final_transform = itk.CompositeTransform()
final_transform.append_transform(my_initial_transform)
final_transform.append_transform(my_result.transforms.transform)

# we can use the result transform to resample the moving image to fixed image space

interpolator = itk.LinearInterpolateImageFunction.New(my_moving_image)

my_warped_image = itk.resample_image_filter(
    my_moving_image,
    transform=final_transform,
    interpolator=interpolator,
    use_reference_image=True,
    reference_image=my_fixed_image
)

```
