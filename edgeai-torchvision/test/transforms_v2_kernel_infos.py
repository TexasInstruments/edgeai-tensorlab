import decimal
import functools
import itertools
import math

import numpy as np
import PIL.Image
import pytest
import torch.testing
import torchvision.ops
import torchvision.transforms.v2.functional as F
from common_utils import (
    ArgsKwargs,
    combinations_grid,
    get_num_channels,
    ImageLoader,
    InfoBase,
    make_bounding_box_loader,
    make_bounding_box_loaders,
    make_detection_mask_loader,
    make_image_loader,
    make_image_loaders,
    make_image_loaders_for_interpolation,
    make_mask_loaders,
    make_video_loader,
    make_video_loaders,
    mark_framework_limitation,
    TestMark,
)
from torch.utils._pytree import tree_map
from torchvision import datapoints
from torchvision.transforms._functional_tensor import _max_value as get_max_value, _parse_pad_padding

__all__ = ["KernelInfo", "KERNEL_INFOS"]


class KernelInfo(InfoBase):
    def __init__(
        self,
        kernel,
        *,
        # Defaults to `kernel.__name__`. Should be set if the function is exposed under a different name
        # TODO: This can probably be removed after roll-out since we shouldn't have any aliasing then
        kernel_name=None,
        # Most common tests use these inputs to check the kernel. As such it should cover all valid code paths, but
        # should not include extensive parameter combinations to keep to overall test count moderate.
        sample_inputs_fn,
        # This function should mirror the kernel. It should have the same signature as the `kernel` and as such also
        # take tensors as inputs. Any conversion into another object type, e.g. PIL images or numpy arrays, should
        # happen inside the function. It should return a tensor or to be more precise an object that can be compared to
        # a tensor by `assert_close`. If omitted, no reference test will be performed.
        reference_fn=None,
        # These inputs are only used for the reference tests and thus can be comprehensive with regard to the parameter
        # values to be tested. If not specified, `sample_inputs_fn` will be used.
        reference_inputs_fn=None,
        # If true-ish, triggers a test that checks the kernel for consistency between uint8 and float32 inputs with the
        # reference inputs. This is usually used whenever we use a PIL kernel as reference.
        # Can be a callable in which case it will be called with `other_args, kwargs`. It should return the same
        # structure, but with adapted parameters. This is useful in case a parameter value is closely tied to the input
        # dtype.
        float32_vs_uint8=False,
        # Some kernels don't have dispatchers that would handle logging the usage. Thus, the kernel has to do it
        # manually. If set, triggers a test that makes sure this happens.
        logs_usage=False,
        # See InfoBase
        test_marks=None,
        # See InfoBase
        closeness_kwargs=None,
    ):
        super().__init__(id=kernel_name or kernel.__name__, test_marks=test_marks, closeness_kwargs=closeness_kwargs)
        self.kernel = kernel
        self.sample_inputs_fn = sample_inputs_fn
        self.reference_fn = reference_fn
        self.reference_inputs_fn = reference_inputs_fn

        if float32_vs_uint8 and not callable(float32_vs_uint8):
            float32_vs_uint8 = lambda other_args, kwargs: (other_args, kwargs)  # noqa: E731
        self.float32_vs_uint8 = float32_vs_uint8
        self.logs_usage = logs_usage


def _pixel_difference_closeness_kwargs(uint8_atol, *, dtype=torch.uint8, mae=False):
    return dict(atol=uint8_atol / 255 * get_max_value(dtype), rtol=0, mae=mae)


def cuda_vs_cpu_pixel_difference(atol=1):
    return {
        (("TestKernels", "test_cuda_vs_cpu"), dtype, "cuda"): _pixel_difference_closeness_kwargs(atol, dtype=dtype)
        for dtype in [torch.uint8, torch.float32]
    }


def pil_reference_pixel_difference(atol=1, mae=False):
    return {
        (("TestKernels", "test_against_reference"), torch.uint8, "cpu"): _pixel_difference_closeness_kwargs(
            atol, mae=mae
        )
    }


def float32_vs_uint8_pixel_difference(atol=1, mae=False):
    return {
        (
            ("TestKernels", "test_float32_vs_uint8"),
            torch.float32,
            "cpu",
        ): _pixel_difference_closeness_kwargs(atol, dtype=torch.float32, mae=mae)
    }


def scripted_vs_eager_float64_tolerances(device, atol=1e-6, rtol=1e-6):
    return {
        (("TestKernels", "test_scripted_vs_eager"), torch.float64, device): {"atol": atol, "rtol": rtol, "mae": False},
    }


def pil_reference_wrapper(pil_kernel):
    @functools.wraps(pil_kernel)
    def wrapper(input_tensor, *other_args, **kwargs):
        if input_tensor.dtype != torch.uint8:
            raise pytest.UsageError(f"Can only test uint8 tensor images against PIL, but input is {input_tensor.dtype}")
        if input_tensor.ndim > 3:
            raise pytest.UsageError(
                f"Can only test single tensor images against PIL, but input has shape {input_tensor.shape}"
            )

        input_pil = F.to_image_pil(input_tensor)
        output_pil = pil_kernel(input_pil, *other_args, **kwargs)
        if not isinstance(output_pil, PIL.Image.Image):
            return output_pil

        output_tensor = F.to_image_tensor(output_pil)

        # 2D mask shenanigans
        if output_tensor.ndim == 2 and input_tensor.ndim == 3:
            output_tensor = output_tensor.unsqueeze(0)
        elif output_tensor.ndim == 3 and input_tensor.ndim == 2:
            output_tensor = output_tensor.squeeze(0)

        return output_tensor

    return wrapper


def xfail_jit(reason, *, condition=None):
    return TestMark(("TestKernels", "test_scripted_vs_eager"), pytest.mark.xfail(reason=reason), condition=condition)


def xfail_jit_python_scalar_arg(name, *, reason=None):
    return xfail_jit(
        reason or f"Python scalar int or float for `{name}` is not supported when scripting",
        condition=lambda args_kwargs: isinstance(args_kwargs.kwargs.get(name), (int, float)),
    )


KERNEL_INFOS = []


def sample_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], dtypes=[torch.float32]):
        yield ArgsKwargs(image_loader)


def reference_inputs_horizontal_flip_image_tensor():
    for image_loader in make_image_loaders(extra_dims=[()], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_horizontal_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders(
        formats=[datapoints.BoundingBoxFormat.XYXY], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, spatial_size=bounding_box_loader.spatial_size
        )


def sample_inputs_horizontal_flip_mask():
    for image_loader in make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_horizontal_flip_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


def reference_horizontal_flip_bounding_box(bounding_box, *, format, spatial_size):
    affine_matrix = np.array(
        [
            [-1, 0, spatial_size[1]],
            [0, 1, 0],
        ],
        dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
    )

    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box, format=format, spatial_size=spatial_size, affine_matrix=affine_matrix
    )

    return expected_bboxes


def reference_inputs_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders(extra_dims=[()]):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
        )


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.horizontal_flip_image_tensor,
            kernel_name="horizontal_flip_image_tensor",
            sample_inputs_fn=sample_inputs_horizontal_flip_image_tensor,
            reference_fn=pil_reference_wrapper(F.horizontal_flip_image_pil),
            reference_inputs_fn=reference_inputs_horizontal_flip_image_tensor,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.horizontal_flip_bounding_box,
            sample_inputs_fn=sample_inputs_horizontal_flip_bounding_box,
            reference_fn=reference_horizontal_flip_bounding_box,
            reference_inputs_fn=reference_inputs_flip_bounding_box,
        ),
        KernelInfo(
            F.horizontal_flip_mask,
            sample_inputs_fn=sample_inputs_horizontal_flip_mask,
        ),
        KernelInfo(
            F.horizontal_flip_video,
            sample_inputs_fn=sample_inputs_horizontal_flip_video,
        ),
    ]
)


def _get_resize_sizes(spatial_size):
    height, width = spatial_size
    length = max(spatial_size)
    yield length
    yield [length]
    yield (length,)
    new_height = int(height * 0.75)
    new_width = int(width * 1.25)
    yield [new_height, new_width]
    yield height, width


def sample_inputs_resize_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=["RGB"], dtypes=[torch.float32]):
        for size in _get_resize_sizes(image_loader.spatial_size):
            yield ArgsKwargs(image_loader, size=size)

    for image_loader, interpolation in itertools.product(
        make_image_loaders(sizes=["random"], color_spaces=["RGB"]),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        yield ArgsKwargs(image_loader, size=[min(image_loader.spatial_size) + 1], interpolation=interpolation)

    yield ArgsKwargs(make_image_loader(size=(11, 17)), size=20, max_size=25)


@pil_reference_wrapper
def reference_resize_image_tensor(*args, **kwargs):
    if not kwargs.pop("antialias", False) and kwargs.get("interpolation", F.InterpolationMode.BILINEAR) in {
        F.InterpolationMode.BILINEAR,
        F.InterpolationMode.BICUBIC,
    }:
        raise pytest.UsageError("Anti-aliasing is always active in PIL")
    return F.resize_image_pil(*args, **kwargs)


def reference_inputs_resize_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders_for_interpolation(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.NEAREST_EXACT,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        for size in _get_resize_sizes(image_loader.spatial_size):
            yield ArgsKwargs(
                image_loader,
                size=size,
                interpolation=interpolation,
                antialias=interpolation
                in {
                    F.InterpolationMode.BILINEAR,
                    F.InterpolationMode.BICUBIC,
                },
            )


def sample_inputs_resize_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        for size in _get_resize_sizes(bounding_box_loader.spatial_size):
            yield ArgsKwargs(bounding_box_loader, spatial_size=bounding_box_loader.spatial_size, size=size)


def sample_inputs_resize_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, size=[min(mask_loader.shape[-2:]) + 1])


def sample_inputs_resize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, size=[min(video_loader.shape[-2:]) + 1])


def reference_resize_bounding_box(bounding_box, *, spatial_size, size, max_size=None):
    old_height, old_width = spatial_size
    new_height, new_width = F._geometry._compute_resized_output_size(spatial_size, size=size, max_size=max_size)

    affine_matrix = np.array(
        [
            [new_width / old_width, 0, 0],
            [0, new_height / old_height, 0],
        ],
        dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
    )

    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box,
        format=bounding_box.format,
        spatial_size=(new_height, new_width),
        affine_matrix=affine_matrix,
    )
    return expected_bboxes, (new_height, new_width)


def reference_inputs_resize_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders(extra_dims=((), (4,))):
        for size in _get_resize_sizes(bounding_box_loader.spatial_size):
            yield ArgsKwargs(bounding_box_loader, size=size, spatial_size=bounding_box_loader.spatial_size)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resize_image_tensor,
            sample_inputs_fn=sample_inputs_resize_image_tensor,
            reference_fn=reference_resize_image_tensor,
            reference_inputs_fn=reference_inputs_resize_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(10, mae=True),
                **cuda_vs_cpu_pixel_difference(),
                **float32_vs_uint8_pixel_difference(1, mae=True),
            },
            test_marks=[
                xfail_jit_python_scalar_arg("size"),
            ],
        ),
        KernelInfo(
            F.resize_bounding_box,
            sample_inputs_fn=sample_inputs_resize_bounding_box,
            reference_fn=reference_resize_bounding_box,
            reference_inputs_fn=reference_inputs_resize_bounding_box,
            closeness_kwargs={
                (("TestKernels", "test_against_reference"), torch.int64, "cpu"): dict(atol=1, rtol=0),
            },
            test_marks=[
                xfail_jit_python_scalar_arg("size"),
            ],
        ),
        KernelInfo(
            F.resize_mask,
            sample_inputs_fn=sample_inputs_resize_mask,
            closeness_kwargs=pil_reference_pixel_difference(10),
            test_marks=[
                xfail_jit_python_scalar_arg("size"),
            ],
        ),
        KernelInfo(
            F.resize_video,
            sample_inputs_fn=sample_inputs_resize_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)


_AFFINE_KWARGS = combinations_grid(
    angle=[-87, 15, 90],
    translate=[(5, 5), (-5, -5)],
    scale=[0.77, 1.27],
    shear=[(12, 12), (0, 0)],
)


def _diversify_affine_kwargs_types(affine_kwargs):
    angle = affine_kwargs["angle"]
    for diverse_angle in [int(angle), float(angle)]:
        yield dict(affine_kwargs, angle=diverse_angle)

    shear = affine_kwargs["shear"]
    for diverse_shear in [tuple(shear), list(shear), int(shear[0]), float(shear[0])]:
        yield dict(affine_kwargs, shear=diverse_shear)


def _full_affine_params(**partial_params):
    partial_params.setdefault("angle", 0.0)
    partial_params.setdefault("translate", [0.0, 0.0])
    partial_params.setdefault("scale", 1.0)
    partial_params.setdefault("shear", [0.0, 0.0])
    partial_params.setdefault("center", None)
    return partial_params


_DIVERSE_AFFINE_PARAMS = [
    _full_affine_params(**{name: arg})
    for name, args in [
        ("angle", [1.0, 2]),
        ("translate", [[1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]),
        ("scale", [0.5]),
        ("shear", [1.0, 2, [1.0], [2], (1.0,), (2,), [1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]),
        ("center", [None, [1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]),
    ]
    for arg in args
]


def get_fills(*, num_channels, dtype):
    yield None

    int_value = get_max_value(dtype)
    float_value = int_value / 2
    yield int_value
    yield float_value

    for vector_type in [list, tuple]:
        yield vector_type([int_value])
        yield vector_type([float_value])

        if num_channels > 1:
            yield vector_type(float_value * c / 10 for c in range(num_channels))
            yield vector_type(int_value if c % 2 == 0 else 0 for c in range(num_channels))


def float32_vs_uint8_fill_adapter(other_args, kwargs):
    fill = kwargs.get("fill")
    if fill is None:
        return other_args, kwargs

    if isinstance(fill, (int, float)):
        fill /= 255
    else:
        fill = type(fill)(fill_ / 255 for fill_ in fill)

    return other_args, dict(kwargs, fill=fill)


def sample_inputs_affine_image_tensor():
    make_affine_image_loaders = functools.partial(
        make_image_loaders, sizes=["random"], color_spaces=["RGB"], dtypes=[torch.float32]
    )

    for image_loader, affine_params in itertools.product(make_affine_image_loaders(), _DIVERSE_AFFINE_PARAMS):
        yield ArgsKwargs(image_loader, **affine_params)

    for image_loader in make_affine_image_loaders():
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, **_full_affine_params(), fill=fill)

    for image_loader, interpolation in itertools.product(
        make_affine_image_loaders(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
    ):
        yield ArgsKwargs(image_loader, **_full_affine_params(), fill=0)


def reference_inputs_affine_image_tensor():
    for image_loader, affine_kwargs in itertools.product(make_image_loaders_for_interpolation(), _AFFINE_KWARGS):
        yield ArgsKwargs(
            image_loader,
            interpolation=F.InterpolationMode.NEAREST,
            **affine_kwargs,
        )


def sample_inputs_affine_bounding_box():
    for bounding_box_loader, affine_params in itertools.product(
        make_bounding_box_loaders(formats=[datapoints.BoundingBoxFormat.XYXY]), _DIVERSE_AFFINE_PARAMS
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            **affine_params,
        )


def _compute_affine_matrix(angle, translate, scale, shear, center):
    rot = math.radians(angle)
    cx, cy = center
    tx, ty = translate
    sx, sy = [math.radians(sh_) for sh_ in shear]

    c_matrix = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    t_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    c_matrix_inv = np.linalg.inv(c_matrix)
    rs_matrix = np.array(
        [
            [scale * math.cos(rot), -scale * math.sin(rot), 0],
            [scale * math.sin(rot), scale * math.cos(rot), 0],
            [0, 0, 1],
        ]
    )
    shear_x_matrix = np.array([[1, -math.tan(sx), 0], [0, 1, 0], [0, 0, 1]])
    shear_y_matrix = np.array([[1, 0, 0], [-math.tan(sy), 1, 0], [0, 0, 1]])
    rss_matrix = np.matmul(rs_matrix, np.matmul(shear_y_matrix, shear_x_matrix))
    true_matrix = np.matmul(t_matrix, np.matmul(c_matrix, np.matmul(rss_matrix, c_matrix_inv)))
    return true_matrix


def reference_affine_bounding_box_helper(bounding_box, *, format, spatial_size, affine_matrix):
    def transform(bbox, affine_matrix_, format_, spatial_size_):
        # Go to float before converting to prevent precision loss in case of CXCYWH -> XYXY and W or H is 1
        in_dtype = bbox.dtype
        if not torch.is_floating_point(bbox):
            bbox = bbox.float()
        bbox_xyxy = F.convert_format_bounding_box(
            bbox.as_subclass(torch.Tensor),
            old_format=format_,
            new_format=datapoints.BoundingBoxFormat.XYXY,
            inplace=True,
        )
        points = np.array(
            [
                [bbox_xyxy[0].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[1].item(), 1.0],
                [bbox_xyxy[0].item(), bbox_xyxy[3].item(), 1.0],
                [bbox_xyxy[2].item(), bbox_xyxy[3].item(), 1.0],
            ]
        )
        transformed_points = np.matmul(points, affine_matrix_.T)
        out_bbox = torch.tensor(
            [
                np.min(transformed_points[:, 0]).item(),
                np.min(transformed_points[:, 1]).item(),
                np.max(transformed_points[:, 0]).item(),
                np.max(transformed_points[:, 1]).item(),
            ],
            dtype=bbox_xyxy.dtype,
        )
        out_bbox = F.convert_format_bounding_box(
            out_bbox, old_format=datapoints.BoundingBoxFormat.XYXY, new_format=format_, inplace=True
        )
        # It is important to clamp before casting, especially for CXCYWH format, dtype=int64
        out_bbox = F.clamp_bounding_box(out_bbox, format=format_, spatial_size=spatial_size_)
        out_bbox = out_bbox.to(dtype=in_dtype)
        return out_bbox

    if bounding_box.ndim < 2:
        bounding_box = [bounding_box]

    expected_bboxes = [transform(bbox, affine_matrix, format, spatial_size) for bbox in bounding_box]
    if len(expected_bboxes) > 1:
        expected_bboxes = torch.stack(expected_bboxes)
    else:
        expected_bboxes = expected_bboxes[0]

    return expected_bboxes


def reference_affine_bounding_box(bounding_box, *, format, spatial_size, angle, translate, scale, shear, center=None):
    if center is None:
        center = [s * 0.5 for s in spatial_size[::-1]]

    affine_matrix = _compute_affine_matrix(angle, translate, scale, shear, center)
    affine_matrix = affine_matrix[:2, :]

    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box, format=format, spatial_size=spatial_size, affine_matrix=affine_matrix
    )

    return expected_bboxes


def reference_inputs_affine_bounding_box():
    for bounding_box_loader, affine_kwargs in itertools.product(
        make_bounding_box_loaders(extra_dims=[()]),
        _AFFINE_KWARGS,
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            **affine_kwargs,
        )


def sample_inputs_affine_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, **_full_affine_params())


def sample_inputs_affine_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, **_full_affine_params())


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.affine_image_tensor,
            sample_inputs_fn=sample_inputs_affine_image_tensor,
            reference_fn=pil_reference_wrapper(F.affine_image_pil),
            reference_inputs_fn=reference_inputs_affine_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=pil_reference_pixel_difference(10, mae=True),
            test_marks=[
                xfail_jit_python_scalar_arg("shear"),
                xfail_jit_python_scalar_arg("fill"),
            ],
        ),
        KernelInfo(
            F.affine_bounding_box,
            sample_inputs_fn=sample_inputs_affine_bounding_box,
            reference_fn=reference_affine_bounding_box,
            reference_inputs_fn=reference_inputs_affine_bounding_box,
            test_marks=[
                xfail_jit_python_scalar_arg("shear"),
            ],
        ),
        KernelInfo(
            F.affine_mask,
            sample_inputs_fn=sample_inputs_affine_mask,
            test_marks=[
                xfail_jit_python_scalar_arg("shear"),
            ],
        ),
        KernelInfo(
            F.affine_video,
            sample_inputs_fn=sample_inputs_affine_video,
        ),
    ]
)


def sample_inputs_convert_format_bounding_box():
    formats = list(datapoints.BoundingBoxFormat)
    for bounding_box_loader, new_format in itertools.product(make_bounding_box_loaders(formats=formats), formats):
        yield ArgsKwargs(bounding_box_loader, old_format=bounding_box_loader.format, new_format=new_format)


def reference_convert_format_bounding_box(bounding_box, old_format, new_format):
    return torchvision.ops.box_convert(
        bounding_box, in_fmt=old_format.name.lower(), out_fmt=new_format.name.lower()
    ).to(bounding_box.dtype)


def reference_inputs_convert_format_bounding_box():
    for args_kwargs in sample_inputs_convert_format_bounding_box():
        if len(args_kwargs.args[0].shape) == 2:
            yield args_kwargs


KERNEL_INFOS.append(
    KernelInfo(
        F.convert_format_bounding_box,
        sample_inputs_fn=sample_inputs_convert_format_bounding_box,
        reference_fn=reference_convert_format_bounding_box,
        reference_inputs_fn=reference_inputs_convert_format_bounding_box,
        logs_usage=True,
    ),
)


def sample_inputs_vertical_flip_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], dtypes=[torch.float32]):
        yield ArgsKwargs(image_loader)


def reference_inputs_vertical_flip_image_tensor():
    for image_loader in make_image_loaders(extra_dims=[()], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_vertical_flip_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders(
        formats=[datapoints.BoundingBoxFormat.XYXY], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(
            bounding_box_loader, format=bounding_box_loader.format, spatial_size=bounding_box_loader.spatial_size
        )


def sample_inputs_vertical_flip_mask():
    for image_loader in make_mask_loaders(sizes=["random"], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_vertical_flip_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


def reference_vertical_flip_bounding_box(bounding_box, *, format, spatial_size):
    affine_matrix = np.array(
        [
            [1, 0, 0],
            [0, -1, spatial_size[0]],
        ],
        dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
    )

    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box, format=format, spatial_size=spatial_size, affine_matrix=affine_matrix
    )

    return expected_bboxes


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.vertical_flip_image_tensor,
            kernel_name="vertical_flip_image_tensor",
            sample_inputs_fn=sample_inputs_vertical_flip_image_tensor,
            reference_fn=pil_reference_wrapper(F.vertical_flip_image_pil),
            reference_inputs_fn=reference_inputs_vertical_flip_image_tensor,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.vertical_flip_bounding_box,
            sample_inputs_fn=sample_inputs_vertical_flip_bounding_box,
            reference_fn=reference_vertical_flip_bounding_box,
            reference_inputs_fn=reference_inputs_flip_bounding_box,
        ),
        KernelInfo(
            F.vertical_flip_mask,
            sample_inputs_fn=sample_inputs_vertical_flip_mask,
        ),
        KernelInfo(
            F.vertical_flip_video,
            sample_inputs_fn=sample_inputs_vertical_flip_video,
        ),
    ]
)

_ROTATE_ANGLES = [-87, 15, 90]


def sample_inputs_rotate_image_tensor():
    make_rotate_image_loaders = functools.partial(
        make_image_loaders, sizes=["random"], color_spaces=["RGB"], dtypes=[torch.float32]
    )

    for image_loader in make_rotate_image_loaders():
        yield ArgsKwargs(image_loader, angle=15.0, expand=True)

    for image_loader, center in itertools.product(
        make_rotate_image_loaders(), [None, [1.0, 0.5], [1, 2], (1.0, 0.5), (1, 2)]
    ):
        yield ArgsKwargs(image_loader, angle=15.0, center=center)

    for image_loader in make_rotate_image_loaders():
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, angle=15.0, fill=fill)

    for image_loader, interpolation in itertools.product(
        make_rotate_image_loaders(),
        [F.InterpolationMode.NEAREST, F.InterpolationMode.BILINEAR],
    ):
        yield ArgsKwargs(image_loader, angle=15.0, fill=0)


def reference_inputs_rotate_image_tensor():
    for image_loader, angle in itertools.product(make_image_loaders_for_interpolation(), _ROTATE_ANGLES):
        yield ArgsKwargs(image_loader, angle=angle)


def sample_inputs_rotate_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            angle=_ROTATE_ANGLES[0],
        )


def reference_inputs_rotate_bounding_box():
    for bounding_box_loader, angle in itertools.product(
        make_bounding_box_loaders(extra_dims=((), (4,))), _ROTATE_ANGLES
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            angle=angle,
        )

    # TODO: add samples with expand=True and center


def reference_rotate_bounding_box(bounding_box, *, format, spatial_size, angle, expand=False, center=None):

    if center is None:
        center = [spatial_size[1] * 0.5, spatial_size[0] * 0.5]

    a = np.cos(angle * np.pi / 180.0)
    b = np.sin(angle * np.pi / 180.0)
    cx = center[0]
    cy = center[1]
    affine_matrix = np.array(
        [
            [a, b, cx - cx * a - b * cy],
            [-b, a, cy + cx * b - a * cy],
        ],
        dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
    )

    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box, format=format, spatial_size=spatial_size, affine_matrix=affine_matrix
    )
    return expected_bboxes, spatial_size


def sample_inputs_rotate_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, angle=15.0)


def sample_inputs_rotate_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, angle=15.0)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.rotate_image_tensor,
            sample_inputs_fn=sample_inputs_rotate_image_tensor,
            reference_fn=pil_reference_wrapper(F.rotate_image_pil),
            reference_inputs_fn=reference_inputs_rotate_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=pil_reference_pixel_difference(1, mae=True),
            test_marks=[
                xfail_jit_python_scalar_arg("fill"),
            ],
        ),
        KernelInfo(
            F.rotate_bounding_box,
            sample_inputs_fn=sample_inputs_rotate_bounding_box,
            reference_fn=reference_rotate_bounding_box,
            reference_inputs_fn=reference_inputs_rotate_bounding_box,
            closeness_kwargs={
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-4, rtol=1e-4),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-4, rtol=1e-4),
            },
        ),
        KernelInfo(
            F.rotate_mask,
            sample_inputs_fn=sample_inputs_rotate_mask,
        ),
        KernelInfo(
            F.rotate_video,
            sample_inputs_fn=sample_inputs_rotate_video,
        ),
    ]
)

_CROP_PARAMS = combinations_grid(top=[-8, 0, 9], left=[-8, 0, 9], height=[12, 20], width=[12, 20])


def sample_inputs_crop_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(sizes=[(16, 17)], color_spaces=["RGB"], dtypes=[torch.float32]),
        [
            dict(top=4, left=3, height=7, width=8),
            dict(top=-1, left=3, height=7, width=8),
            dict(top=4, left=-1, height=7, width=8),
            dict(top=4, left=3, height=17, width=8),
            dict(top=4, left=3, height=7, width=18),
        ],
    ):
        yield ArgsKwargs(image_loader, **params)


def reference_inputs_crop_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(extra_dims=[()], dtypes=[torch.uint8]), _CROP_PARAMS
    ):
        yield ArgsKwargs(image_loader, **params)


def sample_inputs_crop_bounding_box():
    for bounding_box_loader, params in itertools.product(
        make_bounding_box_loaders(), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]
    ):
        yield ArgsKwargs(bounding_box_loader, format=bounding_box_loader.format, **params)


def sample_inputs_crop_mask():
    for mask_loader in make_mask_loaders(sizes=[(16, 17)], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, top=4, left=3, height=7, width=8)


def reference_inputs_crop_mask():
    for mask_loader, params in itertools.product(make_mask_loaders(extra_dims=[()], num_objects=[1]), _CROP_PARAMS):
        yield ArgsKwargs(mask_loader, **params)


def sample_inputs_crop_video():
    for video_loader in make_video_loaders(sizes=[(16, 17)], num_frames=["random"]):
        yield ArgsKwargs(video_loader, top=4, left=3, height=7, width=8)


def reference_crop_bounding_box(bounding_box, *, format, top, left, height, width):
    affine_matrix = np.array(
        [
            [1, 0, -left],
            [0, 1, -top],
        ],
        dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
    )

    spatial_size = (height, width)
    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box, format=format, spatial_size=spatial_size, affine_matrix=affine_matrix
    )
    return expected_bboxes, spatial_size


def reference_inputs_crop_bounding_box():
    for bounding_box_loader, params in itertools.product(
        make_bounding_box_loaders(extra_dims=((), (4,))), [_CROP_PARAMS[0], _CROP_PARAMS[-1]]
    ):
        yield ArgsKwargs(bounding_box_loader, format=bounding_box_loader.format, **params)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.crop_image_tensor,
            kernel_name="crop_image_tensor",
            sample_inputs_fn=sample_inputs_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.crop_image_pil),
            reference_inputs_fn=reference_inputs_crop_image_tensor,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.crop_bounding_box,
            sample_inputs_fn=sample_inputs_crop_bounding_box,
            reference_fn=reference_crop_bounding_box,
            reference_inputs_fn=reference_inputs_crop_bounding_box,
        ),
        KernelInfo(
            F.crop_mask,
            sample_inputs_fn=sample_inputs_crop_mask,
            reference_fn=pil_reference_wrapper(F.crop_image_pil),
            reference_inputs_fn=reference_inputs_crop_mask,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.crop_video,
            sample_inputs_fn=sample_inputs_crop_video,
        ),
    ]
)

_RESIZED_CROP_PARAMS = combinations_grid(top=[-8, 9], left=[-8, 9], height=[12], width=[12], size=[(16, 18)])


def sample_inputs_resized_crop_image_tensor():
    for image_loader in make_image_loaders():
        yield ArgsKwargs(image_loader, **_RESIZED_CROP_PARAMS[0])


@pil_reference_wrapper
def reference_resized_crop_image_tensor(*args, **kwargs):
    if not kwargs.pop("antialias", False) and kwargs.get("interpolation", F.InterpolationMode.BILINEAR) in {
        F.InterpolationMode.BILINEAR,
        F.InterpolationMode.BICUBIC,
    }:
        raise pytest.UsageError("Anti-aliasing is always active in PIL")
    return F.resized_crop_image_pil(*args, **kwargs)


def reference_inputs_resized_crop_image_tensor():
    for image_loader, interpolation, params in itertools.product(
        make_image_loaders_for_interpolation(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.NEAREST_EXACT,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
        _RESIZED_CROP_PARAMS,
    ):
        yield ArgsKwargs(
            image_loader,
            interpolation=interpolation,
            antialias=interpolation
            in {
                F.InterpolationMode.BILINEAR,
                F.InterpolationMode.BICUBIC,
            },
            **params,
        )


def sample_inputs_resized_crop_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(bounding_box_loader, format=bounding_box_loader.format, **_RESIZED_CROP_PARAMS[0])


def sample_inputs_resized_crop_mask():
    for mask_loader in make_mask_loaders():
        yield ArgsKwargs(mask_loader, **_RESIZED_CROP_PARAMS[0])


def sample_inputs_resized_crop_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, **_RESIZED_CROP_PARAMS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.resized_crop_image_tensor,
            sample_inputs_fn=sample_inputs_resized_crop_image_tensor,
            reference_fn=reference_resized_crop_image_tensor,
            reference_inputs_fn=reference_inputs_resized_crop_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **cuda_vs_cpu_pixel_difference(),
                **pil_reference_pixel_difference(3, mae=True),
                **float32_vs_uint8_pixel_difference(3, mae=True),
            },
        ),
        KernelInfo(
            F.resized_crop_bounding_box,
            sample_inputs_fn=sample_inputs_resized_crop_bounding_box,
        ),
        KernelInfo(
            F.resized_crop_mask,
            sample_inputs_fn=sample_inputs_resized_crop_mask,
        ),
        KernelInfo(
            F.resized_crop_video,
            sample_inputs_fn=sample_inputs_resized_crop_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)

_PAD_PARAMS = combinations_grid(
    padding=[[1], [1, 1], [1, 1, 2, 2]],
    padding_mode=["constant", "symmetric", "edge", "reflect"],
)


def sample_inputs_pad_image_tensor():
    make_pad_image_loaders = functools.partial(
        make_image_loaders, sizes=["random"], color_spaces=["RGB"], dtypes=[torch.float32]
    )

    for image_loader, padding in itertools.product(
        make_pad_image_loaders(),
        [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]],
    ):
        yield ArgsKwargs(image_loader, padding=padding)

    for image_loader in make_pad_image_loaders():
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, padding=[1], fill=fill)

    for image_loader, padding_mode in itertools.product(
        # We branch for non-constant padding and integer inputs
        make_pad_image_loaders(dtypes=[torch.uint8]),
        ["constant", "symmetric", "edge", "reflect"],
    ):
        yield ArgsKwargs(image_loader, padding=[1], padding_mode=padding_mode)

    # `torch.nn.functional.pad` does not support symmetric padding, and thus we have a custom implementation. Besides
    # negative padding, this is already handled by the inputs above.
    for image_loader in make_pad_image_loaders():
        yield ArgsKwargs(image_loader, padding=[-1], padding_mode="symmetric")


def reference_inputs_pad_image_tensor():
    for image_loader, params in itertools.product(
        make_image_loaders(extra_dims=[()], dtypes=[torch.uint8]), _PAD_PARAMS
    ):
        for fill in get_fills(
            num_channels=image_loader.num_channels,
            dtype=image_loader.dtype,
        ):
            # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
            if isinstance(fill, (list, tuple)):
                continue

            yield ArgsKwargs(image_loader, fill=fill, **params)


def sample_inputs_pad_bounding_box():
    for bounding_box_loader, padding in itertools.product(
        make_bounding_box_loaders(), [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]]
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            padding=padding,
            padding_mode="constant",
        )


def sample_inputs_pad_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        yield ArgsKwargs(mask_loader, padding=[1])


def reference_inputs_pad_mask():
    for mask_loader, fill, params in itertools.product(
        make_mask_loaders(num_objects=[1], extra_dims=[()]), [None, 127], _PAD_PARAMS
    ):
        yield ArgsKwargs(mask_loader, fill=fill, **params)


def sample_inputs_pad_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, padding=[1])


def reference_pad_bounding_box(bounding_box, *, format, spatial_size, padding, padding_mode):

    left, right, top, bottom = _parse_pad_padding(padding)

    affine_matrix = np.array(
        [
            [1, 0, left],
            [0, 1, top],
        ],
        dtype="float64" if bounding_box.dtype == torch.float64 else "float32",
    )

    height = spatial_size[0] + top + bottom
    width = spatial_size[1] + left + right

    expected_bboxes = reference_affine_bounding_box_helper(
        bounding_box, format=format, spatial_size=(height, width), affine_matrix=affine_matrix
    )
    return expected_bboxes, (height, width)


def reference_inputs_pad_bounding_box():
    for bounding_box_loader, padding in itertools.product(
        make_bounding_box_loaders(extra_dims=((), (4,))), [1, (1,), (1, 2), (1, 2, 3, 4), [1], [1, 2], [1, 2, 3, 4]]
    ):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            padding=padding,
            padding_mode="constant",
        )


def pad_xfail_jit_fill_condition(args_kwargs):
    fill = args_kwargs.kwargs.get("fill")
    if not isinstance(fill, (list, tuple)):
        return False
    elif isinstance(fill, tuple):
        return True
    else:  # isinstance(fill, list):
        return all(isinstance(f, int) for f in fill)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.pad_image_tensor,
            sample_inputs_fn=sample_inputs_pad_image_tensor,
            reference_fn=pil_reference_wrapper(F.pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_image_tensor,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
            test_marks=[
                xfail_jit_python_scalar_arg("padding"),
                xfail_jit(
                    "F.pad only supports vector fills for list of floats", condition=pad_xfail_jit_fill_condition
                ),
            ],
        ),
        KernelInfo(
            F.pad_bounding_box,
            sample_inputs_fn=sample_inputs_pad_bounding_box,
            reference_fn=reference_pad_bounding_box,
            reference_inputs_fn=reference_inputs_pad_bounding_box,
            test_marks=[
                xfail_jit_python_scalar_arg("padding"),
            ],
        ),
        KernelInfo(
            F.pad_mask,
            sample_inputs_fn=sample_inputs_pad_mask,
            reference_fn=pil_reference_wrapper(F.pad_image_pil),
            reference_inputs_fn=reference_inputs_pad_mask,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
        ),
        KernelInfo(
            F.pad_video,
            sample_inputs_fn=sample_inputs_pad_video,
        ),
    ]
)

_PERSPECTIVE_COEFFS = [
    [1.2405, 0.1772, -6.9113, 0.0463, 1.251, -5.235, 0.00013, 0.0018],
    [0.7366, -0.11724, 1.45775, -0.15012, 0.73406, 2.6019, -0.0072, -0.0063],
]
_STARTPOINTS = [[0, 1], [2, 3], [4, 5], [6, 7]]
_ENDPOINTS = [[9, 8], [7, 6], [5, 4], [3, 2]]


def sample_inputs_perspective_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(
                image_loader, startpoints=None, endpoints=None, fill=fill, coefficients=_PERSPECTIVE_COEFFS[0]
            )

    yield ArgsKwargs(make_image_loader(), startpoints=_STARTPOINTS, endpoints=_ENDPOINTS)


def reference_inputs_perspective_image_tensor():
    for image_loader, coefficients, interpolation in itertools.product(
        make_image_loaders_for_interpolation(),
        _PERSPECTIVE_COEFFS,
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
        ],
    ):
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            # FIXME: PIL kernel doesn't support sequences of length 1 if the number of channels is larger. Shouldn't it?
            if isinstance(fill, (list, tuple)):
                continue

            yield ArgsKwargs(
                image_loader,
                startpoints=None,
                endpoints=None,
                interpolation=interpolation,
                fill=fill,
                coefficients=coefficients,
            )


def sample_inputs_perspective_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            startpoints=None,
            endpoints=None,
            coefficients=_PERSPECTIVE_COEFFS[0],
        )

    format = datapoints.BoundingBoxFormat.XYXY
    loader = make_bounding_box_loader(format=format)
    yield ArgsKwargs(
        loader, format=format, spatial_size=loader.spatial_size, startpoints=_STARTPOINTS, endpoints=_ENDPOINTS
    )


def sample_inputs_perspective_mask():
    for mask_loader in make_mask_loaders(sizes=["random"]):
        yield ArgsKwargs(mask_loader, startpoints=None, endpoints=None, coefficients=_PERSPECTIVE_COEFFS[0])

    yield ArgsKwargs(make_detection_mask_loader(), startpoints=_STARTPOINTS, endpoints=_ENDPOINTS)


def reference_inputs_perspective_mask():
    for mask_loader, perspective_coeffs in itertools.product(
        make_mask_loaders(extra_dims=[()], num_objects=[1]), _PERSPECTIVE_COEFFS
    ):
        yield ArgsKwargs(mask_loader, startpoints=None, endpoints=None, coefficients=perspective_coeffs)


def sample_inputs_perspective_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, startpoints=None, endpoints=None, coefficients=_PERSPECTIVE_COEFFS[0])

    yield ArgsKwargs(make_video_loader(), startpoints=_STARTPOINTS, endpoints=_ENDPOINTS)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.perspective_image_tensor,
            sample_inputs_fn=sample_inputs_perspective_image_tensor,
            reference_fn=pil_reference_wrapper(F.perspective_image_pil),
            reference_inputs_fn=reference_inputs_perspective_image_tensor,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
            closeness_kwargs={
                **pil_reference_pixel_difference(2, mae=True),
                **cuda_vs_cpu_pixel_difference(),
                **float32_vs_uint8_pixel_difference(),
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-5, rtol=1e-5),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-5, rtol=1e-5),
            },
            test_marks=[xfail_jit_python_scalar_arg("fill")],
        ),
        KernelInfo(
            F.perspective_bounding_box,
            sample_inputs_fn=sample_inputs_perspective_bounding_box,
            closeness_kwargs={
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-6, rtol=1e-6),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-6, rtol=1e-6),
            },
        ),
        KernelInfo(
            F.perspective_mask,
            sample_inputs_fn=sample_inputs_perspective_mask,
            reference_fn=pil_reference_wrapper(F.perspective_image_pil),
            reference_inputs_fn=reference_inputs_perspective_mask,
            float32_vs_uint8=True,
            closeness_kwargs={
                (("TestKernels", "test_against_reference"), torch.uint8, "cpu"): dict(atol=10, rtol=0),
            },
        ),
        KernelInfo(
            F.perspective_video,
            sample_inputs_fn=sample_inputs_perspective_video,
            closeness_kwargs={
                **cuda_vs_cpu_pixel_difference(),
                **scripted_vs_eager_float64_tolerances("cpu", atol=1e-5, rtol=1e-5),
                **scripted_vs_eager_float64_tolerances("cuda", atol=1e-5, rtol=1e-5),
            },
        ),
    ]
)


def _get_elastic_displacement(spatial_size):
    return torch.rand(1, *spatial_size, 2)


def sample_inputs_elastic_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        displacement = _get_elastic_displacement(image_loader.spatial_size)
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, displacement=displacement, fill=fill)


def reference_inputs_elastic_image_tensor():
    for image_loader, interpolation in itertools.product(
        make_image_loaders_for_interpolation(),
        [
            F.InterpolationMode.NEAREST,
            F.InterpolationMode.BILINEAR,
            F.InterpolationMode.BICUBIC,
        ],
    ):
        displacement = _get_elastic_displacement(image_loader.spatial_size)
        for fill in get_fills(num_channels=image_loader.num_channels, dtype=image_loader.dtype):
            yield ArgsKwargs(image_loader, interpolation=interpolation, displacement=displacement, fill=fill)


def sample_inputs_elastic_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        displacement = _get_elastic_displacement(bounding_box_loader.spatial_size)
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            displacement=displacement,
        )


def sample_inputs_elastic_mask():
    for mask_loader in make_mask_loaders(sizes=["random"]):
        displacement = _get_elastic_displacement(mask_loader.shape[-2:])
        yield ArgsKwargs(mask_loader, displacement=displacement)


def sample_inputs_elastic_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        displacement = _get_elastic_displacement(video_loader.shape[-2:])
        yield ArgsKwargs(video_loader, displacement=displacement)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.elastic_image_tensor,
            sample_inputs_fn=sample_inputs_elastic_image_tensor,
            reference_inputs_fn=reference_inputs_elastic_image_tensor,
            float32_vs_uint8=float32_vs_uint8_fill_adapter,
            closeness_kwargs={
                **float32_vs_uint8_pixel_difference(6, mae=True),
                **cuda_vs_cpu_pixel_difference(),
            },
            test_marks=[xfail_jit_python_scalar_arg("fill")],
        ),
        KernelInfo(
            F.elastic_bounding_box,
            sample_inputs_fn=sample_inputs_elastic_bounding_box,
        ),
        KernelInfo(
            F.elastic_mask,
            sample_inputs_fn=sample_inputs_elastic_mask,
        ),
        KernelInfo(
            F.elastic_video,
            sample_inputs_fn=sample_inputs_elastic_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)


_CENTER_CROP_SPATIAL_SIZES = [(16, 16), (7, 33), (31, 9)]
_CENTER_CROP_OUTPUT_SIZES = [[4, 3], [42, 70], [4], 3, (5, 2), (6,)]


def sample_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=[(16, 17)], color_spaces=["RGB"], dtypes=[torch.float32]),
        [
            # valid `output_size` types for which cropping is applied to both dimensions
            *[5, (4,), (2, 3), [6], [3, 2]],
            # `output_size`'s for which at least one dimension needs to be padded
            *[[4, 18], [17, 5], [17, 18]],
        ],
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def reference_inputs_center_crop_image_tensor():
    for image_loader, output_size in itertools.product(
        make_image_loaders(sizes=_CENTER_CROP_SPATIAL_SIZES, extra_dims=[()], dtypes=[torch.uint8]),
        _CENTER_CROP_OUTPUT_SIZES,
    ):
        yield ArgsKwargs(image_loader, output_size=output_size)


def sample_inputs_center_crop_bounding_box():
    for bounding_box_loader, output_size in itertools.product(make_bounding_box_loaders(), _CENTER_CROP_OUTPUT_SIZES):
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
            output_size=output_size,
        )


def sample_inputs_center_crop_mask():
    for mask_loader in make_mask_loaders(sizes=["random"], num_categories=["random"], num_objects=["random"]):
        height, width = mask_loader.shape[-2:]
        yield ArgsKwargs(mask_loader, output_size=(height // 2, width // 2))


def reference_inputs_center_crop_mask():
    for mask_loader, output_size in itertools.product(
        make_mask_loaders(sizes=_CENTER_CROP_SPATIAL_SIZES, extra_dims=[()], num_objects=[1]), _CENTER_CROP_OUTPUT_SIZES
    ):
        yield ArgsKwargs(mask_loader, output_size=output_size)


def sample_inputs_center_crop_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        height, width = video_loader.shape[-2:]
        yield ArgsKwargs(video_loader, output_size=(height // 2, width // 2))


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.center_crop_image_tensor,
            sample_inputs_fn=sample_inputs_center_crop_image_tensor,
            reference_fn=pil_reference_wrapper(F.center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_image_tensor,
            float32_vs_uint8=True,
            test_marks=[
                xfail_jit_python_scalar_arg("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_bounding_box,
            sample_inputs_fn=sample_inputs_center_crop_bounding_box,
            test_marks=[
                xfail_jit_python_scalar_arg("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_mask,
            sample_inputs_fn=sample_inputs_center_crop_mask,
            reference_fn=pil_reference_wrapper(F.center_crop_image_pil),
            reference_inputs_fn=reference_inputs_center_crop_mask,
            float32_vs_uint8=True,
            test_marks=[
                xfail_jit_python_scalar_arg("output_size"),
            ],
        ),
        KernelInfo(
            F.center_crop_video,
            sample_inputs_fn=sample_inputs_center_crop_video,
        ),
    ]
)


def sample_inputs_gaussian_blur_image_tensor():
    make_gaussian_blur_image_loaders = functools.partial(make_image_loaders, sizes=[(7, 33)], color_spaces=["RGB"])

    for image_loader, kernel_size in itertools.product(make_gaussian_blur_image_loaders(), [5, (3, 3), [3, 3]]):
        yield ArgsKwargs(image_loader, kernel_size=kernel_size)

    for image_loader, sigma in itertools.product(
        make_gaussian_blur_image_loaders(), [None, (3.0, 3.0), [2.0, 2.0], 4.0, [1.5], (3.14,)]
    ):
        yield ArgsKwargs(image_loader, kernel_size=5, sigma=sigma)


def sample_inputs_gaussian_blur_video():
    for video_loader in make_video_loaders(sizes=[(7, 33)], num_frames=[5]):
        yield ArgsKwargs(video_loader, kernel_size=[3, 3])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.gaussian_blur_image_tensor,
            sample_inputs_fn=sample_inputs_gaussian_blur_image_tensor,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
            test_marks=[
                xfail_jit_python_scalar_arg("kernel_size"),
                xfail_jit_python_scalar_arg("sigma"),
            ],
        ),
        KernelInfo(
            F.gaussian_blur_video,
            sample_inputs_fn=sample_inputs_gaussian_blur_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
        ),
    ]
)


def sample_inputs_equalize_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader)


def reference_inputs_equalize_image_tensor():
    # We are not using `make_image_loaders` here since that uniformly samples the values over the whole value range.
    # Since the whole point of this kernel is to transform an arbitrary distribution of values into a uniform one,
    # the information gain is low if we already provide something really close to the expected value.
    def make_uniform_band_image(shape, dtype, device, *, low_factor, high_factor):
        if dtype.is_floating_point:
            low = low_factor
            high = high_factor
        else:
            max_value = torch.iinfo(dtype).max
            low = int(low_factor * max_value)
            high = int(high_factor * max_value)
        return torch.testing.make_tensor(shape, dtype=dtype, device=device, low=low, high=high)

    def make_beta_distributed_image(shape, dtype, device, *, alpha, beta):
        image = torch.distributions.Beta(alpha, beta).sample(shape)
        if not dtype.is_floating_point:
            image.mul_(torch.iinfo(dtype).max).round_()
        return image.to(dtype=dtype, device=device)

    spatial_size = (256, 256)
    for dtype, color_space, fn in itertools.product(
        [torch.uint8],
        ["GRAY", "RGB"],
        [
            lambda shape, dtype, device: torch.zeros(shape, dtype=dtype, device=device),
            lambda shape, dtype, device: torch.full(
                shape, 1.0 if dtype.is_floating_point else torch.iinfo(dtype).max, dtype=dtype, device=device
            ),
            *[
                functools.partial(make_uniform_band_image, low_factor=low_factor, high_factor=high_factor)
                for low_factor, high_factor in [
                    (0.0, 0.25),
                    (0.25, 0.75),
                    (0.75, 1.0),
                ]
            ],
            *[
                functools.partial(make_beta_distributed_image, alpha=alpha, beta=beta)
                for alpha, beta in [
                    (0.5, 0.5),
                    (2, 2),
                    (2, 5),
                    (5, 2),
                ]
            ],
        ],
    ):
        image_loader = ImageLoader(fn, shape=(get_num_channels(color_space), *spatial_size), dtype=dtype)
        yield ArgsKwargs(image_loader)


def sample_inputs_equalize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.equalize_image_tensor,
            kernel_name="equalize_image_tensor",
            sample_inputs_fn=sample_inputs_equalize_image_tensor,
            reference_fn=pil_reference_wrapper(F.equalize_image_pil),
            float32_vs_uint8=True,
            reference_inputs_fn=reference_inputs_equalize_image_tensor,
        ),
        KernelInfo(
            F.equalize_video,
            sample_inputs_fn=sample_inputs_equalize_video,
        ),
    ]
)


def sample_inputs_invert_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader)


def reference_inputs_invert_image_tensor():
    for image_loader in make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_invert_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.invert_image_tensor,
            kernel_name="invert_image_tensor",
            sample_inputs_fn=sample_inputs_invert_image_tensor,
            reference_fn=pil_reference_wrapper(F.invert_image_pil),
            reference_inputs_fn=reference_inputs_invert_image_tensor,
            float32_vs_uint8=True,
        ),
        KernelInfo(
            F.invert_video,
            sample_inputs_fn=sample_inputs_invert_video,
        ),
    ]
)


_POSTERIZE_BITS = [1, 4, 8]


def sample_inputs_posterize_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, bits=_POSTERIZE_BITS[0])


def reference_inputs_posterize_image_tensor():
    for image_loader, bits in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _POSTERIZE_BITS,
    ):
        yield ArgsKwargs(image_loader, bits=bits)


def sample_inputs_posterize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, bits=_POSTERIZE_BITS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.posterize_image_tensor,
            kernel_name="posterize_image_tensor",
            sample_inputs_fn=sample_inputs_posterize_image_tensor,
            reference_fn=pil_reference_wrapper(F.posterize_image_pil),
            reference_inputs_fn=reference_inputs_posterize_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
        ),
        KernelInfo(
            F.posterize_video,
            sample_inputs_fn=sample_inputs_posterize_video,
        ),
    ]
)


def _get_solarize_thresholds(dtype):
    for factor in [0.1, 0.5]:
        max_value = get_max_value(dtype)
        yield (float if dtype.is_floating_point else int)(max_value * factor)


def sample_inputs_solarize_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, threshold=next(_get_solarize_thresholds(image_loader.dtype)))


def reference_inputs_solarize_image_tensor():
    for image_loader in make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]):
        for threshold in _get_solarize_thresholds(image_loader.dtype):
            yield ArgsKwargs(image_loader, threshold=threshold)


def uint8_to_float32_threshold_adapter(other_args, kwargs):
    return other_args, dict(threshold=kwargs["threshold"] / 255)


def sample_inputs_solarize_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, threshold=next(_get_solarize_thresholds(video_loader.dtype)))


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.solarize_image_tensor,
            kernel_name="solarize_image_tensor",
            sample_inputs_fn=sample_inputs_solarize_image_tensor,
            reference_fn=pil_reference_wrapper(F.solarize_image_pil),
            reference_inputs_fn=reference_inputs_solarize_image_tensor,
            float32_vs_uint8=uint8_to_float32_threshold_adapter,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
        ),
        KernelInfo(
            F.solarize_video,
            sample_inputs_fn=sample_inputs_solarize_video,
        ),
    ]
)


def sample_inputs_autocontrast_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader)


def reference_inputs_autocontrast_image_tensor():
    for image_loader in make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]):
        yield ArgsKwargs(image_loader)


def sample_inputs_autocontrast_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.autocontrast_image_tensor,
            kernel_name="autocontrast_image_tensor",
            sample_inputs_fn=sample_inputs_autocontrast_image_tensor,
            reference_fn=pil_reference_wrapper(F.autocontrast_image_pil),
            reference_inputs_fn=reference_inputs_autocontrast_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(),
            },
        ),
        KernelInfo(
            F.autocontrast_video,
            sample_inputs_fn=sample_inputs_autocontrast_video,
        ),
    ]
)

_ADJUST_SHARPNESS_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_sharpness_image_tensor():
    for image_loader in make_image_loaders(
        sizes=["random", (2, 2)],
        color_spaces=("GRAY", "RGB"),
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


def reference_inputs_adjust_sharpness_image_tensor():
    for image_loader, sharpness_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_SHARPNESS_FACTORS,
    ):
        yield ArgsKwargs(image_loader, sharpness_factor=sharpness_factor)


def sample_inputs_adjust_sharpness_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, sharpness_factor=_ADJUST_SHARPNESS_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_sharpness_image_tensor,
            kernel_name="adjust_sharpness_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_sharpness_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_sharpness_image_pil),
            reference_inputs_fn=reference_inputs_adjust_sharpness_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=float32_vs_uint8_pixel_difference(2),
        ),
        KernelInfo(
            F.adjust_sharpness_video,
            sample_inputs_fn=sample_inputs_adjust_sharpness_video,
        ),
    ]
)


def sample_inputs_erase_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(image_loader.num_channels, h, w)
        yield ArgsKwargs(image_loader, i=1, j=2, h=h, w=w, v=v)


def sample_inputs_erase_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        # FIXME: make the parameters more diverse
        h, w = 6, 7
        v = torch.rand(video_loader.num_channels, h, w)
        yield ArgsKwargs(video_loader, i=1, j=2, h=h, w=w, v=v)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.erase_image_tensor,
            kernel_name="erase_image_tensor",
            sample_inputs_fn=sample_inputs_erase_image_tensor,
        ),
        KernelInfo(
            F.erase_video,
            sample_inputs_fn=sample_inputs_erase_video,
        ),
    ]
)

_ADJUST_BRIGHTNESS_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_brightness_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, brightness_factor=_ADJUST_BRIGHTNESS_FACTORS[0])


def reference_inputs_adjust_brightness_image_tensor():
    for image_loader, brightness_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_BRIGHTNESS_FACTORS,
    ):
        yield ArgsKwargs(image_loader, brightness_factor=brightness_factor)


def sample_inputs_adjust_brightness_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, brightness_factor=_ADJUST_BRIGHTNESS_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_brightness_image_tensor,
            kernel_name="adjust_brightness_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_brightness_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_brightness_image_pil),
            reference_inputs_fn=reference_inputs_adjust_brightness_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs=float32_vs_uint8_pixel_difference(),
        ),
        KernelInfo(
            F.adjust_brightness_video,
            sample_inputs_fn=sample_inputs_adjust_brightness_video,
        ),
    ]
)


_ADJUST_CONTRAST_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_contrast_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, contrast_factor=_ADJUST_CONTRAST_FACTORS[0])


def reference_inputs_adjust_contrast_image_tensor():
    for image_loader, contrast_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_CONTRAST_FACTORS,
    ):
        yield ArgsKwargs(image_loader, contrast_factor=contrast_factor)


def sample_inputs_adjust_contrast_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, contrast_factor=_ADJUST_CONTRAST_FACTORS[0])


# TODO: this is just temporary to make CI green for release. We should add proper tolerances after
skip_adjust_contrast_jit = TestMark(("TestKernels", "test_scripted_vs_eager"), pytest.mark.skip(reason="Test is flaky"))

KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_contrast_image_tensor,
            kernel_name="adjust_contrast_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_contrast_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_contrast_image_pil),
            reference_inputs_fn=reference_inputs_adjust_contrast_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(2),
                **cuda_vs_cpu_pixel_difference(),
            },
            test_marks=[skip_adjust_contrast_jit],
        ),
        KernelInfo(
            F.adjust_contrast_video,
            sample_inputs_fn=sample_inputs_adjust_contrast_video,
            closeness_kwargs=cuda_vs_cpu_pixel_difference(),
            test_marks=[skip_adjust_contrast_jit],
        ),
    ]
)

_ADJUST_GAMMA_GAMMAS_GAINS = [
    (0.5, 2.0),
    (0.0, 1.0),
]


def sample_inputs_adjust_gamma_image_tensor():
    gamma, gain = _ADJUST_GAMMA_GAMMAS_GAINS[0]
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, gamma=gamma, gain=gain)


def reference_inputs_adjust_gamma_image_tensor():
    for image_loader, (gamma, gain) in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_GAMMA_GAMMAS_GAINS,
    ):
        yield ArgsKwargs(image_loader, gamma=gamma, gain=gain)


def sample_inputs_adjust_gamma_video():
    gamma, gain = _ADJUST_GAMMA_GAMMAS_GAINS[0]
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, gamma=gamma, gain=gain)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_gamma_image_tensor,
            kernel_name="adjust_gamma_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_gamma_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_gamma_image_pil),
            reference_inputs_fn=reference_inputs_adjust_gamma_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(),
            },
        ),
        KernelInfo(
            F.adjust_gamma_video,
            sample_inputs_fn=sample_inputs_adjust_gamma_video,
        ),
    ]
)


_ADJUST_HUE_FACTORS = [-0.1, 0.5]


def sample_inputs_adjust_hue_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, hue_factor=_ADJUST_HUE_FACTORS[0])


def reference_inputs_adjust_hue_image_tensor():
    for image_loader, hue_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_HUE_FACTORS,
    ):
        yield ArgsKwargs(image_loader, hue_factor=hue_factor)


def sample_inputs_adjust_hue_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, hue_factor=_ADJUST_HUE_FACTORS[0])


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_hue_image_tensor,
            kernel_name="adjust_hue_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_hue_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_hue_image_pil),
            reference_inputs_fn=reference_inputs_adjust_hue_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(2, mae=True),
                **float32_vs_uint8_pixel_difference(),
            },
        ),
        KernelInfo(
            F.adjust_hue_video,
            sample_inputs_fn=sample_inputs_adjust_hue_video,
        ),
    ]
)

_ADJUST_SATURATION_FACTORS = [0.1, 0.5]


def sample_inputs_adjust_saturation_image_tensor():
    for image_loader in make_image_loaders(sizes=["random"], color_spaces=("GRAY", "RGB")):
        yield ArgsKwargs(image_loader, saturation_factor=_ADJUST_SATURATION_FACTORS[0])


def reference_inputs_adjust_saturation_image_tensor():
    for image_loader, saturation_factor in itertools.product(
        make_image_loaders(color_spaces=("GRAY", "RGB"), extra_dims=[()], dtypes=[torch.uint8]),
        _ADJUST_SATURATION_FACTORS,
    ):
        yield ArgsKwargs(image_loader, saturation_factor=saturation_factor)


def sample_inputs_adjust_saturation_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader, saturation_factor=_ADJUST_SATURATION_FACTORS[0])


# TODO: this is just temporary to make CI green for release. We should add proper tolerances after
skip_adjust_saturation_cuda = TestMark(("TestKernels", "test_cuda_vs_cpu"), pytest.mark.skip(reason="Test is flaky"))

KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.adjust_saturation_image_tensor,
            kernel_name="adjust_saturation_image_tensor",
            sample_inputs_fn=sample_inputs_adjust_saturation_image_tensor,
            reference_fn=pil_reference_wrapper(F.adjust_saturation_image_pil),
            reference_inputs_fn=reference_inputs_adjust_saturation_image_tensor,
            float32_vs_uint8=True,
            closeness_kwargs={
                **pil_reference_pixel_difference(),
                **float32_vs_uint8_pixel_difference(2),
            },
            test_marks=[skip_adjust_saturation_cuda],
        ),
        KernelInfo(
            F.adjust_saturation_video,
            sample_inputs_fn=sample_inputs_adjust_saturation_video,
            test_marks=[skip_adjust_saturation_cuda],
        ),
    ]
)


def sample_inputs_clamp_bounding_box():
    for bounding_box_loader in make_bounding_box_loaders():
        yield ArgsKwargs(
            bounding_box_loader,
            format=bounding_box_loader.format,
            spatial_size=bounding_box_loader.spatial_size,
        )


KERNEL_INFOS.append(
    KernelInfo(
        F.clamp_bounding_box,
        sample_inputs_fn=sample_inputs_clamp_bounding_box,
        logs_usage=True,
    )
)

_FIVE_TEN_CROP_SIZES = [7, (6,), [5], (6, 5), [7, 6]]


def _get_five_ten_crop_spatial_size(size):
    if isinstance(size, int):
        crop_height = crop_width = size
    elif len(size) == 1:
        crop_height = crop_width = size[0]
    else:
        crop_height, crop_width = size
    return 2 * crop_height, 2 * crop_width


def sample_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_spatial_size(size)],
            color_spaces=["RGB"],
            dtypes=[torch.float32],
        ):
            yield ArgsKwargs(image_loader, size=size)


def reference_inputs_five_crop_image_tensor():
    for size in _FIVE_TEN_CROP_SIZES:
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_spatial_size(size)], extra_dims=[()], dtypes=[torch.uint8]
        ):
            yield ArgsKwargs(image_loader, size=size)


def sample_inputs_five_crop_video():
    size = _FIVE_TEN_CROP_SIZES[0]
    for video_loader in make_video_loaders(sizes=[_get_five_ten_crop_spatial_size(size)]):
        yield ArgsKwargs(video_loader, size=size)


def sample_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_spatial_size(size)],
            color_spaces=["RGB"],
            dtypes=[torch.float32],
        ):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


def reference_inputs_ten_crop_image_tensor():
    for size, vertical_flip in itertools.product(_FIVE_TEN_CROP_SIZES, [False, True]):
        for image_loader in make_image_loaders(
            sizes=[_get_five_ten_crop_spatial_size(size)], extra_dims=[()], dtypes=[torch.uint8]
        ):
            yield ArgsKwargs(image_loader, size=size, vertical_flip=vertical_flip)


def sample_inputs_ten_crop_video():
    size = _FIVE_TEN_CROP_SIZES[0]
    for video_loader in make_video_loaders(sizes=[_get_five_ten_crop_spatial_size(size)]):
        yield ArgsKwargs(video_loader, size=size)


def multi_crop_pil_reference_wrapper(pil_kernel):
    def wrapper(input_tensor, *other_args, **kwargs):
        output = pil_reference_wrapper(pil_kernel)(input_tensor, *other_args, **kwargs)
        return type(output)(
            F.convert_dtype_image_tensor(F.to_image_tensor(output_pil), dtype=input_tensor.dtype)
            for output_pil in output
        )

    return wrapper


_common_five_ten_crop_marks = [
    xfail_jit_python_scalar_arg("size"),
    mark_framework_limitation(("TestKernels", "test_batched_vs_single"), "Custom batching needed."),
]

KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.five_crop_image_tensor,
            sample_inputs_fn=sample_inputs_five_crop_image_tensor,
            reference_fn=multi_crop_pil_reference_wrapper(F.five_crop_image_pil),
            reference_inputs_fn=reference_inputs_five_crop_image_tensor,
            test_marks=_common_five_ten_crop_marks,
        ),
        KernelInfo(
            F.five_crop_video,
            sample_inputs_fn=sample_inputs_five_crop_video,
            test_marks=_common_five_ten_crop_marks,
        ),
        KernelInfo(
            F.ten_crop_image_tensor,
            sample_inputs_fn=sample_inputs_ten_crop_image_tensor,
            reference_fn=multi_crop_pil_reference_wrapper(F.ten_crop_image_pil),
            reference_inputs_fn=reference_inputs_ten_crop_image_tensor,
            test_marks=_common_five_ten_crop_marks,
        ),
        KernelInfo(
            F.ten_crop_video,
            sample_inputs_fn=sample_inputs_ten_crop_video,
            test_marks=_common_five_ten_crop_marks,
        ),
    ]
)

_NORMALIZE_MEANS_STDS = [
    ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    (0.5, 2.0),
]


def sample_inputs_normalize_image_tensor():
    for image_loader, (mean, std) in itertools.product(
        make_image_loaders(sizes=["random"], color_spaces=["RGB"], dtypes=[torch.float32]),
        _NORMALIZE_MEANS_STDS,
    ):
        yield ArgsKwargs(image_loader, mean=mean, std=std)


def reference_normalize_image_tensor(image, mean, std, inplace=False):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    sub = torch.Tensor.sub_ if inplace else torch.Tensor.sub
    return sub(image, mean).div_(std)


def reference_inputs_normalize_image_tensor():
    yield ArgsKwargs(
        make_image_loader(size=(32, 32), color_space="RGB", extra_dims=[1]),
        mean=[0.5, 0.5, 0.5],
        std=[1.0, 1.0, 1.0],
    )


def sample_inputs_normalize_video():
    mean, std = _NORMALIZE_MEANS_STDS[0]
    for video_loader in make_video_loaders(
        sizes=["random"], color_spaces=["RGB"], num_frames=["random"], dtypes=[torch.float32]
    ):
        yield ArgsKwargs(video_loader, mean=mean, std=std)


KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.normalize_image_tensor,
            kernel_name="normalize_image_tensor",
            sample_inputs_fn=sample_inputs_normalize_image_tensor,
            reference_fn=reference_normalize_image_tensor,
            reference_inputs_fn=reference_inputs_normalize_image_tensor,
            test_marks=[
                xfail_jit_python_scalar_arg("mean"),
                xfail_jit_python_scalar_arg("std"),
            ],
        ),
        KernelInfo(
            F.normalize_video,
            sample_inputs_fn=sample_inputs_normalize_video,
        ),
    ]
)


def sample_inputs_convert_dtype_image_tensor():
    for input_dtype, output_dtype in itertools.product(
        [torch.uint8, torch.int64, torch.float32, torch.float64], repeat=2
    ):
        if input_dtype.is_floating_point and output_dtype == torch.int64:
            # conversion cannot be performed safely
            continue

        for image_loader in make_image_loaders(sizes=["random"], color_spaces=["RGB"], dtypes=[input_dtype]):
            yield ArgsKwargs(image_loader, dtype=output_dtype)


def reference_convert_dtype_image_tensor(image, dtype=torch.float):
    input_dtype = image.dtype
    output_dtype = dtype

    if output_dtype == input_dtype:
        return image

    def fn(value):
        if input_dtype.is_floating_point:
            if output_dtype.is_floating_point:
                return value
            else:
                return int(decimal.Decimal(value) * torch.iinfo(output_dtype).max)
        else:
            input_max_value = torch.iinfo(input_dtype).max

            if output_dtype.is_floating_point:
                return float(decimal.Decimal(value) / input_max_value)
            else:
                output_max_value = torch.iinfo(output_dtype).max

                if input_max_value > output_max_value:
                    factor = (input_max_value + 1) // (output_max_value + 1)
                    return value // factor
                else:
                    factor = (output_max_value + 1) // (input_max_value + 1)
                    return value * factor

    return torch.tensor(tree_map(fn, image.tolist()), dtype=dtype)


def reference_inputs_convert_dtype_image_tensor():
    for input_dtype, output_dtype in itertools.product(
        [
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ],
        repeat=2,
    ):
        if (input_dtype == torch.float32 and output_dtype in {torch.int32, torch.int64}) or (
            input_dtype == torch.float64 and output_dtype == torch.int64
        ):
            continue

        if input_dtype.is_floating_point:
            data = [0.0, 0.5, 1.0]
        else:
            max_value = torch.iinfo(input_dtype).max
            data = [0, max_value // 2, max_value]
        image = torch.tensor(data, dtype=input_dtype)

        yield ArgsKwargs(image, dtype=output_dtype)


def sample_inputs_convert_dtype_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=["random"]):
        yield ArgsKwargs(video_loader)


skip_dtype_consistency = TestMark(
    ("TestKernels", "test_dtype_and_device_consistency"),
    pytest.mark.skip(reason="`convert_dtype_*` kernels convert the dtype by design"),
    condition=lambda args_kwargs: args_kwargs.args[0].dtype != args_kwargs.kwargs.get("dtype", torch.float32),
)

KERNEL_INFOS.extend(
    [
        KernelInfo(
            F.convert_dtype_image_tensor,
            sample_inputs_fn=sample_inputs_convert_dtype_image_tensor,
            reference_fn=reference_convert_dtype_image_tensor,
            reference_inputs_fn=reference_inputs_convert_dtype_image_tensor,
            test_marks=[
                skip_dtype_consistency,
                TestMark(
                    ("TestKernels", "test_against_reference"),
                    pytest.mark.xfail(reason="Conversion overflows"),
                    condition=lambda args_kwargs: (
                        args_kwargs.args[0].dtype in {torch.float16, torch.bfloat16}
                        and not args_kwargs.kwargs["dtype"].is_floating_point
                    )
                    or (
                        args_kwargs.args[0].dtype in {torch.int32, torch.int64}
                        and args_kwargs.kwargs["dtype"] == torch.float16
                    ),
                ),
            ],
        ),
        KernelInfo(
            F.convert_dtype_video,
            sample_inputs_fn=sample_inputs_convert_dtype_video,
            test_marks=[
                skip_dtype_consistency,
            ],
        ),
    ]
)


def sample_inputs_uniform_temporal_subsample_video():
    for video_loader in make_video_loaders(sizes=["random"], num_frames=[4]):
        yield ArgsKwargs(video_loader, num_samples=2)


def reference_uniform_temporal_subsample_video(x, num_samples):
    # Copy-pasted from
    # https://github.com/facebookresearch/pytorchvideo/blob/c8d23d8b7e597586a9e2d18f6ed31ad8aa379a7a/pytorchvideo/transforms/functional.py#L19
    t = x.shape[-4]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, -4, indices)


def reference_inputs_uniform_temporal_subsample_video():
    for video_loader in make_video_loaders(sizes=["random"], color_spaces=["RGB"], num_frames=[10]):
        for num_samples in range(1, video_loader.shape[-4] + 1):
            yield ArgsKwargs(video_loader, num_samples)


KERNEL_INFOS.append(
    KernelInfo(
        F.uniform_temporal_subsample_video,
        sample_inputs_fn=sample_inputs_uniform_temporal_subsample_video,
        reference_fn=reference_uniform_temporal_subsample_video,
        reference_inputs_fn=reference_inputs_uniform_temporal_subsample_video,
    )
)
