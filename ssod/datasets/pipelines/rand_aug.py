"""
Modified from https://github.com/google-research/ssl_detection/blob/master/detection/utils/augmentation.py.
"""
import copy

import cv2
import mmcv
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from mmcv.image.colorspace import bgr2rgb, rgb2bgr
from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose as BaseCompose
from mmdet.datasets.pipelines import transforms
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle, to_tensor
from mmcv.parallel import DataContainer as DC
from imgaug import augmenters as iaa
from .geo_utils import GeometricTransformationBase as GTrans

PARAMETER_MAX = 10


def int_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return int(level * maxval / max_level)


def float_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return float(level) * maxval / max_level


class RandAug(object):
    """refer to https://github.com/google-research/ssl_detection/blob/00d52272f
    61b56eade8d5ace18213cba6c74f6d8/detection/utils/augmentation.py#L240."""

    def __init__(
        self,
        prob: float = 1.0,
        magnitude: int = 10,
        random_magnitude: bool = True,
        record: bool = False,
        magnitude_limit: int = 10,
    ):
        assert 0 <= prob <= 1, f"probability should be in (0,1) but get {prob}"
        assert (
            magnitude <= PARAMETER_MAX
        ), f"magnitude should be small than max value {PARAMETER_MAX} but get {magnitude}"

        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_limit = magnitude_limit
        self.random_magnitude = random_magnitude
        self.record = record
        self.buffer = None

    def __call__(self, results):
        if np.random.random() < self.prob:
            magnitude = self.magnitude
            if self.random_magnitude:
                magnitude = np.random.randint(1, magnitude)
            if self.record:
                if "aug_info" not in results:
                    results["aug_info"] = []
                results["aug_info"].append(self.get_aug_info(magnitude=magnitude))
            results = self.apply(results, magnitude)
        # clear buffer
        return results

    def apply(self, results, magnitude: int = None):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob},magnitude={self.magnitude},max_magnitude={self.magnitude_limit},random_magnitude={self.random_magnitude})"

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                prob=1.0,
                random_magnitude=False,
                record=False,
                magnitude=self.magnitude,
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Identity(RandAug):
    def apply(self, results, magnitude: int = None):
        return results


@PIPELINES.register_module()
class AutoContrast(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            results[key] = rgb2bgr(
                np.asarray(ImageOps.autocontrast(Image.fromarray(img)), dtype=img.dtype)
            )
        return results


@PIPELINES.register_module()
class RandEqualize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            results[key] = rgb2bgr(
                np.asarray(ImageOps.equalize(Image.fromarray(img)), dtype=img.dtype)
            )
        return results


@PIPELINES.register_module()
class RandSolarize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.solarize(
                img, min(int_parameter(magnitude, 256, self.magnitude_limit), 255)
            )
        return results


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of
    PIL."""

    def impl(pil_img, level, max_level=None):
        v = float_parameter(level, 1.8, max_level) + 0.1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


class RandEnhance(RandAug):
    op = None

    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])

            results[key] = rgb2bgr(
                np.asarray(
                    _enhancer_impl(self.op)(
                        Image.fromarray(img), magnitude, self.magnitude_limit
                    ),
                    dtype=img.dtype,
                )
            )
        return results


@PIPELINES.register_module()
class RandColor(RandEnhance):
    op = ImageEnhance.Color


@PIPELINES.register_module()
class RandContrast(RandEnhance):
    op = ImageEnhance.Contrast


@PIPELINES.register_module()
class RandBrightness(RandEnhance):
    op = ImageEnhance.Brightness


@PIPELINES.register_module()
class RandSharpness(RandEnhance):
    op = ImageEnhance.Sharpness


@PIPELINES.register_module()
class RandPosterize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            magnitude = int_parameter(magnitude, 4, self.magnitude_limit)
            results[key] = rgb2bgr(
                np.asarray(
                    ImageOps.posterize(Image.fromarray(img), 4 - magnitude),
                    dtype=img.dtype,
                )
            )
        return results


@PIPELINES.register_module()
class Sequential(BaseCompose):
    def __init__(self, transforms, record: bool = False):
        super().__init__(transforms)
        self.record = record
        self.enable_record(record)

    def enable_record(self, mode: bool = True):
        # enable children to record
        self.record = mode
        for transform in self.transforms:
            transform.enable_record(mode)


@PIPELINES.register_module()
class OneOf(Sequential):
    def __init__(self, transforms, record: bool = False):
        self.transforms = []
        for trans in transforms:
            if isinstance(trans, list):
                self.transforms.append(Sequential(trans))
            else:
                assert isinstance(trans, dict)
                self.transforms.append(Sequential([trans]))
        self.enable_record(record)

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)


@PIPELINES.register_module()
class ShuffledSequential(Sequential):
    def __call__(self, data):
        order = np.random.permutation(len(self.transforms))
        for idx in order:
            t = self.transforms[idx]
            data = t(data)
            if data is None:
                return None
        return data


"""
Geometric Augmentation. Modified from thirdparty/mmdetection/mmdet/datasets/pipelines/auto_augment.py
"""


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {"gt_bboxes": "gt_labels", "gt_bboxes_ignore": "gt_labels_ignore"}
    bbox2mask = {"gt_bboxes": "gt_masks", "gt_bboxes_ignore": "gt_masks_ignore"}
    bbox2seg = {
        "gt_bboxes": "gt_semantic_seg",
    }
    return bbox2label, bbox2mask, bbox2seg


class GeometricAugmentation(object):
    def __init__(
        self,
        img_fill_val=125,
        seg_ignore_label=255,
        min_size=0,
        prob: float = 1.0,
        random_magnitude: bool = True,
        record: bool = False,
    ):
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, "img_fill_val as tuple must have 3 elements."
            img_fill_val = tuple([float(val) for val in img_fill_val])
        assert np.all(
            [0 <= val <= 255 for val in img_fill_val]
        ), "all elements of img_fill_val should between range [0,255]."
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.min_size = min_size
        self.prob = prob
        self.random_magnitude = random_magnitude
        self.record = record

    def __call__(self, results):
        if np.random.random() < self.prob:
            magnitude: dict = self.get_magnitude(results)
            if self.record:
                if "aug_info" not in results:
                    results["aug_info"] = []
                results["aug_info"].append(self.get_aug_info(**magnitude))
            results = self.apply(results, **magnitude)
            self._filter_invalid(results, min_size=self.min_size)
        return results

    def get_magnitude(self, results) -> dict:
        raise NotImplementedError()

    def apply(self, results, **kwargs):
        raise NotImplementedError()

    def enable_record(self, mode: bool = True):
        self.record = mode

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                # make op deterministic
                prob=1.0,
                random_magnitude=False,
                record=False,
                img_fill_val=self.img_fill_val,
                seg_ignore_label=self.seg_ignore_label,
                min_size=self.min_size,
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image."""
        if min_size is None:
            return results
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            bbox_w = results[key][:, 2] - results[key][:, 0]
            bbox_h = results[key][:, 3] - results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]
        return results

    def __repr__(self):
        return f"""{self.__class__.__name__}(
        img_fill_val={self.img_fill_val},
        seg_ignore_label={self.seg_ignore_label},
        min_size={self.magnitude},
        prob: float = {self.prob},
        random_magnitude: bool = {self.random_magnitude},
        )"""


@PIPELINES.register_module()
class RandTranslate(GeometricAugmentation):
    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self, results, x=None, y=None):
        # ratio to pixel
        h, w, c = results["img_shape"]
        if x is not None:
            x = w * x
        if y is not None:
            y = h * y
        if x is not None:
            # translate horizontally
            self._translate(results, x)
        if y is not None:
            # translate veritically
            self._translate(results, y, direction="vertical")
        return results

    def _translate(self, results, offset, direction="horizontal"):
        if self.record:
            GTrans.apply(
                results,
                "shift",
                dx=offset if direction == "horizontal" else 0,
                dy=offset if direction == "vertical" else 0,
            )
        self._translate_img(results, offset, direction=direction)
        self._translate_bboxes(results, offset, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._translate_masks(results, offset, direction=direction)
        self._translate_seg(
            results, offset, fill_val=self.seg_ignore_label, direction=direction
        )

    def _translate_img(self, results, offset, direction="horizontal"):
        for key in results.get("img_fields", ["img"]):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(
                img, offset, direction, self.img_fill_val
            ).astype(img.dtype)

    def _translate_bboxes(self, results, offset, direction="horizontal"):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = results["img_shape"]
        for key in results.get("bbox_fields", []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            if direction == "horizontal":
                min_x = np.maximum(0, min_x + offset)
                max_x = np.minimum(w, max_x + offset)
            elif direction == "vertical":
                min_y = np.maximum(0, min_y + offset)
                max_y = np.minimum(h, max_y + offset)

            # the boxes translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([min_x, min_y, max_x, max_y], axis=-1)

    def _translate_masks(self, results, offset, direction="horizontal", fill_val=0):
        """Translate masks horizontally or vertically."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, fill_val)

    def _translate_seg(self, results, offset, direction="horizontal", fill_val=255):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            results[key] = mmcv.imtranslate(seg, offset, direction, fill_val).astype(
                seg.dtype
            )

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x={self.x}", f"y={self.y}"]
            + repr_str.split("\n")[-1:]
        )


@PIPELINES.register_module()
class RandRotate(GeometricAugmentation):
    def __init__(self, angle=None, center=None, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle
        self.center = center
        self.scale = scale
        if self.angle is None:
            self.prob = 0.0

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.angle, (list, tuple)):
                assert len(self.angle) == 2
                angle = (
                    np.random.random() * (self.angle[1] - self.angle[0]) + self.angle[0]
                )
                magnitude["angle"] = angle
        else:
            if self.angle is not None:
                assert isinstance(self.angle, (int, float))
                magnitude["angle"] = self.angle

        return magnitude

    def apply(self, results, angle: float = None):
        h, w = results["img"].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale).astype(np.double)
        if self.record:
            GTrans.apply(results, "rotate", cv2_rotation_matrix=rotate_matrix)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label
        )
        return results

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(
                img, angle, center, scale, border_value=self.img_fill_val
            )
            results[key] = img_rotated.astype(img.dtype)

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results["img_shape"]
        for key in results.get("bbox_fields", []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            coordinates = np.stack(
                [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            )  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (
                    coordinates,
                    np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype),
                ),
                axis=1,
            )  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            min_x, min_y = (
                np.min(rotated_coords[:, :, 0], axis=1),
                np.min(rotated_coords[:, :, 1], axis=1),
            )
            max_x, max_y = (
                np.max(rotated_coords[:, :, 0], axis=1),
                np.max(rotated_coords[:, :, 1], axis=1),
            )
            min_x, min_y = (
                np.clip(min_x, a_min=0, a_max=w),
                np.clip(min_y, a_min=0, a_max=h),
            )
            max_x, max_y = (
                np.clip(max_x, a_min=min_x, a_max=w),
                np.clip(max_y, a_min=min_y, a_max=h),
            )
            results[key] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
                results[key].dtype
            )

    def _rotate_masks(self, results, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the masks."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self, results, angle, center=None, scale=1.0, fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            results[key] = mmcv.imrotate(
                seg, angle, center, scale, border_value=fill_val
            ).astype(seg.dtype)

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"angle={self.angle}", f"center={self.center}", f"scale={self.scale}"]
            + repr_str.split("\n")[-1:]
        )


@PIPELINES.register_module()
class RandShear(GeometricAugmentation):
    def __init__(self, x=None, y=None, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.interpolation = interpolation
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self, results, x=None, y=None):
        if x is not None:
            # translate horizontally
            self._shear(results, np.tanh(-x * np.pi / 180))
        if y is not None:
            # translate veritically
            self._shear(results, np.tanh(y * np.pi / 180), direction="vertical")
        return results

    def _shear(self, results, magnitude, direction="horizontal"):
        if self.record:
            GTrans.apply(results, "shear", magnitude=magnitude, direction=direction)
        self._shear_img(results, magnitude, direction, interpolation=self.interpolation)
        self._shear_bboxes(results, magnitude, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._shear_masks(
            results, magnitude, direction=direction, interpolation=self.interpolation
        )
        self._shear_seg(
            results,
            magnitude,
            direction=direction,
            interpolation=self.interpolation,
            fill_val=self.seg_ignore_label,
        )

    def _shear_img(
        self, results, magnitude, direction="horizontal", interpolation="bilinear"
    ):
        """Shear the image.

        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`mmcv.imshear`.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_sheared = mmcv.imshear(
                img,
                magnitude,
                direction,
                border_value=self.img_fill_val,
                interpolation=interpolation,
            )
            results[key] = img_sheared.astype(img.dtype)

    def _shear_bboxes(self, results, magnitude, direction="horizontal"):
        """Shear the bboxes."""
        h, w, c = results["img_shape"]
        if direction == "horizontal":
            shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(
                np.float32
            )  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
        for key in results.get("bbox_fields", []):
            min_x, min_y, max_x, max_y = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            coordinates = np.stack(
                [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            )  # [4, 2, nb_box, 1]
            coordinates = (
                coordinates[..., 0].transpose((2, 1, 0)).astype(np.float32)
            )  # [nb_box, 2, 4]
            new_coords = np.matmul(
                shear_matrix[None, :, :], coordinates
            )  # [nb_box, 2, 4]
            min_x = np.min(new_coords[:, 0, :], axis=-1)
            min_y = np.min(new_coords[:, 1, :], axis=-1)
            max_x = np.max(new_coords[:, 0, :], axis=-1)
            max_y = np.max(new_coords[:, 1, :], axis=-1)
            min_x = np.clip(min_x, a_min=0, a_max=w)
            min_y = np.clip(min_y, a_min=0, a_max=h)
            max_x = np.clip(max_x, a_min=min_x, a_max=w)
            max_y = np.clip(max_y, a_min=min_y, a_max=h)
            results[key] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
                results[key].dtype
            )

    def _shear_masks(
        self,
        results,
        magnitude,
        direction="horizontal",
        fill_val=0,
        interpolation="bilinear",
    ):
        """Shear the masks."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.shear(
                (h, w),
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            )

    def _shear_seg(
        self,
        results,
        magnitude,
        direction="horizontal",
        fill_val=255,
        interpolation="bilinear",
    ):
        """Shear the segmentation maps."""
        for key in results.get("seg_fields", []):
            seg = results[key]
            results[key] = mmcv.imshear(
                seg,
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            ).astype(seg.dtype)

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x_magnitude={self.x}", f"y_magnitude={self.y}"]
            + repr_str.split("\n")[-1:]
        )


@PIPELINES.register_module()
class RandErase(GeometricAugmentation):
    def __init__(
        self,
        n_iterations=None,
        size=None,
        squared: bool = True,
        patches=None,
        **kwargs,
    ):
        kwargs.update(min_size=None)
        super().__init__(**kwargs)
        self.n_iterations = n_iterations
        self.size = size
        self.squared = squared
        self.patches = patches

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            n_iterations = self._get_erase_cycle()
            patches = []
            h, w, c = results["img_shape"]
            for i in range(n_iterations):
                # random sample patch size in the image
                ph, pw = self._get_patch_size(h, w)
                # random sample patch left top in the image
                px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
                patches.append([px, py, px + pw, py + ph])
            magnitude["patches"] = patches
        else:
            assert self.patches is not None
            magnitude["patches"] = self.patches

        return magnitude

    def _get_erase_cycle(self):
        if isinstance(self.n_iterations, int):
            n_iterations = self.n_iterations
        else:
            assert (
                isinstance(self.n_iterations, (tuple, list))
                and len(self.n_iterations) == 2
            )
            n_iterations = np.random.randint(*self.n_iterations)
        return n_iterations

    def _get_patch_size(self, h, w):
        if isinstance(self.size, float):
            assert 0 < self.size < 1
            return int(self.size * h), int(self.size * w)
        else:
            assert isinstance(self.size, (tuple, list))
            assert len(self.size) == 2
            assert 0 <= self.size[0] < 1 and 0 <= self.size[1] < 1
            w_ratio = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
            h_ratio = w_ratio

            if not self.squared:
                h_ratio = (
                    np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
                )
            return int(h_ratio * h), int(w_ratio * w)

    def apply(self, results, patches: list):
        for patch in patches:
            self._erase_image(results, patch, fill_val=self.img_fill_val)
            self._erase_mask(results, patch)
            self._erase_seg(results, patch, fill_val=self.seg_ignore_label)
        return results

    def _erase_image(self, results, patch, fill_val=128):
        for key in results.get("img_fields", ["img"]):
            tmp = results[key].copy()
            x1, y1, x2, y2 = patch
            tmp[y1:y2, x1:x2, :] = fill_val
            results[key] = tmp

    def _erase_mask(self, results, patch, fill_val=0):
        for key in results.get("mask_fields", []):
            masks = results[key]
            if isinstance(masks, PolygonMasks):
                # convert mask to bitmask
                masks = masks.to_bitmap()
            x1, y1, x2, y2 = patch
            tmp = masks.masks.copy()
            tmp[:, y1:y2, x1:x2] = fill_val
            masks = BitmapMasks(tmp, masks.height, masks.width)
            results[key] = masks

    def _erase_seg(self, results, patch, fill_val=0):
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            x1, y1, x2, y2 = patch
            seg[y1:y2, x1:x2] = fill_val
            results[key] = seg


@PIPELINES.register_module()
class RecomputeBox(object):
    def __init__(self, record=False):
        self.record = record

    def __call__(self, results):
        if self.record:
            if "aug_info" not in results:
                results["aug_info"] = []
            results["aug_info"].append(dict(type="RecomputeBox"))
        _, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                masks = results[mask_key]
                results[key] = self._recompute_bbox(masks)
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode

    def _recompute_bbox(self, masks):
        boxes = np.zeros(masks.masks.shape[0], 4, dtype=np.float32)
        x_any = np.any(masks.masks, axis=1)
        y_any = np.any(masks.masks, axis=2)
        for idx in range(masks.masks.shape[0]):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = np.array(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32
                )
        return boxes


# TODO: Implement Augmentation Inside Box
@PIPELINES.register_module()
class RandCrop(transforms.RandomCrop):
    def __init__(self, record=False, **kwargs):
        super().__init__(**kwargs)
        self.record = record

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.apply == False:
                crop_size = (img.shape[0], img.shape[1])
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape
        results['bbox_offset'] = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
        results['crop_size'] = crop_size

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results

    def __call__(self, results):
        results = super().__call__(results)
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class CellFormatBundle(DefaultFormatBundle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels', 'gt_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        if 'ori_img' in results:
            img = results['ori_img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['ori_img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        return results


@PIPELINES.register_module()
class AddOriImg(object):
    def __init__(self):
        pass

    def __call__(self, results):
        results['ori_img'] = results['img']
        return results



@PIPELINES.register_module()
class AddOriImg_Resize(object):
    def __init__(self):
        pass

    def __call__(self, results):
        results['ori_img'] = mmcv.imrescale(results['img'], (250, 250))
        return results


@PIPELINES.register_module()
class AddPointsAttr(object):
    def __init__(self):
        pass

    def __call__(self, results):
        img_shape = results['img_shape']
        results['gt_points'] = np.stack([(results['gt_bboxes'][:,0]+results['gt_bboxes'][:,2])/2, (results['gt_bboxes'][:,1]+results['gt_bboxes'][:,3])/2 ], axis=1)
        results['gt_points'][:, 0] = np.clip(results['gt_points'][:, 0], 0, img_shape[1])
        results['gt_points'][:, 1] = np.clip(results['gt_points'][:, 1], 0, img_shape[0])
        return results


@PIPELINES.register_module()
class AddPointsMapAttr(object):
    def __init__(self):
        pass

    def __call__(self, results):
        img_shape = results['img_shape']
        point_map = np.zeros(img_shape[:2])

        results['gt_points'] = np.stack([(results['gt_bboxes'][:,0]+results['gt_bboxes'][:,2])/2, (results['gt_bboxes'][:,1]+results['gt_bboxes'][:,3])/2 ], axis=1)
        results['gt_points'][:, 0] = np.clip(results['gt_points'][:, 0], 0, img_shape[1])
        results['gt_points'][:, 1] = np.clip(results['gt_points'][:, 1], 0, img_shape[0])
        return results





@PIPELINES.register_module()
class RandResize(transforms.Resize):
    def __init__(self, record=False, **kwargs):
        super().__init__(**kwargs)
        self.record = record

    def __call__(self, results):
        results = super().__call__(results)
        if self.record:
            scale_factor = results["scale_factor"]
            GTrans.apply(results, "scale", sx=scale_factor[0], sy=scale_factor[1])

            if "aug_info" not in results:
                results["aug_info"] = []
            new_h, new_w = results["img"].shape[:2]
            results["aug_info"].append(
                dict(
                    type=self.__class__.__name__,
                    record=False,
                    img_scale=(new_w, new_h),
                    keep_ratio=False,
                    bbox_clip_border=self.bbox_clip_border,
                    backend=self.backend,
                )
            )
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class RandTransform(object):
    def __init__(self):
        pass

    def __call__(self, results):
        GTrans.apply(
            results,
            "randt"
        )
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class RandFlip(transforms.RandomFlip):
    def __init__(self, record=False, **kwargs):
        super().__init__(**kwargs)
        self.record = record

    def __call__(self, results):
        results = super().__call__(results)
        if self.record:
            if "aug_info" not in results:
                results["aug_info"] = []
            if results["flip"]:
                GTrans.apply(
                    results,
                    "flip",
                    direction=results["flip_direction"],
                    shape=results["img_shape"][:2],
                )
                results["aug_info"].append(
                    dict(
                        type=self.__class__.__name__,
                        record=False,
                        flip_ratio=1.0,
                        direction=results["flip_direction"],
                    )
                )
            else:
                results["aug_info"].append(
                    dict(
                        type=self.__class__.__name__,
                        record=False,
                        flip_ratio=0.0,
                        direction="vertical",
                    )
                )
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(results))
            if res is None:
                return None
            # res["img_metas"]["tag"] = k
            multi_results.append(res)
        return multi_results


@PIPELINES.register_module()
class Gaussian_Blur(object):
    def __init__(self, max_ksize=3):
        self.max_ksize = max_ksize

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]

            ksize = np.random.randint(0, self.max_ksize, size=(2, ))
            ksize = tuple((ksize * 2 + 1).tolist())
            ret = cv2.GaussianBlur(
                img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
            )
            ret = np.reshape(ret, img.shape)
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Median_Blur(object):
    def __init__(self, max_ksize=3):
        self.max_ksize = max_ksize

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]

            ksize = np.random.randint(0, self.max_ksize)
            ksize = ksize * 2 + 1
            ret = cv2.medianBlur(
                img, ksize
            )
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class AdditiveGaussianNose(object):
    def __init__(self, loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5):
        self.loc = loc
        self.scale = scale
        self.per_channel = per_channel

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            ret = iaa.AdditiveGaussianNoise(
                            loc=self.loc, scale=self.scale, per_channel=self.per_channel
                        ).augment_image(img)
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Add_To_Hue(object):
    def __init__(self, range=None):
        self.range = range

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            hue = np.random.uniform(*self.range)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            if hsv.dtype.itemsize == 1:
                # OpenCV uses 0-179 for 8-bit images
                hsv[..., 0] = (hsv[..., 0] + hue) % 180
            else:
                # OpenCV uses 0-360 for floating point images
                hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
            ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Add_To_Saturation(object):
    def __init__(self, range=None):
        self.range = range

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            value = 1 + np.random.uniform(*self.range)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
            ret = np.clip(ret, 0, 255)
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Add_To_Contrast(object):
    def __init__(self, range=None):
        self.range = range

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            value = np.random.uniform(*self.range)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            ret = img * value + mean * (1 - value)
            ret = np.clip(ret, 0, 255)
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode


@PIPELINES.register_module()
class Add_To_Brightness(object):
    def __init__(self, range=None):
        self.range = range

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            value = np.random.uniform(*self.range)
            ret = np.clip(img + value, 0, 255)
            ret = ret.astype(img.dtype)
            results[key] = ret
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode




'''
import matplotlib.pyplot as plt
import torch
global_img = res['img']
plt.imshow(global_img.data.permute((1,2,0))/255.)
plt.show()
global_gt = res['gt_points']
glabal_res = res

plt.imshow(res['img'].data.permute((1,2,0))/255.)
plt.scatter(res['gt_points'].data[:,0], res['gt_points'].data[:,1],s=1,color=(1,0,0))
plt.show()

crop_size = res["img_metas"].data["crop_size"]
img_shape = res['img_metas'].data['img_shape']
offset = res["img_metas"].data["bbox_offset"]
crop_x1, crop_x2 = int(offset[0]), int(offset[0] + crop_size[1])
crop_y1, crop_y2 = int(offset[1]), int(offset[1] + crop_size[0])
crop_img = global_img.data[:, crop_y1:crop_y2, crop_x1:crop_x2]
plt.imshow(crop_img.permute((1,2,0))/255.)
plt.show()

valid_inds = (global_gt.data[:,0] >= offset[0]) & (global_gt.data[:,1] >= offset[1]) & (global_gt.data[:,0] <= offset[0]+crop_size[1]) & (global_gt.data[:,1] <= offset[1]+crop_size[0])
crop_gt = global_gt.data[valid_inds,:]
local_gt = res['gt_points'].data
crop_gt = crop_gt - offset[:2]
plt.imshow(global_img.data.permute((1,2,0))/255.)
plt.scatter(crop_gt[:,0], crop_gt[:,1],s=1,color=(1,0,0))
plt.show()


import torch
trans_matrix = [torch.from_numpy(res['img_metas'].data["transform_matrix"])]
Ori_matrix = [torch.from_numpy(np.float32([[1,0,0],[0,1,0],[0,0,1]]))]

def _get_trans_mat(a, b):
    return [bt @ at.inverse() for bt, at in zip(b, a)]

M = _get_trans_mat(Ori_matrix, trans_matrix)
points = torch.cat([crop_gt, crop_gt.new_ones(crop_gt.shape[0], 1)], dim=1)
points = torch.matmul(M[0], points.t()).t()
points = points[:, :2] / points[:, 2:3]

plt.imshow(res['img'].data.permute((1,2,0))/255.)
plt.scatter(points[:,0], points[:,1],s=1,color=(1,0,0))
plt.show()
'''