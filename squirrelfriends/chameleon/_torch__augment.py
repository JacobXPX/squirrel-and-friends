import albumentations as aug
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(in_size=[1024, 1024],
                   out_size=[512, 512],
                   prob=0.5,
                   level="medium",
                   add_cutout=True,
                   to_tensorv2=True,
                   bbox_params=None,
                   keypoint_params=None,
                   additional_targets=None):
    """
    Get a common Compose transforms for training images.
    For bbox_params, keypoint_params, additional_targets arguments,
    Refering https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        prob (float): probability of applying all list of transforms. Default: 0.5
        level (str): light, medium, heavy, default medium
        add_cutout (boolean): if adding cutout on image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.
        bbox_params (BboxParams): Parameters for bounding boxes transforms.
        keypoint_params (KeypointParams): Parameters for keypoints transforms.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {"image2": "image"}.

    Returns:
        phase_transforms (dict(Compose)): Compose transforms for train / valid / test images.
    """

    phase_transforms = {}
    phase_transforms["train"] = get_train_transforms(
        in_size=in_size,
        out_size=out_size,
        prob=prob,
        level=level,
        add_cutout=add_cutout,
        to_tensorv2=to_tensorv2,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets
    )
    phase_transforms["valid"] = get_valid_transforms(
        in_size=in_size,
        out_size=out_size,
        to_tensorv2=to_tensorv2,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets
    )
    phase_transforms["test"] = get_test_transforms(
        in_size=in_size,
        out_size=out_size,
        to_tensorv2=to_tensorv2
    )
    return phase_transforms


def get_train_transforms(in_size=[1024, 1024],
                         out_size=[512, 512],
                         prob=0.5,
                         level="medium",
                         add_cutout=True,
                         to_tensorv2=True,
                         bbox_params=None,
                         keypoint_params=None,
                         additional_targets=None
                         ):
    """
    Get a common Compose transforms for training images.
    For bbox_params, keypoint_params, additional_targets arguments,
    Refering https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        prob (float): probability of applying all list of transforms. Default: 0.5
        level (str): light, medium, heavy, default medium
        add_cutout (boolean): if adding cutout on image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.
        bbox_params (BboxParams): Parameters for bounding boxes transforms.
        keypoint_params (KeypointParams): Parameters for keypoints transforms.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {"image2": "image"}.

    Returns:
        aug_compose (Compose): common Compose transforms for train images.
    """

    if level == "light":
        level_ratio = 0.5
    elif level == "medium":
        level_ratio = 1
    elif level == "heavy":
        level_ratio = 2
    crop = [
        aug.RandomSizedCrop(
            min_max_height=(
                int(in_size[1] * (1.0 - level_ratio * 0.25)), int(in_size[1] * 1.0)),
            height=in_size[0],
            width=in_size[1],
            p=prob
        ),
    ]
    RGB = [
        aug.OneOf([
            aug.RandomBrightnessContrast(
                brightness_limit=0.2 * level_ratio,
                contrast_limit=0.2 * level_ratio,
                p=prob
            ),
            aug.RandomContrast(
                limit=0.2 * level_ratio,
                p=prob
            ),
        ],
            p=prob
        ),
        aug.OneOf([
            aug.RGBShift(
                r_shift_limit=int(20 * level_ratio),
                g_shift_limit=int(20 * level_ratio),
                b_shift_limit=int(20 * level_ratio),
                p=prob
            ),
            aug.HueSaturationValue(
                hue_shift_limit=int(20 * level_ratio),
                sat_shift_limit=int(20 * level_ratio),
                val_shift_limit=int(20 * level_ratio),
                p=prob
            ),
        ],
            p=prob
        ),
        aug.OneOf([
            aug.RandomGamma(
                gamma_limit=(int(80 * level_ratio), int(120 * level_ratio)),
                p=prob
            ),
        ],
            p=prob
        ),
        aug.OneOf([
            aug.MotionBlur(
                blur_limit=int(7 * level_ratio),
                p=prob
            ),
            aug.GaussianBlur(
                blur_limit=int(7 * level_ratio),
                p=prob
            ),
        ],
            p=prob
        ),
        aug.GaussNoise(
            var_limit=(10.0 * level_ratio, 50.0 * level_ratio),
            p=prob
        ),
    ]
    rotate = [
        aug.OneOf([
            aug.HorizontalFlip(p=prob),
            aug.VerticalFlip(p=prob),
            aug.RandomRotate90(p=prob)
        ],
            p=prob
        ),
    ]

    compose_lst = [
        *crop,
        *RGB,
        *rotate,
        aug.Resize(
            height=out_size[0],
            width=out_size[1],
            p=1.0
        )
    ]
    if add_cutout:
        compose_lst.append(
            aug.Cutout(
                num_holes=8,
                max_h_size=int(out_size[1]//16),
                max_w_size=int(out_size[0]//16),
                fill_value=0,
                p=prob
            )
        )
    if to_tensorv2:
        compose_lst.append(ToTensorV2(p=1.0))

    aug_compose = aug.Compose(
        compose_lst,
        p=1.0,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets,
    )

    return aug_compose


def get_valid_transforms(in_size=[1024, 1024],
                         out_size=[512, 512],
                         to_tensorv2=True,
                         bbox_params=None,
                         keypoint_params=None,
                         additional_targets=None
                         ):
    """
    Get a common Compose transforms for training images.
    For bbox_params, keypoint_params, additional_targets arguments,
    Refering https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.
        bbox_params (BboxParams): Parameters for bounding boxes transforms.
        keypoint_params (KeypointParams): Parameters for keypoints transforms.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {"image2": "image"}.

    Returns:
        aug_compose (Compose): common Compose transforms for valid images.
    """

    compose_lst = [
        aug.Resize(
            height=out_size[0],
            width=out_size[1],
            p=1.0
        ),
    ]
    if to_tensorv2:
        compose_lst.append(ToTensorV2(p=1.0))

    aug_compose = aug.Compose(
        compose_lst,
        p=1.0,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets,
    )

    return aug_compose


def get_test_transforms(in_size=[1024, 1024],
                        out_size=[512, 512],
                        to_tensorv2=True
                        ):
    """
    Get a common Compose transforms for training images.

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.

    Returns:
        aug_compose (Compose): common Compose transforms for test images.
    """

    compose_lst = [
        aug.Resize(
            height=out_size[0],
            width=out_size[1],
            p=1.0
        ),
    ]
    if to_tensorv2:
        compose_lst.append(ToTensorV2(p=1.0))

    aug_compose = aug.Compose(
        compose_lst,
        p=1.0
    )

    return aug_compose
